#include <vector>
#include <cuda_fp16.h>

#include "../tester/utils.h"

#include <type_traits>
#include <cmath>

/**
 * @brief Computes the trace of a matrix.
 *
 * The trace of a matrix is defined as the sum of its diagonal elements.
 * This function expects a flattened row-major matrix stored in a
 * std::vector. If the matrix is not square, the trace will sum up
 * elements along the main diagonal up to the smaller of rows or cols.
 *
 * @tparam T The numeric type of matrix elements (e.g., float, int).
 * @param h_input A flattened matrix of size rows * cols.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @return The trace (sum of diagonal values) of the matrix.
 */

template <typename T>
__device__ inline void atomic_add_T(T* addr, T val);

template <>
__device__ inline void atomic_add_T<float>(float* addr, float val) { atomicAdd(addr, val); }

template <>
__device__ inline void atomic_add_T<int>(int* addr, int val) { atomicAdd(addr, val); }

template <typename T>
__device__ T block_reduce(T val, T* sdata) {
  size_t tid = threadIdx.x;

  // 每线程写入自己的局部和
  sdata[tid] = val;
  // 同步：确保所有线程都写完
  __syncthreads();

  // 树形归约：每轮减半
  // stride 初始为 blockDim/2
  size_t stride = blockDim.x / 2;
  while (stride > 0) {
    if (tid < stride) {
      sdata[tid] = sdata[tid] + sdata[tid + stride];
    }
    __syncthreads();
    stride = stride  / 2;
  }

  // 返回 block_sum，即 sdata[0]
  return sdata[0];
}

template <typename T>
__global__ void trace_diag_atomic(const T* d_diag, size_t n, T* d_out) {
  __shared__ T sdata[256];

  const size_t tid = threadIdx.x;
  const size_t idx = (size_t)blockIdx.x * blockDim.x + tid;
  const size_t stride = (size_t)blockDim.x * gridDim.x;

  T local = 0;
  for (size_t i = idx; i < n; i += stride) local += d_diag[i];

  const T block_sum = block_reduce(local, sdata);
  if (tid == 0) atomic_add_T<T>(d_out, block_sum);
}

template <typename T>
T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
  const size_t n = rows < cols ? rows : cols;
  if (n == 0) return T(0);

  // 1) host 上提取对角线（O(n)）
  std::vector<T> h_diag(n);
  for (size_t i = 0; i < n; ++i) {
    h_diag[i] = h_input[i * cols + i];
  }

  // 2) device 上只存 diag（n 个元素）
  T* d_diag = nullptr;
  T* d_out  = nullptr;
  cudaMalloc(&d_diag, n * sizeof(T));
  cudaMalloc(&d_out, sizeof(T));

  cudaMemcpy(d_diag, h_diag.data(), n * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemset(d_out, 0, sizeof(T));

  // 3) 单 kernel
  const int block = 256;
  int grid = (int)((n + block - 1) / block);
  if (grid > 1024) grid = 1024;
  if (grid < 1) grid = 1;

  trace_diag_atomic<T><<<grid, block>>>(d_diag, n, d_out);

  // 4) 拷回
  T h_out{};
  cudaMemcpy(&h_out, d_out, sizeof(T), cudaMemcpyDeviceToHost);

  cudaFree(d_diag);
  cudaFree(d_out);
  return h_out;
}

/**
 * @brief Computes flash attention for given query, key, and value tensors.
 * 
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length  
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */
template <typename T>
__global__ void cuda_flashAttention(
  const T* d_q, const T* d_k, const T* d_v, T* d_o, 
  int batch_size, int target_seq_len, int src_seq_len, int query_heads, int kv_heads, int head_dim, 
  bool is_causal
);

template <typename T>
__device__ float to_float(T x);

template <typename T>
__device__ T from_float(float x);

__device__ float block_reduce_max(float val, float* sdata);

__device__ float block_reduce_sum(float val, float* sdata);

__device__ float warp_reduce_max(float v);

__device__ float warp_reduce_sum(float v);

template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {     
  // 快速处理边界
  if (batch_size <= 0 || target_seq_len <= 0 || query_heads <= 0 || head_dim <= 0) {
    h_o.clear();
    return;
  }
  if (src_seq_len <= 0 || kv_heads <= 0) {
    // 没有 key/value：输出可以全 0
    h_o.assign((size_t)batch_size * target_seq_len * query_heads * head_dim, T(0));
    return;
  }                    
  
  // 计算输入的总元素数
  size_t total_qo_elems = batch_size * target_seq_len * query_heads * head_dim;
  size_t total_kv_elems = batch_size * src_seq_len * kv_heads * head_dim;

  size_t qo_bytes = total_qo_elems * sizeof(T);
  size_t kv_bytes = total_kv_elems * sizeof(T);

  // 尺寸检查
  if (h_q.size() != total_qo_elems || h_k.size() != total_kv_elems || h_v.size() != total_kv_elems) {
    printf("flashAttention input size mismatch\n");
    return;
  }

  // 如果 h_o 的大小不对，就调整到正确大小
  if (h_o.size() != total_qo_elems) {
    h_o.resize(total_qo_elems);
  }

  // 声明 device 输入与输出变量
  T* d_q;
  T* d_k;
  T* d_v;
  T* d_o;

  // 分配输入与输出内存
  cudaMalloc(&d_q, qo_bytes);
  cudaMalloc(&d_k, kv_bytes);
  cudaMalloc(&d_v, kv_bytes);
  cudaMalloc(&d_o, qo_bytes);

  // 拷贝输入至 device
  cudaMemcpy(d_q, h_q.data(), qo_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_k, h_k.data(), kv_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_v, h_v.data(), kv_bytes, cudaMemcpyHostToDevice);

  // 清零 d_o
  cudaMemset(d_o, 0, qo_bytes);

  // 设置 grid，block，shared memory
  dim3 grid(batch_size, target_seq_len, query_heads);
  dim3 block(32);
  size_t smem_size = 0;

  // launch kernel
  cuda_flashAttention<<<grid, block, smem_size>>>(
    d_q, d_k, d_v, d_o, 
    batch_size, target_seq_len, src_seq_len, query_heads, kv_heads, head_dim, 
    is_causal
  );

  // 从 device 写回结果
  cudaMemcpy(h_o.data(), d_o, qo_bytes, cudaMemcpyDeviceToHost);

  // 释放 cuda 内存
  cudaFree(d_q);
  cudaFree(d_k);
  cudaFree(d_v);
  cudaFree(d_o);

  return;
}

template <typename T>
__global__ void cuda_flashAttention(
  const T* d_q, const T* d_k, const T* d_v, T* d_o,
  int batch_size, int target_seq_len, int src_seq_len,
  int query_heads, int kv_heads, int head_dim, bool is_causal
) {
  // warp-only
  const int lane = threadIdx.x;
  if (lane >= 32) return;

  const int b  = (int)blockIdx.x;
  const int t  = (int)blockIdx.y;
  const int qh = (int)blockIdx.z;

  const int B  = batch_size;
  const int Tt = target_seq_len;
  const int Ss = src_seq_len;
  const int QH = query_heads;
  const int KH = kv_heads;
  const int D  = head_dim;

  if (b >= B || t >= Tt || qh >= QH) return;
  if (Ss <= 0 || D <= 0) return;

  // GQA 映射
  int kh;
  if (KH == 1) kh = 0;
  else if (QH % KH == 0) kh = qh / (QH / KH);
  else kh = qh % KH;
  kh = kh < KH ? kh : (KH - 1);

  // causal end
  const int s_end = is_causal ? (t < (Ss - 1) ? t : (Ss - 1)) : (Ss - 1);
  if (s_end < 0) return;

  const size_t q_base = ((size_t)(b * Tt + t) * (size_t)QH + (size_t)qh) * (size_t)D;
  const size_t o_base = q_base;

  const float scale = 1.0f / sqrtf((float)D);

  __shared__ float s_q[64];
  __shared__ float s_partial[32];
  for (int dd = lane; dd < D; dd += 32) {
    s_q[dd] = to_float(d_q[q_base + (size_t)dd]);
  }

  // m = max(score)
  float thread_max = -1e20f;
  for (int s = lane; s <= s_end; s += 32) {
    const size_t k_base =
        (((size_t)b * (size_t)Ss + (size_t)s) * (size_t)KH + (size_t)kh) * (size_t)D;

    float dot = 0.0f;
    #pragma unroll
    for (int dd = 0; dd < D; ++dd) {
      dot += s_q[dd] * to_float(d_k[k_base + (size_t)dd]);
    }
    const float score = dot * scale;
    thread_max = thread_max > score ? thread_max : score;
  }

  float m = warp_reduce_max(thread_max);
  m = __shfl_sync(0xffffffff, m, 0); // 广播 lane0

  // den = sum(exp(score - m))
  float thread_den = 0.0f;
  for (int s = lane; s <= s_end; s += 32) {
    const size_t k_base =
        (((size_t)b * (size_t)Ss + (size_t)s) * (size_t)KH + (size_t)kh) * (size_t)D;

    float dot = 0.0f;
    #pragma unroll
    for (int dd = 0; dd < D; ++dd) {
      dot += s_q[dd] * to_float(d_k[k_base + (size_t)dd]);
    }
    const float score = dot * scale;
    thread_den += expf(score - m);
  }

  float den = warp_reduce_sum(thread_den);
  den = __shfl_sync(0xffffffff, den, 0);

  // output 
  // 每个 lane 处理 d=lane，当 D > 32时 d=lane+32
  float acc0 = 0.0f;
  float acc1 = 0.0f;

  const int d0 = lane * 2;
  const int d1 = d0 + 1;

  for (int s = 0; s <= s_end; ++s) {
    const size_t k_base =
        (((size_t)b * (size_t)Ss + (size_t)s) * (size_t)KH + (size_t)kh) * (size_t)D;
    const size_t v_base = k_base;
    
    float w = 0.0f;

    if constexpr (std::is_same_v<T, float>) {
      // ===== 精度优先：float 走 lane0 串行 dd=0..D-1 =====
      if (lane == 0) {
        float dot = 0.0f;
        for (int dd = 0; dd < D; ++dd) {
          dot += s_q[dd] * d_k[k_base + (size_t)dd];  // float 直接读
        }
        const float score = dot * scale;
        w = expf(score - m) / den;
      }
      w = __shfl_sync(0xffffffff, w, 0);

    } else {
      // ===== 性能优先：half 走并行 partial + 固定顺序合并 =====
      float partial = 0.0f;
      for (int dd = lane; dd < D; dd += 32) {
        partial += s_q[dd] * to_float(d_k[k_base + (size_t)dd]); // half->float
      }
      s_partial[lane] = partial;
      __syncwarp();

      if (lane == 0) {
        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
          dot += s_partial[i];
        }
        const float score = dot * scale;
        w = expf(score - m) / den;
      }
      w = __shfl_sync(0xffffffff, w, 0);
    }
    // 所有 lane 都会执行到这里，w 已经广播为寄存器里的标量
    if constexpr (std::is_same_v<T, half>) {
      // half 路径：尽量 half2 load（要求 d0 是偶数——我们现在就是偶数）
      if (d0 + 1 < D) {
        // v_base + d0 是 half2 对齐的（cudaMalloc 对齐很大 + d0 偶数）
        const half2 hv2 = *reinterpret_cast<const half2*>(d_v + v_base + (size_t)d0);
        const float2 fv2 = __half22float2(hv2);
        acc0 += w * fv2.x;
        acc1 += w * fv2.y;
      } else if (d0 < D) {
        // D==1 或 D 为奇数时最后一个元素
        acc0 += w * __half2float(d_v[v_base + (size_t)d0]);
      }
    } else {
      // float 路径：float2 load（d0 偶数）
      if (d0 + 1 < D) {
        const float2 vv2 = *reinterpret_cast<const float2*>(d_v + v_base + (size_t)d0);
        acc0 += w * vv2.x;
        acc1 += w * vv2.y;
      } else if (d0 < D) {
        acc0 += w * d_v[v_base + (size_t)d0];
      }
    }
  }

  if constexpr (std::is_same_v<T, half>) {
    if (d0 + 1 < D) {
      const half2 ho2 = __floats2half2_rn(acc0, acc1);
      *reinterpret_cast<half2*>(d_o + o_base + (size_t)d0) = ho2;
    } else if (d0 < D) {
      d_o[o_base + (size_t)d0] = __float2half(acc0);
    }
  } else {
    if (d0 + 1 < D) {
      *reinterpret_cast<float2*>(d_o + o_base + (size_t)d0) = make_float2(acc0, acc1);
    } else if (d0 < D) {
      d_o[o_base + (size_t)d0] = acc0;
    }
  }
}

template <typename T>
__device__ float to_float(T x) {
  if constexpr (std::is_same_v<T, half>) {
    return __half2float(x);
  } else {
    return static_cast<float>(x);
  }
}

template <typename T>
__device__ T from_float(float x) {
  if constexpr (std::is_same_v<T, half>) {
    return __float2half(x);
  } else {
    return static_cast<T>(x);
  }
}

__device__ float block_reduce_max(float val, float* sdata) {
  // 声明 shared memory 缓冲区，shared 数组大小至少 blockDim.x
  size_t tid = threadIdx.x;

  // 每线程写入自己的局部和
  sdata[tid] = val;
  // 同步：确保所有线程都写完
  __syncthreads();

  // 树形归约：每轮减半
  // stride 初始为 blockDim/2
  size_t stride = blockDim.x / 2;
  while (stride > 0) {
    if (tid < stride) {
      sdata[tid] = sdata[tid] >= sdata[tid + stride] ? sdata[tid] : sdata[tid + stride];
    }
    __syncthreads();
    stride = stride  / 2;
  }

  // 返回 block_sum_max，即 sdata[0]
  return sdata[0];
}

__device__ float block_reduce_sum(float val, float* sdata) {
  size_t tid = threadIdx.x;

  // 每线程写入自己的局部和
  sdata[tid] = val;
  // 同步：确保所有线程都写完
  __syncthreads();

  // 树形归约：每轮减半
  // stride 初始为 blockDim/2
  size_t stride = blockDim.x / 2;
  while (stride > 0) {
    if (tid < stride) {
      sdata[tid] = sdata[tid] + sdata[tid + stride];
    }
    __syncthreads();
    stride = stride  / 2;
  }

  // 返回 block_sum，即 sdata[0]
  return sdata[0];
}

__device__ float warp_reduce_sum(float v) {
  // warpSize == 32
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffff, v, offset);
  }
  return v;
}

__device__ float warp_reduce_max(float v) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    float other = __shfl_down_sync(0xffffffff, v, offset);
    v = v > other ? v : other;
  }
  return v;
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template int trace<int>(const std::vector<int>&, size_t, size_t);
template float trace<float>(const std::vector<float>&, size_t, size_t);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);
