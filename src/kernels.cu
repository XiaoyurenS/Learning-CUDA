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
__device__ T block_reduce(T val, T* sdata);

template <typename T>
__global__ void trace_cuda(const T* d_input, size_t n, size_t cols, T* d_partials);

template <typename T>
__global__ void reduce_partials(const T* d_partials, size_t num_partials, T* d_out);

template <typename T>
T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
  // 计算对角线元素个数
  const size_t n = rows < cols ? rows : cols;

  if (n == 0) {
    return T(0);
  } else {
    // 分配内存
    size_t bytes = rows * cols * sizeof(T); // 计算输入的内存总开销
    T* d_input; // 声明 device 输入
    T* d_out; // 声明 device 输出

    cudaMalloc(&d_input, bytes); // 分配输入内存
    cudaMalloc(&d_out, sizeof(T)); // 分配输出内存

    // 拷贝输入至 device
    cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice);

    // 设置 grid，block，shared memory
    size_t block = 256;
    size_t grid = (n + block -1) / block < 1024 ? (n + block -1) / block : 1024;
    size_t smem_size = block * sizeof(T);

    // 声明中间变量与内存分配
    T* d_partials;
    cudaMalloc(&d_partials, grid * sizeof(T));
    cudaMemset(d_out, 0, sizeof(T));

    // launch kernel
    trace_cuda<<<grid, block, smem_size>>>(d_input, n, cols, d_partials);

    reduce_partials<<<1, block, smem_size>>>(d_partials, grid, d_out);

    // 声明 host 结果，并从 device 写回结果
    T h_out = T(0);
    cudaMemcpy(&h_out, d_out, sizeof(T), cudaMemcpyDeviceToHost);

    // 释放 cuda 内存
    cudaFree(d_input);
    cudaFree(d_out);
    cudaFree(d_partials);

    return h_out;
  }
}

template <typename T>
__global__ void trace_cuda(const T* d_input, size_t n, size_t cols, T* d_partials) {
  __shared__ T sdata[256];
  size_t tid = threadIdx.x;
  size_t idx = blockIdx.x * blockDim.x + tid;
  size_t stride = blockDim.x * gridDim.x;

  // 每个线程：grid-stride 遍历 i：累加 d_input[i*cols+i] 到 local_sum
  T local_sum = 0;
  for (size_t i = idx; i < n; i += stride) {
    local_sum += d_input[i * cols + i];
  }

  // 每个 block：block_reduce(local_sum) 得到 block_sum
  T block_sum = block_reduce(local_sum, sdata);

  if (tid == 0) {
    d_partials[blockIdx.x] = block_sum;
  }
}

template <typename T>
__global__ void reduce_partials(const T* d_partials, size_t num_partials, T* d_out){
  __shared__ T sdata[256];
  size_t tid = threadIdx.x;

  // 每个线程：grid-stride 遍历 i：累加 partials[i] 到 local_sum
  T local_sum = 0;
  for (size_t i = tid; i < num_partials; i += blockDim.x) {
    local_sum += d_partials[i];
  }

  // 每个 block：block_reduce(local_sum) 得到 block_sum
  T block_sum = block_reduce(local_sum, sdata);

  if (tid == 0) {
    d_out[0] = block_sum;
  }
}

template <typename T>
__device__ T block_reduce(T val, T* sdata) {
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
      sdata[tid] = sdata[tid] + sdata[tid + stride];
    }
    __syncthreads();
    stride = stride  / 2;
  }

  // 返回 block_sum，即 sdata[0]
  return sdata[0];
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
  dim3 block(256);
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
  int batch_size, int target_seq_len, int src_seq_len, int query_heads, int kv_heads, int head_dim, 
  bool is_causal
) {
  __shared__ float s_reduce[256];
  __shared__ float s_m;
  __shared__ float s_den;

  // grid/block 映射
  int b = (int)blockIdx.x;
  int t = (int)blockIdx.y;
  int qh = (int)blockIdx.z;
  int tid = threadIdx.x;

  int B = batch_size;
  int Tt = target_seq_len;
  int Ss = src_seq_len;

  int QH = query_heads;
  int KH = kv_heads;
  int D = head_dim;

  if (b >= B || t >= Tt || qh >= QH) return;
  if (Ss <= 0 || D <= 0) return;

  // GQA 映射
  int kh;
  if (KH == 1) {
    kh = 0;
  } else if (QH % KH == 0) {
    size_t group = QH / KH;
    kh = qh / group;
  } else {
    kh = qh % KH;
  }
  kh = min(kh, KH - 1);

  int s_end = is_causal ? min(Ss - 1, t) : (Ss - 1);

  size_t q_base = ((b * Tt + t) * QH + qh) * D;
  size_t o_base = q_base;

  const float scale = 1.0f / sqrtf((float)D);

  // Step 1：计算 m = max score（block 内归约）
  float thread_max = -1e20f;

  for (size_t s = tid; s <= s_end; s += blockDim.x) {
    // 计算 dot(q, k_s)
    size_t k_base = ((b * Ss + s) * KH + kh) * D;
    float dot = 0.0f;
    for (size_t d = 0; d < D; ++d) {
      dot += to_float(d_q[q_base + d]) * to_float(d_k[k_base + d]);
    }
    float score = dot * scale;
    thread_max = thread_max >= score ? thread_max : score;
  }
  float m = block_reduce_max(thread_max, s_reduce);

  if (tid == 0) {
    s_m = m;
  }
  __syncthreads();
  m = s_m;

  // Step 2：计算 den = sum exp(score - m)（block 内归约）
  float thread_den = 0.0f;

  for (int s = tid; s <= s_end; s += blockDim.x) {
    size_t k_base = ((b * Ss + s) * KH + kh) * D;
    float dot = 0.0f;
    for (int d = 0; d < D; ++d) {
      dot += to_float(d_q[q_base + d]) * to_float(d_k[k_base + d]);
    }
    float score = dot * scale;
    thread_den += expf(score - m);
  }
  float den = block_reduce_sum(thread_den, s_reduce);

  if (tid == 0) {
    s_den = den;
  }
  __syncthreads();
  den = s_den;

  // Step 3：算输出向量 o[d]（每线程负责多个 d）
  // 每个线程先把自己负责的 out_accum[d] 初始化 0
  for (int d = tid; d < D; d += blockDim.x) {
    float acc = 0.0f;
    for (int s = 0; s <= s_end; ++s) {
      size_t k_base = ((b * Ss + s) * KH + kh) * D;
      size_t v_base = k_base;

      float dot = 0.0f;
      for (int dd = 0; dd < D; ++dd) {
        dot += to_float(d_q[q_base + dd]) * to_float(d_k[k_base + dd]);
      }
      float score = dot * scale;
      float w = expf(score - m) / den;

      acc += w * to_float(d_v[v_base + d]);
    }
    d_o[o_base + d] = from_float<T>(acc);
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
    return static_cast<float>(x);
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
      sdata[tid] = sdata[tid] + sdata[tid + stride];
    }
    __syncthreads();
    stride = stride  / 2;
  }

  // 返回 block_sum，即 sdata[0]
  return sdata[0];
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
