#include <vector>
#include <cuda_fp16.h>

#include "../tester/utils.h"

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

    // cuda launch
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
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {       
  // TODO: Implement the flash attention function
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
