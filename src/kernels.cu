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
T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
  const size_t n = rows < cols ? rows : cols;

  if (n == 0) {
    return T(0);
  } else {
    size_t bytes = rows * cols * sizeof(T);
    T* d_input;
    T* d_out;

    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_out, sizeof(T));

    cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice);

    size_t block = 256;
    size_t grid = (n + block -1) / block < 1024 ? (n + block -1) / block : 1024;
    size_t smem_size = block * sizeof(T);

    T* d_partials;
    cudaMalloc(&d_partials, grid * sizeof(T));
    cudaMemset(d_out, 0, sizeof(T));

    trace_cuda<<<grid, block, smem_size>>>(d_input, n, cols, d_partials);

    reduce_partias<<<1, block, smem_size>>>(d_partials, grid, d_out);

    T h_out = T(0);
    cudaMemcpy(&h_out, d_out, sizeof(T), cudaMemcpyDeviceToHost);

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

  T local_sum = 0;
  for (size_t i = idx; i < n; i += stride) {
    local_sum += d_input[i * cols + i];
  }

  T block_sum = block_reduce(local_sum, sdata);

  if (tid == 0) {
    d_partials[blockIdx.x] = block_sum;
  }
}

template <typename T>
__global__ void reduce_partias(const T* d_partials, size_t num_partials, T* d_out){
  __shared__ T sdata[256];
  size_t tid = threadIdx.x;

  T local_sum = 0;
  for (size_t i = tid; i < num_partials; i += blockDim.x) {
    local_sum += d_partials[i];
  }

  T block_sum = block_reduce(local_sum, sdata);

  if (tid == 0) {
    d_out[0] = block_sum;
  }
}

template <typename T>
__device__ T block_reduce(T val, T* sdata) {
  size_t tid = threadIdx.x;

  sdata[tid] = val;
  __syncthreads();

  size_t stride = blockDim.x / 2;

  while (stride > 0) {
    if (tid < stride) {
      sdata[tid] = sdata[tid] + sdata[tid + stride];
    }
    __syncthreads();
    stride = stride  / 2;
  }

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
