#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void emblayer_vec_scatter_kernel(
    const scalar_t* __restrict__ vec_values,
    const int64_t* __restrict__ prefix,
    const int64_t* __restrict__ feat_indices,
    scalar_t* __restrict__ output,
    int64_t batch_size,
    int64_t output_dim,
    int64_t nnz) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= nnz) {
    return;
  }

  // Find row by binary search over prefix.
  int64_t left = 0;
  int64_t right = batch_size;
  while (left < right) {
    int64_t mid = (left + right) >> 1;
    if (prefix[mid + 1] <= idx) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }
  const int64_t row = left;
  const int64_t col = feat_indices[idx];

  output[row * output_dim + col] = vec_values[idx];
}

torch::Tensor emblayer_vec_cuda(
    torch::Tensor vec_values,
    torch::Tensor prefix,
    torch::Tensor feat_indices,
    int64_t output_dim,
    double pad_value) {
  const int64_t batch_size = prefix.numel() - 1;
  const int64_t nnz = vec_values.numel();

  auto prefix_cuda = prefix.to(vec_values.device(), true);
  auto feat_cuda = feat_indices.to(vec_values.device(), true);

  auto output = torch::full({batch_size, output_dim}, pad_value, vec_values.options());

  const int threads = 256;
  const int blocks = static_cast<int>((nnz + threads - 1) / threads);

  AT_DISPATCH_ALL_TYPES(vec_values.scalar_type(), "emblayer_vec_cuda", ([&] {
    emblayer_vec_scatter_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
        vec_values.data_ptr<scalar_t>(),
        prefix_cuda.data_ptr<int64_t>(),
        feat_cuda.data_ptr<int64_t>(),
        output.data_ptr<scalar_t>(),
        batch_size,
        output_dim,
        nnz);
  }));

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
