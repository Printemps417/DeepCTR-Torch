#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

template <typename scalar_t>
__global__ void multislice_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t* __restrict__ starts,
    const int64_t* __restrict__ offsets,
    int64_t in_cols,
    int64_t out_cols,
    int64_t num_slices,
    int64_t total_elems) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total_elems) {
    return;
  }

  const int64_t row = idx / out_cols;
  const int64_t out_col = idx % out_cols;

  int64_t slice_id = 0;
  while (slice_id + 1 < num_slices && out_col >= offsets[slice_id + 1]) {
    ++slice_id;
  }

  const int64_t local_col = out_col - offsets[slice_id];
  const int64_t in_col = starts[slice_id] + local_col;

  output[idx] = input[row * in_cols + in_col];
}

torch::Tensor multislice_cuda(
    torch::Tensor input,
    torch::Tensor starts_cpu,
    torch::Tensor lengths_cpu) {
  const auto batch = input.size(0);
  const auto in_cols = input.size(1);
  const auto num_slices = starts_cpu.size(0);

  auto starts_ptr = starts_cpu.data_ptr<int64_t>();
  auto lengths_ptr = lengths_cpu.data_ptr<int64_t>();

  std::vector<int64_t> offsets_host(num_slices + 1, 0);
  int64_t out_cols = 0;
  for (int64_t i = 0; i < num_slices; ++i) {
    out_cols += lengths_ptr[i];
    offsets_host[i + 1] = out_cols;
  }

  auto output = torch::empty({batch, out_cols}, input.options());

  auto starts = starts_cpu.to(input.device(), true);
  auto offsets = torch::from_blob(offsets_host.data(), {num_slices + 1}, torch::kInt64)
                     .clone()
                     .to(input.device(), true);

  const int64_t total_elems = batch * out_cols;
  const int threads = 256;
  const int blocks = static_cast<int>((total_elems + threads - 1) / threads);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "multislice_cuda", ([&] {
    multislice_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
        input.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        starts.data_ptr<int64_t>(),
        offsets.data_ptr<int64_t>(),
        in_cols,
        out_cols,
        num_slices,
        total_elems);
  }));

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
