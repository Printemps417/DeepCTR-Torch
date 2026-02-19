#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void emblayer_seq_kernel(
    const scalar_t* __restrict__ seq_values,
    const int64_t* __restrict__ prefix,
    const int64_t* __restrict__ seq_lengths,
    const int64_t* __restrict__ seq_offsets,
    scalar_t* __restrict__ output,
    int64_t batch_size,
    int64_t max_seq_num,
    int64_t max_seq_len,
    int64_t total_elems) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total_elems) {
    return;
  }

  const int64_t span_per_batch = max_seq_num * max_seq_len;
  const int64_t batch_id = idx / span_per_batch;
  const int64_t rem = idx % span_per_batch;
  const int64_t seq_slot = rem / max_seq_len;
  const int64_t token_pos = rem % max_seq_len;

  const int64_t seq_begin = prefix[batch_id];
  const int64_t seq_end = prefix[batch_id + 1];
  const int64_t seq_count = seq_end - seq_begin;

  if (seq_slot >= seq_count) {
    return;
  }

  const int64_t seq_idx = seq_begin + seq_slot;
  const int64_t seq_len = seq_lengths[seq_idx];
  if (token_pos >= seq_len) {
    return;
  }

  const int64_t src_idx = seq_offsets[seq_idx] + token_pos;
  output[idx] = seq_values[src_idx];
}

torch::Tensor emblayer_seq_cuda(
    torch::Tensor seq_values,
    torch::Tensor prefix,
    torch::Tensor seq_lengths,
    torch::Tensor seq_offsets,
    int64_t max_seq_num,
    int64_t max_seq_len,
    int64_t pad_value) {
  const int64_t batch_size = prefix.numel() - 1;
  const int64_t total_elems = batch_size * max_seq_num * max_seq_len;

  auto prefix_cuda = prefix.to(seq_values.device(), true);
  auto lengths_cuda = seq_lengths.to(seq_values.device(), true);
  auto offsets_cuda = seq_offsets.to(seq_values.device(), true);

  auto output = torch::full(
      {batch_size, max_seq_num, max_seq_len},
      pad_value,
      seq_values.options());

  const int threads = 256;
  const int blocks = static_cast<int>((total_elems + threads - 1) / threads);

  AT_DISPATCH_ALL_TYPES(seq_values.scalar_type(), "emblayer_seq_cuda", ([&] {
    emblayer_seq_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
        seq_values.data_ptr<scalar_t>(),
        prefix_cuda.data_ptr<int64_t>(),
        lengths_cuda.data_ptr<int64_t>(),
        offsets_cuda.data_ptr<int64_t>(),
        output.data_ptr<scalar_t>(),
        batch_size,
        max_seq_num,
        max_seq_len,
        total_elems);
  }));

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
