#include <torch/extension.h>

#include <vector>

torch::Tensor multislice_cuda(
    torch::Tensor input,
    torch::Tensor starts,
    torch::Tensor lengths);

torch::Tensor multislice(
    torch::Tensor input,
    torch::Tensor starts,
    torch::Tensor lengths) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(input.dim() == 2, "input must be a 2D tensor [batch, cols]");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(starts.device().is_cpu(), "starts must be a CPU tensor");
  TORCH_CHECK(lengths.device().is_cpu(), "lengths must be a CPU tensor");
  TORCH_CHECK(starts.scalar_type() == torch::kInt64, "starts must be int64");
  TORCH_CHECK(lengths.scalar_type() == torch::kInt64, "lengths must be int64");
  TORCH_CHECK(starts.dim() == 1, "starts must be a 1D tensor");
  TORCH_CHECK(lengths.dim() == 1, "lengths must be a 1D tensor");
  TORCH_CHECK(starts.size(0) == lengths.size(0), "starts and lengths must have same size");
  TORCH_CHECK(starts.size(0) > 0, "at least one slice is required");

  auto starts_ptr = starts.data_ptr<int64_t>();
  auto lengths_ptr = lengths.data_ptr<int64_t>();
  const int64_t in_cols = input.size(1);
  for (int64_t i = 0; i < starts.size(0); ++i) {
    TORCH_CHECK(starts_ptr[i] >= 0, "starts must be non-negative");
    TORCH_CHECK(lengths_ptr[i] > 0, "lengths must be positive");
    TORCH_CHECK(starts_ptr[i] + lengths_ptr[i] <= in_cols,
                "slice range out of bounds for input cols");
  }

  return multislice_cuda(input, starts, lengths);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("multislice", &multislice, "MultiSlice CUDA op (concat selected ranges)");
}
