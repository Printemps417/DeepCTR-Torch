#include <torch/extension.h>

#include <vector>

torch::Tensor emblayer_vec_cuda(
    torch::Tensor vec_values,
    torch::Tensor prefix,
    torch::Tensor feat_indices,
    int64_t output_dim,
    double pad_value);

torch::Tensor emblayer_vec_fast(
    torch::Tensor vec_values,
    torch::Tensor prefix,
    torch::Tensor feat_indices,
    int64_t output_dim,
    double pad_value) {
  TORCH_CHECK(vec_values.is_cuda(), "vec_values must be CUDA");
  TORCH_CHECK(prefix.is_cuda(), "prefix must be CUDA");
  TORCH_CHECK(feat_indices.is_cuda(), "feat_indices must be CUDA");

  TORCH_CHECK(vec_values.dim() == 1, "vec_values must be 1D");
  TORCH_CHECK(prefix.dim() == 1, "prefix must be 1D");
  TORCH_CHECK(feat_indices.dim() == 1, "feat_indices must be 1D");

  TORCH_CHECK(prefix.scalar_type() == torch::kInt64, "prefix must be int64 in fast path");
  TORCH_CHECK(feat_indices.scalar_type() == torch::kInt64, "feat_indices must be int64 in fast path");
  TORCH_CHECK(feat_indices.numel() == vec_values.numel(), "feat_indices size must equal vec_values size");
  TORCH_CHECK(output_dim > 0, "output_dim must be > 0");

  TORCH_CHECK(prefix.is_contiguous(), "prefix must be contiguous");
  TORCH_CHECK(feat_indices.is_contiguous(), "feat_indices must be contiguous");

  return emblayer_vec_cuda(vec_values, prefix, feat_indices, output_dim, pad_value);
}

torch::Tensor emblayer_vec(
    torch::Tensor vec_values,
    torch::Tensor prefix,
    torch::Tensor feat_indices,
    int64_t output_dim,
    double pad_value) {
  TORCH_CHECK(vec_values.is_cuda(), "vec_values must be a CUDA tensor");
  TORCH_CHECK(vec_values.dim() == 1, "vec_values must be 1D");
  TORCH_CHECK(vec_values.is_contiguous(), "vec_values must be contiguous");

  TORCH_CHECK(prefix.dim() == 1, "prefix must be 1D");
  TORCH_CHECK(
      prefix.scalar_type() == torch::kInt32 || prefix.scalar_type() == torch::kInt64,
      "prefix must be int32 or int64");
  TORCH_CHECK(prefix.numel() >= 2, "prefix must have size >= 2");

  TORCH_CHECK(feat_indices.dim() == 1, "feat_indices must be 1D");
  TORCH_CHECK(
      feat_indices.scalar_type() == torch::kInt32 || feat_indices.scalar_type() == torch::kInt64,
      "feat_indices must be int32 or int64");
  TORCH_CHECK(
      feat_indices.numel() == vec_values.numel(),
      "feat_indices size must equal vec_values size");

  TORCH_CHECK(output_dim > 0, "output_dim must be > 0");

  auto prefix_cpu = prefix.to(torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU),
                              /*non_blocking=*/false,
                              /*copy=*/true);
  auto feat_indices_cpu =
      feat_indices.to(torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU),
                      /*non_blocking=*/false,
                      /*copy=*/true);

  auto prefix_ptr = prefix_cpu.data_ptr<int64_t>();
  auto feat_ptr = feat_indices_cpu.data_ptr<int64_t>();

  const int64_t batch_size = prefix_cpu.numel() - 1;
  TORCH_CHECK(prefix_ptr[0] == 0, "prefix[0] must be 0");
  TORCH_CHECK(prefix_ptr[batch_size] == vec_values.numel(),
              "prefix[-1] must equal vec_values.size(0)");

  for (int64_t b = 0; b < batch_size; ++b) {
    TORCH_CHECK(prefix_ptr[b] <= prefix_ptr[b + 1], "prefix must be non-decreasing");
  }

  for (int64_t i = 0; i < feat_indices_cpu.numel(); ++i) {
    TORCH_CHECK(feat_ptr[i] >= 0 && feat_ptr[i] < output_dim,
                "feat_indices out of output_dim range");
  }

  return emblayer_vec_cuda(vec_values, prefix_cpu, feat_indices_cpu, output_dim, pad_value);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "emblayer_vec",
      &emblayer_vec,
      py::arg("vec_values"),
      py::arg("prefix"),
      py::arg("feat_indices"),
      py::arg("output_dim"),
      py::arg("pad_value") = 0.0,
      "emblayerVec CUDA op: rebuild dense [B, D] tensor from compressed non-seq feature buffer");

  m.def(
      "emblayer_vec_fast",
      &emblayer_vec_fast,
      py::arg("vec_values"),
      py::arg("prefix"),
      py::arg("feat_indices"),
      py::arg("output_dim"),
      py::arg("pad_value") = 0.0,
      "emblayerVec fast path: metadata already on CUDA int64 tensors");
}
