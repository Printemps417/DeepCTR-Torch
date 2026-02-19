#include <torch/extension.h>

#include <vector>

torch::Tensor emblayer_seq_cuda(
    torch::Tensor seq_values,
    torch::Tensor prefix,
    torch::Tensor seq_lengths,
    torch::Tensor seq_offsets,
    int64_t max_seq_num,
    int64_t max_seq_len,
    int64_t pad_value);

torch::Tensor emblayer_seq(
    torch::Tensor seq_values,
    torch::Tensor prefix,
    torch::Tensor seq_lengths,
    int64_t pad_value,
    int64_t max_seq_num,
    int64_t max_seq_len) {
  TORCH_CHECK(seq_values.is_cuda(), "seq_values must be a CUDA tensor");
  TORCH_CHECK(seq_values.dim() == 1, "seq_values must be 1D");
  TORCH_CHECK(seq_values.is_contiguous(), "seq_values must be contiguous");

  TORCH_CHECK(prefix.dim() == 1, "prefix must be 1D");
  TORCH_CHECK(
      prefix.scalar_type() == torch::kInt32 || prefix.scalar_type() == torch::kInt64,
      "prefix must be int32 or int64");
  TORCH_CHECK(prefix.numel() >= 2, "prefix must have size >= 2");

  TORCH_CHECK(seq_lengths.dim() == 1, "seq_lengths must be 1D");
  TORCH_CHECK(
      seq_lengths.scalar_type() == torch::kInt32 || seq_lengths.scalar_type() == torch::kInt64,
      "seq_lengths must be int32 or int64");

  auto prefix_cpu = prefix.to(torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU),
                              /*non_blocking=*/false,
                              /*copy=*/true);
  auto lengths_cpu = seq_lengths.to(torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU),
                                    /*non_blocking=*/false,
                                    /*copy=*/true);

  auto prefix_ptr = prefix_cpu.data_ptr<int64_t>();
  auto lengths_ptr = lengths_cpu.data_ptr<int64_t>();

  const int64_t batch_size = prefix_cpu.numel() - 1;
  const int64_t num_seq = seq_lengths.numel();

  TORCH_CHECK(prefix_ptr[0] == 0, "prefix[0] must be 0");
  TORCH_CHECK(prefix_ptr[batch_size] == num_seq,
              "prefix[-1] must equal seq_lengths.size(0)");

  std::vector<int64_t> seq_offsets(num_seq + 1, 0);
  int64_t inferred_max_seq_len = 0;
  for (int64_t i = 0; i < num_seq; ++i) {
    TORCH_CHECK(lengths_ptr[i] >= 0, "seq_lengths must be non-negative");
    seq_offsets[i + 1] = seq_offsets[i] + lengths_ptr[i];
    if (lengths_ptr[i] > inferred_max_seq_len) {
      inferred_max_seq_len = lengths_ptr[i];
    }
  }

  TORCH_CHECK(seq_offsets[num_seq] == seq_values.numel(),
              "sum(seq_lengths) must equal seq_values.numel()");

  int64_t inferred_max_seq_num = 0;
  for (int64_t b = 0; b < batch_size; ++b) {
    TORCH_CHECK(prefix_ptr[b] <= prefix_ptr[b + 1], "prefix must be non-decreasing");
    const int64_t cnt = prefix_ptr[b + 1] - prefix_ptr[b];
    if (cnt > inferred_max_seq_num) {
      inferred_max_seq_num = cnt;
    }
  }

  if (max_seq_num <= 0) {
    max_seq_num = inferred_max_seq_num;
  }
  if (max_seq_len <= 0) {
    max_seq_len = inferred_max_seq_len;
  }

  TORCH_CHECK(max_seq_num >= inferred_max_seq_num,
              "max_seq_num is smaller than required by prefix");
  TORCH_CHECK(max_seq_len >= inferred_max_seq_len,
              "max_seq_len is smaller than required by seq_lengths");

  auto seq_offsets_tensor = torch::from_blob(
      seq_offsets.data(), {num_seq + 1}, torch::TensorOptions().dtype(torch::kInt64))
                                .clone();

  return emblayer_seq_cuda(
      seq_values,
      prefix_cpu,
      lengths_cpu,
      seq_offsets_tensor,
      max_seq_num,
      max_seq_len,
      pad_value);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "emblayer_seq",
      &emblayer_seq,
      py::arg("seq_values"),
      py::arg("prefix"),
      py::arg("seq_lengths"),
      py::arg("pad_value") = 0,
      py::arg("max_seq_num") = -1,
      py::arg("max_seq_len") = -1,
      "emblayerSeq CUDA op: rebuild padded [B, max_seq_num, max_seq_len] tensor from compressed sequence buffer");
}
