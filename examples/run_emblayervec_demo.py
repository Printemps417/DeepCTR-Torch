# -*- coding: utf-8 -*-
import os

import torch
from torch.utils.cpp_extension import load


def load_emblayervec_extension():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ext_dir = os.path.join(current_dir, 'emblayer_op')
    build_dir = os.path.join(current_dir, '.torch_extensions')
    os.makedirs(build_dir, exist_ok=True)

    return load(
        name='emblayer_vec_cuda_ext',
        sources=[
            os.path.join(ext_dir, 'emblayer_vec.cpp'),
            os.path.join(ext_dir, 'emblayer_vec_cuda.cu'),
        ],
        extra_cflags=['-O3'],
        extra_cuda_cflags=['-O3'],
        build_directory=build_dir,
        verbose=True,
    )


def cpu_reference(vec_values, prefix, feat_indices, output_dim, pad_value=0.0):
    batch_size = prefix.numel() - 1
    out = torch.full((batch_size, output_dim), pad_value, dtype=vec_values.dtype)
    for b in range(batch_size):
        begin = int(prefix[b].item())
        end = int(prefix[b + 1].item())
        for i in range(begin, end):
            col = int(feat_indices[i].item())
            out[b, col] = vec_values[i]
    return out


if __name__ == '__main__':
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for emblayerVec demo, but no CUDA device found.')

    ext = load_emblayervec_extension()

    # Compressed vector format example:
    # batch0 has 3 features, batch1 has 2 features, batch2 has 4 features
    prefix = torch.tensor([0, 3, 5, 9], dtype=torch.int32)
    feat_indices = torch.tensor([0, 3, 5, 1, 4, 0, 2, 5, 6], dtype=torch.int32)
    vec_values = torch.tensor([1.1, 1.3, 1.5, 2.1, 2.4, 3.0, 3.2, 3.5, 3.6], dtype=torch.float32)

    output_dim = 8
    pad_value = 0.0

    out_cpu = cpu_reference(vec_values, prefix.to(torch.int64), feat_indices.to(torch.int64), output_dim, pad_value)
    out_gpu = ext.emblayer_vec(
        vec_values.cuda(),
        prefix,
        feat_indices,
        output_dim,
        pad_value,
    ).cpu()

    same = torch.allclose(out_cpu, out_gpu, atol=1e-6)
    print('emblayerVec demo done')
    print('output shape:', tuple(out_gpu.shape))
    print('cpu/gpu match:', same)
    print('output tensor:')
    print(out_gpu)

    if not same:
        raise RuntimeError('emblayerVec output mismatch against CPU reference')
