# -*- coding: utf-8 -*-
import os

import torch
from torch.utils.cpp_extension import load


def load_emblayerseq_extension():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ext_dir = os.path.join(current_dir, 'emblayer_op')
    build_dir = os.path.join(current_dir, '.torch_extensions')
    os.makedirs(build_dir, exist_ok=True)

    return load(
        name='emblayer_seq_cuda_ext',
        sources=[
            os.path.join(ext_dir, 'emblayer_seq.cpp'),
            os.path.join(ext_dir, 'emblayer_seq_cuda.cu'),
        ],
        extra_cflags=['-O3'],
        extra_cuda_cflags=['-O3'],
        build_directory=build_dir,
        verbose=True,
    )


def cpu_reference(seq_values, prefix, seq_lengths, pad_value=0):
    batch_size = prefix.numel() - 1
    max_seq_num = 0
    for b in range(batch_size):
        max_seq_num = max(max_seq_num, int(prefix[b + 1] - prefix[b]))
    max_seq_len = int(seq_lengths.max().item()) if seq_lengths.numel() > 0 else 0

    out = torch.full((batch_size, max_seq_num, max_seq_len), pad_value, dtype=seq_values.dtype)

    seq_offsets = torch.zeros(seq_lengths.numel() + 1, dtype=torch.int64)
    if seq_lengths.numel() > 0:
        seq_offsets[1:] = torch.cumsum(seq_lengths, dim=0)

    for b in range(batch_size):
        begin = int(prefix[b].item())
        end = int(prefix[b + 1].item())
        for s in range(begin, end):
            slot = s - begin
            l = int(seq_lengths[s].item())
            off = int(seq_offsets[s].item())
            out[b, slot, :l] = seq_values[off:off + l]
    return out


if __name__ == '__main__':
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for emblayerSeq demo, but no CUDA device found.')

    ext = load_emblayerseq_extension()

    # Compressed format example:
    # batch 0 has 2 seqs, batch 1 has 1 seq, batch 2 has 3 seqs
    # prefix: [0, 2, 3, 6]
    prefix = torch.tensor([0, 2, 3, 6], dtype=torch.int64)
    seq_lengths = torch.tensor([3, 2, 4, 1, 3, 2], dtype=torch.int64)

    # Flatten all tokens in iobuffer order
    seq_values = torch.tensor([
        11, 12, 13,        # seq0
        21, 22,            # seq1
        31, 32, 33, 34,    # seq2
        41,                # seq3
        51, 52, 53,        # seq4
        61, 62,            # seq5
    ], dtype=torch.int64)

    pad_value = 0
    out_cpu = cpu_reference(seq_values, prefix, seq_lengths, pad_value=pad_value)

    out_gpu = ext.emblayer_seq(
        seq_values.cuda(),
        prefix,
        seq_lengths,
        pad_value,
        -1,
        -1,
    ).cpu()

    same = torch.equal(out_cpu, out_gpu)
    print('emblayerSeq demo done')
    print('output shape:', tuple(out_gpu.shape))
    print('cpu/gpu match:', same)
    print('output tensor:')
    print(out_gpu)

    if not same:
        raise RuntimeError('emblayerSeq output mismatch against CPU reference')
