# -*- coding: utf-8 -*-
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

from deepctr_torch.inputs import DenseFeat, SparseFeat, VarLenSparseFeat
from deepctr_torch.models.din import DIN


def load_multislice_extension():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ext_dir = os.path.join(current_dir, 'multislice_op')
    build_dir = os.path.join(current_dir, '.torch_extensions')
    os.makedirs(build_dir, exist_ok=True)
    return load(
        name='multislice_cuda_ext',
        sources=[
            os.path.join(ext_dir, 'multislice.cpp'),
            os.path.join(ext_dir, 'multislice_cuda.cu'),
        ],
        extra_cflags=['-O3'],
        extra_cuda_cflags=['-O3'],
        build_directory=build_dir,
        verbose=False,
    )


def load_emblayer_seq_extension():
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
        verbose=False,
    )


def build_din_model(device, max_seq_len=128):
    feature_columns = [
        SparseFeat('user', 200000, embedding_dim=16),
        SparseFeat('gender', 4, embedding_dim=8),
        SparseFeat('item', 500000, embedding_dim=16),
        SparseFeat('item_gender', 4, embedding_dim=8),
        DenseFeat('score', 1),
    ]
    feature_columns += [
        VarLenSparseFeat(SparseFeat('hist_item', 500000, embedding_dim=16), max_seq_len, length_name='seq_length'),
        VarLenSparseFeat(SparseFeat('hist_item_gender', 4, embedding_dim=8), max_seq_len, length_name='seq_length'),
    ]
    model = DIN(feature_columns, ['item', 'item_gender'], device=device, att_weight_normalization=True)
    model.eval()
    return model, feature_columns


def _to_device(tensor, device):
    if device.startswith('cuda'):
        return tensor.pin_memory().to(device, non_blocking=True)
    return tensor.to(device)


def build_inputs(model, feature_columns, batch_size, max_seq_len, avg_seq_len, seed=2026):
    rng = np.random.default_rng(seed)
    input_dim = max(end for _, end in model.feature_index.values())
    full = np.zeros((batch_size, input_dim), dtype=np.float32)

    sparse_features = [f for f in feature_columns if isinstance(f, SparseFeat)]
    dense_features = [f for f in feature_columns if isinstance(f, DenseFeat)]
    varlen_features = [f for f in feature_columns if isinstance(f, VarLenSparseFeat)]

    lengths = rng.poisson(lam=max(1.0, float(avg_seq_len)), size=(batch_size,))
    lengths = np.clip(lengths, 1, max_seq_len).astype(np.int32)

    seq_length_name = None
    for feat in varlen_features:
        if feat.length_name is not None:
            seq_length_name = feat.length_name

    non_seq_spans = []
    for feat in sparse_features:
        s, e = model.feature_index[feat.name]
        vals = rng.integers(1, feat.vocabulary_size, size=(batch_size,), endpoint=False).astype(np.float32)
        full[:, s:e] = vals.reshape(batch_size, 1)
        non_seq_spans.append((s, e))

    for feat in dense_features:
        s, e = model.feature_index[feat.name]
        vals = rng.random((batch_size, feat.dimension), dtype=np.float32)
        full[:, s:e] = vals
        non_seq_spans.append((s, e))

    if seq_length_name is not None:
        s, e = model.feature_index[seq_length_name]
        full[:, s:e] = lengths.reshape(batch_size, 1).astype(np.float32)
        non_seq_spans.append((s, e))

    seq_feature_info = []
    seq_tokens_by_feature = {}
    for feat in varlen_features:
        s, e = model.feature_index[feat.name]
        padded = np.zeros((batch_size, feat.maxlen), dtype=np.float32)
        per_row_tokens = []
        for row in range(batch_size):
            cur_len = int(lengths[row])
            seq_vals = rng.integers(1, feat.vocabulary_size, size=(cur_len,), endpoint=False).astype(np.int32)
            padded[row, :cur_len] = seq_vals.astype(np.float32)
            per_row_tokens.append(seq_vals)
        full[:, s:e] = padded
        seq_tokens_by_feature[feat.name] = per_row_tokens
        seq_feature_info.append({'name': feat.name, 'col_start': s, 'col_end': e, 'maxlen': feat.maxlen})

    non_seq_full = np.zeros_like(full)
    non_seq_spans = sorted(non_seq_spans, key=lambda x: x[0])
    for s, e in non_seq_spans:
        non_seq_full[:, s:e] = full[:, s:e]

    compact_layout = []
    starts = []
    lengths_spans = []
    cursor = 0
    for s, e in non_seq_spans:
        w = e - s
        compact_layout.append((s, e, cursor, cursor + w))
        starts.append(s)
        lengths_spans.append(w)
        cursor += w

    num_seq_per_sample = len(seq_feature_info)
    seq_prefix = np.arange(0, (batch_size + 1) * num_seq_per_sample, num_seq_per_sample, dtype=np.int32)
    seq_lengths = np.repeat(lengths, repeats=num_seq_per_sample).astype(np.int32)
    seq_values = []
    for row in range(batch_size):
        for info in seq_feature_info:
            seq_values.append(seq_tokens_by_feature[info['name']][row])
    seq_values = np.concatenate(seq_values, axis=0).astype(np.int32)

    baseline_bytes = full.nbytes
    hybrid_bytes = non_seq_full.nbytes + seq_values.nbytes + seq_prefix.nbytes + seq_lengths.nbytes

    return {
        'full': torch.from_numpy(full),
        'non_seq_full': torch.from_numpy(non_seq_full),
        'seq_values': torch.from_numpy(seq_values),
        'seq_prefix': torch.from_numpy(seq_prefix),
        'seq_lengths': torch.from_numpy(seq_lengths),
        'seq_feature_info': seq_feature_info,
        'compact_layout': compact_layout,
        'starts': torch.tensor(starts, dtype=torch.int64),
        'lengths_spans': torch.tensor(lengths_spans, dtype=torch.int64),
        'num_seq_per_sample': num_seq_per_sample,
        'max_seq_len': max_seq_len,
        'input_dim': input_dim,
        'bytes': {'baseline': baseline_bytes, 'hybrid': hybrid_bytes},
        'stats': {'avg_len': float(np.mean(lengths)), 'min_len': int(np.min(lengths)), 'max_len': int(np.max(lengths))},
    }


class DINMultiSliceEmblayerSeqInfer(nn.Module):
    def __init__(self, model, seq_feature_info, compact_layout, starts, lengths_spans, input_dim, num_seq_per_sample, max_seq_len, multislice_ext, emblayer_seq_ext):
        super().__init__()
        self.model = model
        self.seq_feature_info = seq_feature_info
        self.compact_layout = compact_layout
        self.starts = starts
        self.lengths_spans = lengths_spans
        self.input_dim = input_dim
        self.num_seq_per_sample = num_seq_per_sample
        self.max_seq_len = max_seq_len
        self.multislice_ext = multislice_ext
        self.emblayer_seq_ext = emblayer_seq_ext

    def forward(self, non_seq_full, seq_values, seq_prefix, seq_lengths):
        compact = self.multislice_ext.multislice(non_seq_full, self.starts, self.lengths_spans)

        x = torch.zeros((non_seq_full.size(0), self.input_dim), device=non_seq_full.device, dtype=non_seq_full.dtype)
        for dst_s, dst_e, src_s, src_e in self.compact_layout:
            x[:, dst_s:dst_e] = compact[:, src_s:src_e]

        seq_padded = self.emblayer_seq_ext.emblayer_seq(
            seq_values,
            seq_prefix,
            seq_lengths,
            0,
            self.num_seq_per_sample,
            self.max_seq_len,
        )
        for seq_idx, info in enumerate(self.seq_feature_info):
            x[:, info['col_start']:info['col_end']] = seq_padded[:, seq_idx, :].to(x.dtype)
        return self.model(x)


def benchmark_baseline(model, host_data, device, iters, warmup, include_h2d):
    if not include_h2d:
        full_dev = _to_device(host_data['full'], device)

    if device.startswith('cuda'):
        torch.cuda.synchronize()

    with torch.no_grad():
        for _ in range(warmup):
            if include_h2d:
                full_dev = _to_device(host_data['full'], device)
            _ = model(full_dev)

    if device.startswith('cuda'):
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            if include_h2d:
                full_dev = _to_device(host_data['full'], device)
            y = model(full_dev)
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    return (t1 - t0) * 1000.0 / iters, y


def benchmark_hybrid(infer_model, host_data, device, iters, warmup, include_h2d):
    if not include_h2d:
        non_seq_full_dev = _to_device(host_data['non_seq_full'], device)
        seq_values_dev = _to_device(host_data['seq_values'], device)
        seq_prefix_dev = _to_device(host_data['seq_prefix'], device)
        seq_lengths_dev = _to_device(host_data['seq_lengths'], device)

    if device.startswith('cuda'):
        torch.cuda.synchronize()

    with torch.no_grad():
        for _ in range(warmup):
            if include_h2d:
                non_seq_full_dev = _to_device(host_data['non_seq_full'], device)
                seq_values_dev = _to_device(host_data['seq_values'], device)
                seq_prefix_dev = _to_device(host_data['seq_prefix'], device)
                seq_lengths_dev = _to_device(host_data['seq_lengths'], device)
            _ = infer_model(non_seq_full_dev, seq_values_dev, seq_prefix_dev, seq_lengths_dev)

    if device.startswith('cuda'):
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            if include_h2d:
                non_seq_full_dev = _to_device(host_data['non_seq_full'], device)
                seq_values_dev = _to_device(host_data['seq_values'], device)
                seq_prefix_dev = _to_device(host_data['seq_prefix'], device)
                seq_lengths_dev = _to_device(host_data['seq_lengths'], device)
            y_hybrid = infer_model(non_seq_full_dev, seq_values_dev, seq_prefix_dev, seq_lengths_dev)

    if device.startswith('cuda'):
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    return (t1 - t0) * 1000.0 / iters, y_hybrid


def h2d_only(host_data, device, iters, warmup):
    if not device.startswith('cuda'):
        return 0.0, 0.0

    def transfer_baseline():
        _ = _to_device(host_data['full'], device)

    def transfer_hybrid():
        _ = _to_device(host_data['non_seq_full'], device)
        _ = _to_device(host_data['seq_values'], device)
        _ = _to_device(host_data['seq_prefix'], device)
        _ = _to_device(host_data['seq_lengths'], device)

    for _ in range(warmup):
        transfer_baseline()
        transfer_hybrid()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        transfer_baseline()
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    for _ in range(iters):
        transfer_hybrid()
    torch.cuda.synchronize()
    t3 = time.perf_counter()

    base_ms = (t1 - t0) * 1000.0 / iters
    hyb_ms = (t3 - t2) * 1000.0 / iters
    return base_ms, hyb_ms


def main():
    parser = argparse.ArgumentParser(description='Benchmark hybrid path: MultiSlice + EmblayerSeq')
    parser.add_argument('--iters', type=int, default=1000)
    parser.add_argument('--warmup', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--max_seq_len', type=int, default=128)
    parser.add_argument('--avg_seq_len', type=float, default=8.0)
    parser.add_argument('--include_h2d', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    device = 'cpu'
    if not args.cpu and torch.cuda.is_available():
        device = 'cuda:0'
    if device == 'cpu':
        raise RuntimeError('this benchmark is intended for CUDA path')

    model, feature_columns = build_din_model(device=device, max_seq_len=args.max_seq_len)
    host_data = build_inputs(model, feature_columns, args.batch_size, args.max_seq_len, args.avg_seq_len)

    multislice_ext = load_multislice_extension()
    emblayer_seq_ext = load_emblayer_seq_extension()

    infer_model = DINMultiSliceEmblayerSeqInfer(
        model=model,
        seq_feature_info=host_data['seq_feature_info'],
        compact_layout=host_data['compact_layout'],
        starts=host_data['starts'],
        lengths_spans=host_data['lengths_spans'],
        input_dim=host_data['input_dim'],
        num_seq_per_sample=host_data['num_seq_per_sample'],
        max_seq_len=host_data['max_seq_len'],
        multislice_ext=multislice_ext,
        emblayer_seq_ext=emblayer_seq_ext,
    ).to(device).eval()

    baseline_ms, y_base = benchmark_baseline(model, host_data, device, args.iters, args.warmup, args.include_h2d)
    hybrid_ms, y_hybrid = benchmark_hybrid(infer_model, host_data, device, args.iters, args.warmup, args.include_h2d)
    diff = torch.max(torch.abs(y_base - y_hybrid)).item()

    print('mode: multislice+emblayerSeq')
    print('device:', device)
    print('batch_size:', args.batch_size)
    print('max_seq_len:', args.max_seq_len)
    print('avg_seq_len(sampled):', round(host_data['stats']['avg_len'], 4), 'min/max:', host_data['stats']['min_len'], '/', host_data['stats']['max_len'])
    print('include_h2d:', args.include_h2d)
    print('baseline_input_bytes_per_batch:', host_data['bytes']['baseline'])
    print('hybrid_input_bytes_per_batch:', host_data['bytes']['hybrid'])
    print('input_bytes_reduction_ratio:', round((host_data['bytes']['baseline'] - host_data['bytes']['hybrid']) / host_data['bytes']['baseline'], 6))
    print('baseline avg latency (ms):', round(baseline_ms, 6))
    print('hybrid avg latency (ms):', round(hybrid_ms, 6))
    print('speedup (baseline/hybrid):', round(baseline_ms / hybrid_ms if hybrid_ms > 0 else 0.0, 6))
    print('max |baseline - hybrid|:', diff)

    if args.include_h2d:
        h2d_base, h2d_hybrid = h2d_only(host_data, device, args.iters, args.warmup)
        print('baseline h2d-only (ms):', round(h2d_base, 6))
        print('hybrid h2d-only (ms):', round(h2d_hybrid, 6))
        print('h2d speedup (baseline/hybrid):', round(h2d_base / h2d_hybrid if h2d_hybrid > 0 else 0.0, 6))

    if diff > 1e-6:
        raise RuntimeError('hybrid output mismatch against baseline')


if __name__ == '__main__':
    main()
