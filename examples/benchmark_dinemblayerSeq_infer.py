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
        verbose=False,
    )


def build_din_model(device, max_seq_len=64):
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
    behavior_feature_list = ['item', 'item_gender']

    model = DIN(feature_columns, behavior_feature_list, device=device, att_weight_normalization=True)
    model.eval()
    return model, feature_columns


def _sample_seq_lengths(batch_size, max_seq_len, avg_seq_len, rng):
    raw = rng.poisson(lam=max(1.0, float(avg_seq_len)), size=(batch_size,))
    raw = np.clip(raw, 1, max_seq_len).astype(np.int64)
    return raw


def build_batch_inputs(model, feature_columns, batch_size, max_seq_len, avg_seq_len, seed=2026):
    rng = np.random.default_rng(seed)
    input_dim = max(end for _, end in model.feature_index.values())

    lengths = _sample_seq_lengths(batch_size, max_seq_len, avg_seq_len, rng)

    full = np.zeros((batch_size, input_dim), dtype=np.float32)

    sparse_feats = [f for f in feature_columns if isinstance(f, SparseFeat)]
    dense_feats = [f for f in feature_columns if isinstance(f, DenseFeat)]
    varlen_feats = [f for f in feature_columns if isinstance(f, VarLenSparseFeat)]

    seq_feature_info = []

    non_seq_spans = []

    for feat in sparse_feats:
        s, e = model.feature_index[feat.name]
        vals = rng.integers(1, feat.vocabulary_size, size=(batch_size,), endpoint=False).astype(np.float32)
        full[:, s:e] = vals.reshape(batch_size, 1)
        non_seq_spans.append((s, e))

    for feat in dense_feats:
        s, e = model.feature_index[feat.name]
        vals = rng.random((batch_size, feat.dimension), dtype=np.float32)
        full[:, s:e] = vals
        non_seq_spans.append((s, e))

    seq_length_name = None
    for feat in varlen_feats:
        if feat.length_name is not None:
            seq_length_name = feat.length_name

    if seq_length_name is not None:
        s, e = model.feature_index[seq_length_name]
        full[:, s:e] = lengths.reshape(batch_size, 1).astype(np.float32)
        non_seq_spans.append((s, e))

    seq_tokens_by_feature = {}

    for feat in varlen_feats:
        s, e = model.feature_index[feat.name]
        padded = np.zeros((batch_size, feat.maxlen), dtype=np.float32)
        feat_tokens = []
        for i in range(batch_size):
            l = int(lengths[i])
            seq_vals = rng.integers(1, feat.vocabulary_size, size=(l,), endpoint=False).astype(np.int32)
            padded[i, :l] = seq_vals.astype(np.float32)
            feat_tokens.append(seq_vals)
        full[:, s:e] = padded
        seq_tokens_by_feature[feat.name] = feat_tokens
        seq_feature_info.append({
            'name': feat.name,
            'col_start': s,
            'col_end': e,
            'maxlen': feat.maxlen,
            'vocabulary_size': feat.vocabulary_size,
        })

    # Build a single compressed iobuffer for all seq features.
    # For each batch sample, append seqs in seq_feature_info order.
    num_seq_per_sample = len(seq_feature_info)
    combined_prefix = np.arange(0, (batch_size + 1) * num_seq_per_sample, num_seq_per_sample, dtype=np.int32)
    combined_lengths = np.repeat(lengths.astype(np.int32), repeats=num_seq_per_sample)

    combined_values = []
    for i in range(batch_size):
        for info in seq_feature_info:
            combined_values.append(seq_tokens_by_feature[info['name']][i])
    combined_values = np.concatenate(combined_values, axis=0).astype(np.int32)

    seq_host = {
        'values': combined_values,
        'prefix': combined_prefix,
        'lengths': combined_lengths,
        'num_seq_per_sample': num_seq_per_sample,
        'max_seq_len': max_seq_len,
        'features': seq_feature_info,
    }

    stats = {
        'avg_len': float(np.mean(lengths)),
        'max_len': int(np.max(lengths)),
        'min_len': int(np.min(lengths)),
    }

    non_seq_spans = sorted(non_seq_spans, key=lambda x: x[0])
    compact_parts = [full[:, s:e] for s, e in non_seq_spans]
    non_seq_compact = np.concatenate(compact_parts, axis=-1).astype(np.float32)

    compact_layout = []
    cursor = 0
    for s, e in non_seq_spans:
        w = e - s
        compact_layout.append((s, e, cursor, cursor + w))
        cursor += w

    full_t = torch.from_numpy(full)
    non_seq_t = torch.from_numpy(non_seq_compact)

    baseline_bytes = full.nbytes
    emblayer_bytes = non_seq_compact.nbytes
    emblayer_bytes += seq_host['values'].nbytes + seq_host['prefix'].nbytes + seq_host['lengths'].nbytes

    return full_t, non_seq_t, seq_host, compact_layout, input_dim, stats, baseline_bytes, emblayer_bytes


class DINEmblayerSeqInfer(nn.Module):
    def __init__(self, model, seq_host_info, compact_layout, input_dim, emblayer_ext):
        super().__init__()
        self.model = model
        self.seq_host_info = seq_host_info
        self.compact_layout = compact_layout
        self.input_dim = input_dim
        self.emblayer_ext = emblayer_ext

    def forward(self, non_seq_compact, seq_device_dict):
        batch_size = non_seq_compact.size(0)
        x = torch.zeros((batch_size, self.input_dim), device=non_seq_compact.device, dtype=non_seq_compact.dtype)

        for dst_s, dst_e, src_s, src_e in self.compact_layout:
            x[:, dst_s:dst_e] = non_seq_compact[:, src_s:src_e]

        seq_padded = self.emblayer_ext.emblayer_seq(
            seq_device_dict['values'],
            seq_device_dict['prefix'],
            seq_device_dict['lengths'],
            0,
            self.seq_host_info['num_seq_per_sample'],
            self.seq_host_info['max_seq_len'],
        )

        for seq_idx, info in enumerate(self.seq_host_info['features']):
            seq_2d = seq_padded[:, seq_idx, :].to(x.dtype)
            x[:, info['col_start']:info['col_end']] = seq_2d
        return self.model(x)


def _to_device(t, device):
    if device.startswith('cuda'):
        return t.pin_memory().to(device, non_blocking=True)
    return t.to(device)


def benchmark_h2d_only_baseline(full_host, device, iters, warmup):
    if not device.startswith('cuda'):
        return 0.0
    for _ in range(warmup):
        _ = _to_device(full_host, device)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = _to_device(full_host, device)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / iters


def benchmark_h2d_only_emblayer(non_seq_host, seq_host_tensors, device, iters, warmup):
    if not device.startswith('cuda'):
        return 0.0
    for _ in range(warmup):
        _ = _to_device(non_seq_host, device)
        _ = _to_device(seq_host_tensors['values'], device)
        _ = _to_device(seq_host_tensors['prefix'], device)
        _ = _to_device(seq_host_tensors['lengths'], device)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = _to_device(non_seq_host, device)
        _ = _to_device(seq_host_tensors['values'], device)
        _ = _to_device(seq_host_tensors['prefix'], device)
        _ = _to_device(seq_host_tensors['lengths'], device)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / iters


def benchmark_baseline(model, full_host, device, iters, warmup, include_h2d):
    if not include_h2d:
        full_dev = _to_device(full_host, device)
    if device.startswith('cuda'):
        torch.cuda.synchronize()

    with torch.no_grad():
        for _ in range(warmup):
            x = _to_device(full_host, device) if include_h2d else full_dev
            _ = model(x)

    if device.startswith('cuda'):
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            x = _to_device(full_host, device) if include_h2d else full_dev
            y = model(x)
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / iters, y


def benchmark_emblayer(model, emblayer_model, non_seq_host, seq_host_tensors, device, iters, warmup, include_h2d):
    if not include_h2d:
        non_seq_dev = _to_device(non_seq_host, device)
        seq_dev = {
            'values': _to_device(seq_host_tensors['values'], device),
            'prefix': _to_device(seq_host_tensors['prefix'], device),
            'lengths': _to_device(seq_host_tensors['lengths'], device),
        }

    if device.startswith('cuda'):
        torch.cuda.synchronize()

    with torch.no_grad():
        for _ in range(warmup):
            if include_h2d:
                non_seq_dev = _to_device(non_seq_host, device)
                seq_dev = {
                    'values': _to_device(seq_host_tensors['values'], device),
                    'prefix': _to_device(seq_host_tensors['prefix'], device),
                    'lengths': _to_device(seq_host_tensors['lengths'], device),
                }
            _ = emblayer_model(non_seq_dev, seq_dev)

    if device.startswith('cuda'):
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            if include_h2d:
                non_seq_dev = _to_device(non_seq_host, device)
                seq_dev = {
                    'values': _to_device(seq_host_tensors['values'], device),
                    'prefix': _to_device(seq_host_tensors['prefix'], device),
                    'lengths': _to_device(seq_host_tensors['lengths'], device),
                }
            y = emblayer_model(non_seq_dev, seq_dev)
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / iters, y


def main():
    parser = argparse.ArgumentParser(description='Benchmark DIN baseline vs EmblayerSeq compressed-seq path')
    parser.add_argument('--mode', type=str, choices=['baseline', 'emblayer', 'both'], default='both')
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

    model, feature_columns = build_din_model(device=device, max_seq_len=args.max_seq_len)
    full_host, non_seq_host, seq_host, compact_layout, input_dim, stats, baseline_bytes, emblayer_bytes = build_batch_inputs(
        model=model,
        feature_columns=feature_columns,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        avg_seq_len=args.avg_seq_len,
    )

    print('device:', device)
    print('batch_size:', args.batch_size)
    print('max_seq_len:', args.max_seq_len)
    print('avg_seq_len(sampled):', round(stats['avg_len'], 4), 'min/max:', stats['min_len'], '/', stats['max_len'])
    print('include_h2d:', args.include_h2d)
    print('baseline_input_bytes_per_batch:', baseline_bytes)
    print('emblayer_input_bytes_per_batch:', emblayer_bytes)
    print('input_bytes_reduction_ratio:', round((baseline_bytes - emblayer_bytes) / baseline_bytes, 6))

    baseline_ms = None
    emblayer_ms = None
    h2d_baseline_ms = None
    h2d_emblayer_ms = None
    y_baseline = None
    y_emblayer = None

    seq_host_tensors = {
        'values': torch.from_numpy(seq_host['values']),
        'prefix': torch.from_numpy(seq_host['prefix']),
        'lengths': torch.from_numpy(seq_host['lengths']),
    }

    if args.mode in ('baseline', 'both'):
        baseline_ms, y_baseline = benchmark_baseline(
            model=model,
            full_host=full_host,
            device=device,
            iters=args.iters,
            warmup=args.warmup,
            include_h2d=args.include_h2d,
        )
        print('baseline avg latency (ms):', round(baseline_ms, 6))
        if args.include_h2d:
            h2d_baseline_ms = benchmark_h2d_only_baseline(
                full_host=full_host,
                device=device,
                iters=args.iters,
                warmup=args.warmup,
            )
            print('baseline h2d-only (ms):', round(h2d_baseline_ms, 6))

    if args.mode in ('emblayer', 'both'):
        if device == 'cpu':
            raise RuntimeError('emblayer mode requires CUDA device')
        emblayer_ext = load_emblayerseq_extension()
        emblayer_model = DINEmblayerSeqInfer(
            model=model,
            seq_host_info=seq_host,
            compact_layout=compact_layout,
            input_dim=input_dim,
            emblayer_ext=emblayer_ext,
        ).to(device).eval()
        emblayer_ms, y_emblayer = benchmark_emblayer(
            model=model,
            emblayer_model=emblayer_model,
            non_seq_host=non_seq_host,
            seq_host_tensors=seq_host_tensors,
            device=device,
            iters=args.iters,
            warmup=args.warmup,
            include_h2d=args.include_h2d,
        )
        print('emblayer avg latency (ms):', round(emblayer_ms, 6))
        if args.include_h2d:
            h2d_emblayer_ms = benchmark_h2d_only_emblayer(
                non_seq_host=non_seq_host,
                seq_host_tensors=seq_host_tensors,
                device=device,
                iters=args.iters,
                warmup=args.warmup,
            )
            print('emblayer h2d-only (ms):', round(h2d_emblayer_ms, 6))

    if y_baseline is not None and y_emblayer is not None:
        diff = torch.max(torch.abs(y_baseline - y_emblayer)).item()
        print('max |baseline - emblayer|:', diff)
        if diff > 1e-6:
            raise RuntimeError('baseline and emblayer outputs mismatch')

    if baseline_ms is not None and emblayer_ms is not None:
        speedup = baseline_ms / emblayer_ms
        print('speedup (baseline/emblayer):', round(speedup, 6))
    if h2d_baseline_ms is not None and h2d_emblayer_ms is not None:
        h2d_speedup = h2d_baseline_ms / h2d_emblayer_ms if h2d_emblayer_ms > 0 else 0.0
        print('h2d speedup (baseline/emblayer):', round(h2d_speedup, 6))


if __name__ == '__main__':
    main()
