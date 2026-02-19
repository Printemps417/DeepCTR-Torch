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


def load_emblayer_vec_extension():
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
        verbose=False,
    )


def build_din_model(device, max_seq_len=256):
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


def _sync_if_cuda(device):
    if device.startswith('cuda'):
        torch.cuda.synchronize()


def build_inputs(model, feature_columns, batch_size, max_seq_len, avg_seq_len, seed=2026):
    rng = np.random.default_rng(seed)
    input_dim = max(end for _, end in model.feature_index.values())
    full = np.zeros((batch_size, input_dim), dtype=np.float32)

    sparse_features = [f for f in feature_columns if isinstance(f, SparseFeat)]
    dense_features = [f for f in feature_columns if isinstance(f, DenseFeat)]
    varlen_features = [f for f in feature_columns if isinstance(f, VarLenSparseFeat)]

    lengths = rng.poisson(lam=max(1.0, float(avg_seq_len)), size=(batch_size,))
    lengths = np.clip(lengths, 1, max_seq_len).astype(np.int32)

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

    seq_length_name = None
    for feat in varlen_features:
        if feat.length_name is not None:
            seq_length_name = feat.length_name
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

    non_seq_spans = sorted(non_seq_spans, key=lambda x: x[0])

    # non-seq full-width tensor (for stage3)
    non_seq_full = np.zeros_like(full)
    for s, e in non_seq_spans:
        non_seq_full[:, s:e] = full[:, s:e]

    # non-seq compact tensor (for stage5)
    compact_parts = [full[:, s:e] for s, e in non_seq_spans]
    non_seq_compact = np.concatenate(compact_parts, axis=-1).astype(np.float32)

    compact_layout = []
    starts = []
    span_lengths = []
    cursor = 0
    for s, e in non_seq_spans:
        width = e - s
        compact_layout.append((s, e, cursor, cursor + width))
        starts.append(s)
        span_lengths.append(width)
        cursor += width

    # seq compressed: single iobuffer for all seq features
    num_seq_per_sample = len(seq_feature_info)
    seq_prefix = np.arange(0, (batch_size + 1) * num_seq_per_sample, num_seq_per_sample, dtype=np.int32)
    seq_lengths = np.repeat(lengths, repeats=num_seq_per_sample).astype(np.int32)
    seq_values = []
    for row in range(batch_size):
        for info in seq_feature_info:
            seq_values.append(seq_tokens_by_feature[info['name']][row])
    seq_values = np.concatenate(seq_values, axis=0).astype(np.int32)

    # vec compressed (for stage4)
    vec_values = []
    vec_indices = []
    vec_prefix = [0]
    for row in range(batch_size):
        for s, e in non_seq_spans:
            for col in range(s, e):
                vec_values.append(full[row, col])
                vec_indices.append(col)
        vec_prefix.append(len(vec_values))

    vec_values = np.asarray(vec_values, dtype=np.float32)
    vec_indices = np.asarray(vec_indices, dtype=np.int32)
    vec_prefix = np.asarray(vec_prefix, dtype=np.int32)

    bytes_baseline = full.nbytes
    bytes_multislice = full.nbytes
    bytes_multislice_seq = non_seq_full.nbytes + seq_values.nbytes + seq_prefix.nbytes + seq_lengths.nbytes
    bytes_seq_only = non_seq_compact.nbytes + seq_values.nbytes + seq_prefix.nbytes + seq_lengths.nbytes
    bytes_joint = vec_values.nbytes + vec_indices.nbytes + vec_prefix.nbytes + seq_values.nbytes + seq_prefix.nbytes + seq_lengths.nbytes

    return {
        'full': torch.from_numpy(full),
        'non_seq_full': torch.from_numpy(non_seq_full),
        'non_seq_compact': torch.from_numpy(non_seq_compact),
        'seq_values': torch.from_numpy(seq_values),
        'seq_prefix': torch.from_numpy(seq_prefix),
        'seq_lengths': torch.from_numpy(seq_lengths),
        'vec_values': torch.from_numpy(vec_values),
        'vec_indices': torch.from_numpy(vec_indices),
        'vec_prefix': torch.from_numpy(vec_prefix),
        'compact_layout': compact_layout,
        'seq_feature_info': seq_feature_info,
        'starts': torch.tensor(starts, dtype=torch.int64),
        'span_lengths': torch.tensor(span_lengths, dtype=torch.int64),
        'input_dim': input_dim,
        'num_seq_per_sample': num_seq_per_sample,
        'max_seq_len': max_seq_len,
        'bytes': {
            'baseline': bytes_baseline,
            'multislice': bytes_multislice,
            'multislice_seq': bytes_multislice_seq,
            'seq_only': bytes_seq_only,
            'joint': bytes_joint,
        },
        'stats': {
            'avg_len': float(np.mean(lengths)),
            'min_len': int(np.min(lengths)),
            'max_len': int(np.max(lengths)),
        },
    }


class DINMultiSliceInfer(nn.Module):
    def __init__(self, model, compact_layout, starts, span_lengths, multislice_ext):
        super().__init__()
        self.model = model
        self.compact_layout = compact_layout
        self.starts = starts
        self.span_lengths = span_lengths
        self.multislice_ext = multislice_ext

    def forward(self, full_input):
        compact = self.multislice_ext.multislice(full_input, self.starts, self.span_lengths)
        x = full_input.clone()
        for dst_s, dst_e, src_s, src_e in self.compact_layout:
            x[:, dst_s:dst_e] = compact[:, src_s:src_e]
        return self.model(x)


class DINSeqOnlyInfer(nn.Module):
    def __init__(self, model, seq_feature_info, compact_layout, input_dim, num_seq_per_sample, max_seq_len, emblayer_seq_ext):
        super().__init__()
        self.model = model
        self.seq_feature_info = seq_feature_info
        self.compact_layout = compact_layout
        self.input_dim = input_dim
        self.num_seq_per_sample = num_seq_per_sample
        self.max_seq_len = max_seq_len
        self.emblayer_seq_ext = emblayer_seq_ext

    def forward(self, non_seq_compact, seq_values, seq_prefix, seq_lengths):
        batch_size = non_seq_compact.size(0)
        x = torch.zeros((batch_size, self.input_dim), device=non_seq_compact.device, dtype=non_seq_compact.dtype)
        for dst_s, dst_e, src_s, src_e in self.compact_layout:
            x[:, dst_s:dst_e] = non_seq_compact[:, src_s:src_e]

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


class DINMultiSliceSeqInfer(nn.Module):
    def __init__(self, model, seq_feature_info, compact_layout, starts, span_lengths, input_dim, num_seq_per_sample, max_seq_len, multislice_ext, emblayer_seq_ext):
        super().__init__()
        self.model = model
        self.seq_feature_info = seq_feature_info
        self.compact_layout = compact_layout
        self.starts = starts
        self.span_lengths = span_lengths
        self.input_dim = input_dim
        self.num_seq_per_sample = num_seq_per_sample
        self.max_seq_len = max_seq_len
        self.multislice_ext = multislice_ext
        self.emblayer_seq_ext = emblayer_seq_ext

    def forward(self, non_seq_full, seq_values, seq_prefix, seq_lengths):
        compact = self.multislice_ext.multislice(non_seq_full, self.starts, self.span_lengths)
        batch_size = non_seq_full.size(0)
        x = torch.zeros((batch_size, self.input_dim), device=non_seq_full.device, dtype=non_seq_full.dtype)
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


class DINJointInfer(nn.Module):
    def __init__(self, model, seq_feature_info, input_dim, num_seq_per_sample, max_seq_len, emblayer_seq_ext, emblayer_vec_ext):
        super().__init__()
        self.model = model
        self.seq_feature_info = seq_feature_info
        self.input_dim = input_dim
        self.num_seq_per_sample = num_seq_per_sample
        self.max_seq_len = max_seq_len
        self.emblayer_seq_ext = emblayer_seq_ext
        self.emblayer_vec_ext = emblayer_vec_ext

    def forward(self, vec_values, vec_prefix, vec_indices, seq_values, seq_prefix, seq_lengths):
        x = self.emblayer_vec_ext.emblayer_vec(vec_values, vec_prefix, vec_indices, self.input_dim, 0.0)
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


def _prepare_inputs(mode, host_data, device):
    if mode == 'baseline':
        return {'full': _to_device(host_data['full'], device)}
    if mode == 'multislice':
        return {'full': _to_device(host_data['full'], device)}
    if mode == 'multislice_seq':
        return {
            'non_seq_full': _to_device(host_data['non_seq_full'], device),
            'seq_values': _to_device(host_data['seq_values'], device),
            'seq_prefix': _to_device(host_data['seq_prefix'], device),
            'seq_lengths': _to_device(host_data['seq_lengths'], device),
        }
    if mode == 'seq_only':
        return {
            'non_seq_compact': _to_device(host_data['non_seq_compact'], device),
            'seq_values': _to_device(host_data['seq_values'], device),
            'seq_prefix': _to_device(host_data['seq_prefix'], device),
            'seq_lengths': _to_device(host_data['seq_lengths'], device),
        }
    if mode == 'joint':
        return {
            'vec_values': _to_device(host_data['vec_values'], device),
            'vec_prefix': _to_device(host_data['vec_prefix'], device),
            'vec_indices': _to_device(host_data['vec_indices'], device),
            'seq_values': _to_device(host_data['seq_values'], device),
            'seq_prefix': _to_device(host_data['seq_prefix'], device),
            'seq_lengths': _to_device(host_data['seq_lengths'], device),
        }
    raise ValueError(f'unknown mode: {mode}')


def _forward_mode(mode, model, infer_models, dev_inputs):
    if mode == 'baseline':
        return model(dev_inputs['full'])
    if mode == 'multislice':
        return infer_models['multislice'](dev_inputs['full'])
    if mode == 'multislice_seq':
        return infer_models['multislice_seq'](
            dev_inputs['non_seq_full'],
            dev_inputs['seq_values'],
            dev_inputs['seq_prefix'],
            dev_inputs['seq_lengths'],
        )
    if mode == 'seq_only':
        return infer_models['seq_only'](
            dev_inputs['non_seq_compact'],
            dev_inputs['seq_values'],
            dev_inputs['seq_prefix'],
            dev_inputs['seq_lengths'],
        )
    if mode == 'joint':
        return infer_models['joint'](
            dev_inputs['vec_values'],
            dev_inputs['vec_prefix'],
            dev_inputs['vec_indices'],
            dev_inputs['seq_values'],
            dev_inputs['seq_prefix'],
            dev_inputs['seq_lengths'],
        )
    raise ValueError(f'unknown mode: {mode}')


def benchmark_mode(mode, model, infer_models, host_data, device, iters, warmup, include_h2d):
    if not include_h2d:
        dev_inputs = _prepare_inputs(mode, host_data, device)

    _sync_if_cuda(device)
    with torch.no_grad():
        for _ in range(warmup):
            if include_h2d:
                dev_inputs = _prepare_inputs(mode, host_data, device)
            _ = _forward_mode(mode, model, infer_models, dev_inputs)

    _sync_if_cuda(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            if include_h2d:
                dev_inputs = _prepare_inputs(mode, host_data, device)
            y = _forward_mode(mode, model, infer_models, dev_inputs)
    _sync_if_cuda(device)
    t1 = time.perf_counter()

    e2e_ms = (t1 - t0) * 1000.0 / iters
    return e2e_ms, y


def h2d_only_ms(mode, host_data, device, iters, warmup):
    if not device.startswith('cuda'):
        return 0.0

    for _ in range(warmup):
        _ = _prepare_inputs(mode, host_data, device)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = _prepare_inputs(mode, host_data, device)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / iters


def main():
    parser = argparse.ArgumentParser(description='Unified 5-stage benchmark for DIN Emblayer optimizations')
    parser.add_argument('--mode', type=str, choices=['baseline', 'multislice', 'multislice_seq', 'seq_only', 'joint', 'all'], default='all')
    parser.add_argument('--iters', type=int, default=1000)
    parser.add_argument('--warmup', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--max_seq_len', type=int, default=256)
    parser.add_argument('--avg_seq_len', type=float, default=8.0)
    parser.add_argument('--include_h2d', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    device = 'cpu'
    if not args.cpu and torch.cuda.is_available():
        device = 'cuda:0'

    modes = ['baseline', 'multislice', 'multislice_seq', 'seq_only', 'joint'] if args.mode == 'all' else [args.mode]

    if device == 'cpu' and any(m != 'baseline' for m in modes):
        raise RuntimeError('optimized modes require CUDA device')

    model, feature_columns = build_din_model(device=device, max_seq_len=args.max_seq_len)
    host_data = build_inputs(
        model=model,
        feature_columns=feature_columns,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        avg_seq_len=args.avg_seq_len,
    )

    print('device:', device)
    print('batch_size:', args.batch_size)
    print('max_seq_len:', args.max_seq_len)
    print('avg_seq_len(sampled):', round(host_data['stats']['avg_len'], 4),
          'min/max:', host_data['stats']['min_len'], '/', host_data['stats']['max_len'])
    print('include_h2d:', args.include_h2d)
    for key in ['baseline', 'multislice', 'multislice_seq', 'seq_only', 'joint']:
        print(f'input_bytes {key}:', host_data['bytes'][key])

    multislice_ext = None
    seq_ext = None
    vec_ext = None
    infer_models = {}

    if any(m in ('multislice', 'multislice_seq') for m in modes):
        multislice_ext = load_multislice_extension()
    if any(m in ('multislice_seq', 'seq_only', 'joint') for m in modes):
        seq_ext = load_emblayer_seq_extension()
    if 'joint' in modes:
        vec_ext = load_emblayer_vec_extension()

    if 'multislice' in modes:
        infer_models['multislice'] = DINMultiSliceInfer(
            model=model,
            compact_layout=host_data['compact_layout'],
            starts=host_data['starts'],
            span_lengths=host_data['span_lengths'],
            multislice_ext=multislice_ext,
        ).to(device).eval()

    if 'multislice_seq' in modes:
        infer_models['multislice_seq'] = DINMultiSliceSeqInfer(
            model=model,
            seq_feature_info=host_data['seq_feature_info'],
            compact_layout=host_data['compact_layout'],
            starts=host_data['starts'],
            span_lengths=host_data['span_lengths'],
            input_dim=host_data['input_dim'],
            num_seq_per_sample=host_data['num_seq_per_sample'],
            max_seq_len=host_data['max_seq_len'],
            multislice_ext=multislice_ext,
            emblayer_seq_ext=seq_ext,
        ).to(device).eval()

    if 'seq_only' in modes:
        infer_models['seq_only'] = DINSeqOnlyInfer(
            model=model,
            seq_feature_info=host_data['seq_feature_info'],
            compact_layout=host_data['compact_layout'],
            input_dim=host_data['input_dim'],
            num_seq_per_sample=host_data['num_seq_per_sample'],
            max_seq_len=host_data['max_seq_len'],
            emblayer_seq_ext=seq_ext,
        ).to(device).eval()

    if 'joint' in modes:
        infer_models['joint'] = DINJointInfer(
            model=model,
            seq_feature_info=host_data['seq_feature_info'],
            input_dim=host_data['input_dim'],
            num_seq_per_sample=host_data['num_seq_per_sample'],
            max_seq_len=host_data['max_seq_len'],
            emblayer_seq_ext=seq_ext,
            emblayer_vec_ext=vec_ext,
        ).to(device).eval()

    outputs = {}
    e2e = {}
    h2d = {}

    for mode in modes:
        e2e_ms, y = benchmark_mode(
            mode=mode,
            model=model,
            infer_models=infer_models,
            host_data=host_data,
            device=device,
            iters=args.iters,
            warmup=args.warmup,
            include_h2d=args.include_h2d,
        )
        outputs[mode] = y
        e2e[mode] = e2e_ms
        print(f'{mode} avg latency (ms):', round(e2e_ms, 6))

        if args.include_h2d:
            h2d_ms = h2d_only_ms(mode, host_data, device, args.iters, args.warmup)
            h2d[mode] = h2d_ms
            print(f'{mode} h2d-only (ms):', round(h2d_ms, 6))
            print(f'{mode} est-compute(ms):', round(max(e2e_ms - h2d_ms, 0.0), 6))

    if 'baseline' in outputs:
        base = outputs['baseline']
        for mode in modes:
            if mode == 'baseline':
                continue
            diff = torch.max(torch.abs(base - outputs[mode])).item()
            print(f'max |baseline - {mode}|:', diff)
            if diff > 1e-6:
                raise RuntimeError(f'baseline and {mode} outputs mismatch')

    if 'baseline' in e2e:
        for mode in modes:
            if mode == 'baseline':
                continue
            print(f'speedup baseline/{mode}:', round(e2e['baseline'] / e2e[mode], 6))
    if 'baseline' in h2d:
        for mode in modes:
            if mode == 'baseline' or mode not in h2d:
                continue
            print(f'h2d speedup baseline/{mode}:', round(h2d['baseline'] / h2d[mode], 6))


if __name__ == '__main__':
    main()