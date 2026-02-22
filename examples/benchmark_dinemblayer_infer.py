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


def build_din_model(
    device,
    max_seq_len=256,
    num_sparse=200,
    num_seq=100,
    even_vocab_size=500000,
    odd_vocab_size=4,
):
    if num_seq > num_sparse:
        raise ValueError('num_seq must be <= num_sparse')

    sparse_feature_columns = []
    for idx in range(num_sparse):
        if idx % 2 == 0:
            vocab_size = even_vocab_size
            embedding_dim = 32
        else:
            vocab_size = odd_vocab_size
            embedding_dim = 16
        sparse_feature_columns.append(
            SparseFeat(f'sparse_{idx}', vocab_size, embedding_dim=embedding_dim)
        )

    history_feature_names = [f'sparse_{idx}' for idx in range(num_seq)]
    varlen_feature_columns = []
    for idx in range(num_seq):
        if idx % 2 == 0:
            vocab_size = even_vocab_size
            embedding_dim = 32
        else:
            vocab_size = odd_vocab_size
            embedding_dim = 16
        varlen_feature_columns.append(
            VarLenSparseFeat(
                SparseFeat(f'hist_sparse_{idx}', vocab_size, embedding_dim=embedding_dim),
                max_seq_len,
                length_name='seq_length',
            )
        )

    feature_columns = sparse_feature_columns + [DenseFeat('score', 1)] + varlen_feature_columns
    model = DIN(feature_columns, history_feature_names, device=device, att_weight_normalization=True)
    model.eval()
    return model, feature_columns


def _to_device(tensor, device):
    if device.startswith('cuda'):
        return tensor.pin_memory().to(device, non_blocking=True)
    return tensor.to(device)


def _to_int64_host(tensor):
    if tensor.dtype == torch.int64:
        return tensor
    return tensor.to(torch.int64)


def _prepare_host_inputs_for_copy(mode, host_data, scheduler_side_concat):
    if mode == 'baseline':
        return {'full': host_data['full']}
    if mode == 'multislice':
        return {'full': host_data['full']}
    if scheduler_side_concat and mode in ('multislice_seq', 'seq_only', 'joint', 'joint_v2'):
        return {'full': host_data['full']}
    if mode == 'multislice_seq':
        return {
            'non_seq_full': host_data['non_seq_full'],
            'seq_values': host_data['seq_values'],
            'seq_prefix': _to_int64_host(host_data['seq_prefix']),
            'seq_lengths': _to_int64_host(host_data['seq_lengths']),
            'seq_offsets': _to_int64_host(host_data['seq_offsets']),
        }
    if mode == 'seq_only':
        return {
            'non_seq_compact': host_data['non_seq_compact'],
            'seq_values': host_data['seq_values'],
            'seq_prefix': _to_int64_host(host_data['seq_prefix']),
            'seq_lengths': _to_int64_host(host_data['seq_lengths']),
            'seq_offsets': _to_int64_host(host_data['seq_offsets']),
        }
    if mode == 'joint':
        return {
            'vec_values': host_data['vec_values'],
            'vec_prefix': _to_int64_host(host_data['vec_prefix']),
            'vec_indices': _to_int64_host(host_data['vec_indices']),
            'seq_values': host_data['seq_values'],
            'seq_prefix': _to_int64_host(host_data['seq_prefix']),
            'seq_lengths': _to_int64_host(host_data['seq_lengths']),
            'seq_offsets': _to_int64_host(host_data['seq_offsets']),
        }
    if mode == 'joint_v2':
        return {
            'user_vec_values': host_data['user_vec_values'],
            'user_vec_indices': _to_int64_host(host_data['user_vec_indices']),
            'item_vec_values': host_data['item_vec_values'],
            'item_vec_prefix': _to_int64_host(host_data['item_vec_prefix']),
            'item_vec_indices': _to_int64_host(host_data['item_vec_indices']),
            'seq_values': host_data['seq_values'],
            'seq_prefix': _to_int64_host(host_data['seq_prefix']),
            'seq_lengths': _to_int64_host(host_data['seq_lengths']),
            'seq_offsets': _to_int64_host(host_data['seq_offsets']),
        }
    raise ValueError(f'unknown mode for pure h2d: {mode}')


def _sync_if_cuda(device):
    if device.startswith('cuda'):
        torch.cuda.synchronize()


def _merge_layout(compact_layout):
    if not compact_layout:
        return []

    merged = [list(compact_layout[0])]
    for dst_s, dst_e, src_s, src_e in compact_layout[1:]:
        last = merged[-1]
        if last[1] == dst_s and last[3] == src_s:
            last[1] = dst_e
            last[3] = src_e
        else:
            merged.append([dst_s, dst_e, src_s, src_e])

    return [tuple(item) for item in merged]


def _build_seq_assign_layout(seq_feature_info, max_seq_len):
    if not seq_feature_info:
        return []

    layout = []
    for seq_idx, info in enumerate(seq_feature_info):
        if info['col_end'] - info['col_start'] != max_seq_len:
            raise ValueError('seq feature width must equal max_seq_len for packed assignment')
        layout.append((info['col_start'], info['col_end'], seq_idx, seq_idx + 1))

    merged = [list(layout[0])]
    for dst_s, dst_e, seq_s, seq_e in layout[1:]:
        last = merged[-1]
        if last[1] == dst_s and last[3] == seq_s:
            last[1] = dst_e
            last[3] = seq_e
        else:
            merged.append([dst_s, dst_e, seq_s, seq_e])

    return [tuple(item) for item in merged]


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
    user_non_seq_spans = []
    item_non_seq_spans = []
    user_sparse_count = int(getattr(model, 'user_sparse_count_for_emblayer_v2', 0))
    for feat in sparse_features:
        sparse_idx = int(feat.name.split('_')[-1])
        s, e = model.feature_index[feat.name]
        if sparse_idx < user_sparse_count:
            val = float(rng.integers(1, feat.vocabulary_size, endpoint=False))
            full[:, s:e] = np.full((batch_size, 1), val, dtype=np.float32)
            user_non_seq_spans.append((s, e))
        else:
            vals = rng.integers(1, feat.vocabulary_size, size=(batch_size,), endpoint=False).astype(np.float32)
            full[:, s:e] = vals.reshape(batch_size, 1)
            item_non_seq_spans.append((s, e))
        non_seq_spans.append((s, e))

    for feat in dense_features:
        s, e = model.feature_index[feat.name]
        vals = rng.random((batch_size, feat.dimension), dtype=np.float32)
        full[:, s:e] = vals
        non_seq_spans.append((s, e))
        item_non_seq_spans.append((s, e))

    seq_length_name = None
    for feat in varlen_features:
        if feat.length_name is not None:
            seq_length_name = feat.length_name
    if seq_length_name is not None:
        s, e = model.feature_index[seq_length_name]
        full[:, s:e] = lengths.reshape(batch_size, 1).astype(np.float32)
        non_seq_spans.append((s, e))
        item_non_seq_spans.append((s, e))

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
    user_non_seq_spans = sorted(user_non_seq_spans, key=lambda x: x[0])
    item_non_seq_spans = sorted(item_non_seq_spans, key=lambda x: x[0])

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
    seq_offsets = np.zeros((seq_lengths.shape[0] + 1,), dtype=np.int64)
    seq_offsets[1:] = np.cumsum(seq_lengths.astype(np.int64), axis=0)

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

    user_vec_values = []
    user_vec_indices = []
    if user_non_seq_spans:
        for s, e in user_non_seq_spans:
            for col in range(s, e):
                user_vec_values.append(full[0, col])
                user_vec_indices.append(col)
    user_vec_values = np.asarray(user_vec_values, dtype=np.float32)
    user_vec_indices = np.asarray(user_vec_indices, dtype=np.int32)

    item_vec_values = []
    item_vec_indices = []
    item_vec_prefix = [0]
    for row in range(batch_size):
        for s, e in item_non_seq_spans:
            for col in range(s, e):
                item_vec_values.append(full[row, col])
                item_vec_indices.append(col)
        item_vec_prefix.append(len(item_vec_values))
    item_vec_values = np.asarray(item_vec_values, dtype=np.float32)
    item_vec_indices = np.asarray(item_vec_indices, dtype=np.int32)
    item_vec_prefix = np.asarray(item_vec_prefix, dtype=np.int32)

    bytes_baseline = full.nbytes
    bytes_multislice = full.nbytes
    bytes_multislice_seq = non_seq_full.nbytes + seq_values.nbytes + seq_prefix.nbytes + seq_lengths.nbytes
    bytes_seq_only = non_seq_compact.nbytes + seq_values.nbytes + seq_prefix.nbytes + seq_lengths.nbytes
    bytes_joint = vec_values.nbytes + vec_indices.nbytes + vec_prefix.nbytes + seq_values.nbytes + seq_prefix.nbytes + seq_lengths.nbytes
    bytes_joint_v2 = user_vec_values.nbytes + user_vec_indices.nbytes + item_vec_values.nbytes + item_vec_indices.nbytes + item_vec_prefix.nbytes + seq_values.nbytes + seq_prefix.nbytes + seq_lengths.nbytes

    return {
        'full': torch.from_numpy(full),
        'non_seq_full': torch.from_numpy(non_seq_full),
        'non_seq_compact': torch.from_numpy(non_seq_compact),
        'seq_values': torch.from_numpy(seq_values),
        'seq_prefix': torch.from_numpy(seq_prefix),
        'seq_lengths': torch.from_numpy(seq_lengths),
        'seq_offsets': torch.from_numpy(seq_offsets),
        'vec_values': torch.from_numpy(vec_values),
        'vec_indices': torch.from_numpy(vec_indices),
        'vec_prefix': torch.from_numpy(vec_prefix),
        'user_vec_values': torch.from_numpy(user_vec_values),
        'user_vec_indices': torch.from_numpy(user_vec_indices),
        'item_vec_values': torch.from_numpy(item_vec_values),
        'item_vec_indices': torch.from_numpy(item_vec_indices),
        'item_vec_prefix': torch.from_numpy(item_vec_prefix),
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
            'joint_v2': bytes_joint_v2,
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
        self.compact_layout = _merge_layout(compact_layout)
        self.starts = starts
        self.span_lengths = span_lengths
        self.multislice_ext = multislice_ext

    def forward(self, full_input):
        compact = self.multislice_ext.multislice(full_input, self.starts, self.span_lengths)
        x = full_input.clone()
        for dst_s, dst_e, src_s, src_e in self.compact_layout:
            x[:, dst_s:dst_e] = compact[:, src_s:src_e]
        return self.model(x)


class DINBaselinePerInputSlice(nn.Module):
    def __init__(self, model, input_spans):
        super().__init__()
        self.model = model
        self.input_spans = input_spans

    def forward(self, full_input):
        parts = [full_input[:, span[0]:span[1]] for span in self.input_spans]
        rebuilt = torch.cat(parts, dim=-1)
        return self.model(rebuilt)


class DINSeqOnlyInfer(nn.Module):
    def __init__(self, model, seq_feature_info, compact_layout, input_dim, num_seq_per_sample, max_seq_len, emblayer_seq_ext):
        super().__init__()
        self.model = model
        self.compact_layout = _merge_layout(compact_layout)
        self.seq_assign_layout = _build_seq_assign_layout(seq_feature_info, max_seq_len)
        self.input_dim = input_dim
        self.num_seq_per_sample = num_seq_per_sample
        self.max_seq_len = max_seq_len
        self.emblayer_seq_ext = emblayer_seq_ext

    def forward(self, non_seq_compact, seq_values, seq_prefix, seq_lengths, seq_offsets):
        batch_size = non_seq_compact.size(0)
        x = torch.zeros((batch_size, self.input_dim), device=non_seq_compact.device, dtype=non_seq_compact.dtype)
        for dst_s, dst_e, src_s, src_e in self.compact_layout:
            x[:, dst_s:dst_e] = non_seq_compact[:, src_s:src_e]

        seq_padded = self.emblayer_seq_ext.emblayer_seq_fast(
            seq_values,
            seq_prefix,
            seq_lengths,
            seq_offsets,
            0,
            self.num_seq_per_sample,
            self.max_seq_len,
        )
        for dst_s, dst_e, seq_s, seq_e in self.seq_assign_layout:
            packed = seq_padded[:, seq_s:seq_e, :].reshape(seq_padded.size(0), -1)
            x[:, dst_s:dst_e] = packed.to(x.dtype)
        return self.model(x)


class DINMultiSliceSeqInfer(nn.Module):
    def __init__(self, model, seq_feature_info, compact_layout, starts, span_lengths, input_dim, num_seq_per_sample, max_seq_len, multislice_ext, emblayer_seq_ext):
        super().__init__()
        self.model = model
        self.seq_assign_layout = _build_seq_assign_layout(seq_feature_info, max_seq_len)
        self.compact_layout = _merge_layout(compact_layout)
        self.starts = starts
        self.span_lengths = span_lengths
        self.input_dim = input_dim
        self.num_seq_per_sample = num_seq_per_sample
        self.max_seq_len = max_seq_len
        self.multislice_ext = multislice_ext
        self.emblayer_seq_ext = emblayer_seq_ext

    def forward(self, non_seq_full, seq_values, seq_prefix, seq_lengths, seq_offsets):
        compact = self.multislice_ext.multislice(non_seq_full, self.starts, self.span_lengths)
        batch_size = non_seq_full.size(0)
        x = torch.zeros((batch_size, self.input_dim), device=non_seq_full.device, dtype=non_seq_full.dtype)
        for dst_s, dst_e, src_s, src_e in self.compact_layout:
            x[:, dst_s:dst_e] = compact[:, src_s:src_e]

        seq_padded = self.emblayer_seq_ext.emblayer_seq_fast(
            seq_values,
            seq_prefix,
            seq_lengths,
            seq_offsets,
            0,
            self.num_seq_per_sample,
            self.max_seq_len,
        )
        for dst_s, dst_e, seq_s, seq_e in self.seq_assign_layout:
            packed = seq_padded[:, seq_s:seq_e, :].reshape(seq_padded.size(0), -1)
            x[:, dst_s:dst_e] = packed.to(x.dtype)
        return self.model(x)


class DINJointInfer(nn.Module):
    def __init__(self, model, seq_feature_info, input_dim, num_seq_per_sample, max_seq_len, emblayer_seq_ext, emblayer_vec_ext):
        super().__init__()
        self.model = model
        self.seq_assign_layout = _build_seq_assign_layout(seq_feature_info, max_seq_len)
        self.input_dim = input_dim
        self.num_seq_per_sample = num_seq_per_sample
        self.max_seq_len = max_seq_len
        self.emblayer_seq_ext = emblayer_seq_ext
        self.emblayer_vec_ext = emblayer_vec_ext

    def forward(self, vec_values, vec_prefix, vec_indices, seq_values, seq_prefix, seq_lengths, seq_offsets):
        x = self.emblayer_vec_ext.emblayer_vec_fast(vec_values, vec_prefix, vec_indices, self.input_dim, 0.0)
        seq_padded = self.emblayer_seq_ext.emblayer_seq_fast(
            seq_values,
            seq_prefix,
            seq_lengths,
            seq_offsets,
            0,
            self.num_seq_per_sample,
            self.max_seq_len,
        )
        for dst_s, dst_e, seq_s, seq_e in self.seq_assign_layout:
            packed = seq_padded[:, seq_s:seq_e, :].reshape(seq_padded.size(0), -1)
            x[:, dst_s:dst_e] = packed.to(x.dtype)
        return self.model(x)


class DINJointInferV2(nn.Module):
    def __init__(self, model, seq_feature_info, input_dim, num_seq_per_sample, max_seq_len, emblayer_seq_ext, emblayer_vec_ext):
        super().__init__()
        self.model = model
        self.seq_assign_layout = _build_seq_assign_layout(seq_feature_info, max_seq_len)
        self.input_dim = input_dim
        self.num_seq_per_sample = num_seq_per_sample
        self.max_seq_len = max_seq_len
        self.emblayer_seq_ext = emblayer_seq_ext
        self.emblayer_vec_ext = emblayer_vec_ext

    def forward(self, user_vec_values, user_vec_indices, item_vec_values, item_vec_prefix, item_vec_indices,
                seq_values, seq_prefix, seq_lengths, seq_offsets):
        x = self.emblayer_vec_ext.emblayer_vec_v2_fast(
            user_vec_values,
            user_vec_indices,
            item_vec_values,
            item_vec_prefix,
            item_vec_indices,
            self.input_dim,
            0.0,
        )
        seq_padded = self.emblayer_seq_ext.emblayer_seq_fast(
            seq_values,
            seq_prefix,
            seq_lengths,
            seq_offsets,
            0,
            self.num_seq_per_sample,
            self.max_seq_len,
        )
        for dst_s, dst_e, seq_s, seq_e in self.seq_assign_layout:
            packed = seq_padded[:, seq_s:seq_e, :].reshape(seq_padded.size(0), -1)
            x[:, dst_s:dst_e] = packed.to(x.dtype)
        return self.model(x)


def _prepare_inputs(mode, host_data, device, scheduler_side_concat):
    if mode == 'baseline':
        return {'full': _to_device(host_data['full'], device)}
    if mode == 'multislice':
        return {'full': _to_device(host_data['full'], device)}
    if scheduler_side_concat and mode in ('multislice_seq', 'seq_only', 'joint', 'joint_v2'):
        return {'full': _to_device(host_data['full'], device)}
    if mode == 'multislice_seq':
        return {
            'non_seq_full': _to_device(host_data['non_seq_full'], device),
            'seq_values': _to_device(host_data['seq_values'], device),
            'seq_prefix': _to_device(host_data['seq_prefix'].to(torch.int64), device),
            'seq_lengths': _to_device(host_data['seq_lengths'].to(torch.int64), device),
            'seq_offsets': _to_device(host_data['seq_offsets'].to(torch.int64), device),
        }
    if mode == 'seq_only':
        return {
            'non_seq_compact': _to_device(host_data['non_seq_compact'], device),
            'seq_values': _to_device(host_data['seq_values'], device),
            'seq_prefix': _to_device(host_data['seq_prefix'].to(torch.int64), device),
            'seq_lengths': _to_device(host_data['seq_lengths'].to(torch.int64), device),
            'seq_offsets': _to_device(host_data['seq_offsets'].to(torch.int64), device),
        }
    if mode == 'joint':
        return {
            'vec_values': _to_device(host_data['vec_values'], device),
            'vec_prefix': _to_device(host_data['vec_prefix'].to(torch.int64), device),
            'vec_indices': _to_device(host_data['vec_indices'].to(torch.int64), device),
            'seq_values': _to_device(host_data['seq_values'], device),
            'seq_prefix': _to_device(host_data['seq_prefix'].to(torch.int64), device),
            'seq_lengths': _to_device(host_data['seq_lengths'].to(torch.int64), device),
            'seq_offsets': _to_device(host_data['seq_offsets'].to(torch.int64), device),
        }
    if mode == 'joint_v2':
        return {
            'user_vec_values': _to_device(host_data['user_vec_values'], device),
            'user_vec_indices': _to_device(host_data['user_vec_indices'].to(torch.int64), device),
            'item_vec_values': _to_device(host_data['item_vec_values'], device),
            'item_vec_prefix': _to_device(host_data['item_vec_prefix'].to(torch.int64), device),
            'item_vec_indices': _to_device(host_data['item_vec_indices'].to(torch.int64), device),
            'seq_values': _to_device(host_data['seq_values'], device),
            'seq_prefix': _to_device(host_data['seq_prefix'].to(torch.int64), device),
            'seq_lengths': _to_device(host_data['seq_lengths'].to(torch.int64), device),
            'seq_offsets': _to_device(host_data['seq_offsets'].to(torch.int64), device),
        }
    raise ValueError(f'unknown mode: {mode}')


def _forward_mode(mode, model, infer_models, dev_inputs, scheduler_side_concat):
    if mode == 'baseline':
        return infer_models['baseline'](dev_inputs['full'])
    if mode == 'multislice':
        return infer_models['multislice'](dev_inputs['full'])
    if scheduler_side_concat and mode in ('multislice_seq', 'seq_only', 'joint', 'joint_v2'):
        return infer_models['baseline_raw'](dev_inputs['full'])
    if mode == 'multislice_seq':
        return infer_models['multislice_seq'](
            dev_inputs['non_seq_full'],
            dev_inputs['seq_values'],
            dev_inputs['seq_prefix'],
            dev_inputs['seq_lengths'],
            dev_inputs['seq_offsets'],
        )
    if mode == 'seq_only':
        return infer_models['seq_only'](
            dev_inputs['non_seq_compact'],
            dev_inputs['seq_values'],
            dev_inputs['seq_prefix'],
            dev_inputs['seq_lengths'],
            dev_inputs['seq_offsets'],
        )
    if mode == 'joint':
        return infer_models['joint'](
            dev_inputs['vec_values'],
            dev_inputs['vec_prefix'],
            dev_inputs['vec_indices'],
            dev_inputs['seq_values'],
            dev_inputs['seq_prefix'],
            dev_inputs['seq_lengths'],
            dev_inputs['seq_offsets'],
        )
    if mode == 'joint_v2':
        return infer_models['joint_v2'](
            dev_inputs['user_vec_values'],
            dev_inputs['user_vec_indices'],
            dev_inputs['item_vec_values'],
            dev_inputs['item_vec_prefix'],
            dev_inputs['item_vec_indices'],
            dev_inputs['seq_values'],
            dev_inputs['seq_prefix'],
            dev_inputs['seq_lengths'],
            dev_inputs['seq_offsets'],
        )
    raise ValueError(f'unknown mode: {mode}')


def _build_export_payload(mode, infer_models, host_data, device, scheduler_side_concat, export_no_custom_ops):
    if export_no_custom_ops and mode in ('multislice', 'multislice_seq', 'seq_only', 'joint', 'joint_v2'):
        return infer_models['baseline_raw'], (_to_device(host_data['full'], device),), ['full_input']

    if mode == 'baseline':
        return infer_models['baseline'], (_to_device(host_data['full'], device),), ['full_input']
    if mode == 'multislice':
        return infer_models['multislice'], (_to_device(host_data['full'], device),), ['full_input']

    if scheduler_side_concat and mode in ('multislice_seq', 'seq_only', 'joint', 'joint_v2'):
        return infer_models['baseline_raw'], (_to_device(host_data['full'], device),), ['full_input']

    if mode == 'multislice_seq':
        return infer_models['multislice_seq'], (
            _to_device(host_data['non_seq_full'], device),
            _to_device(host_data['seq_values'], device),
            _to_device(host_data['seq_prefix'].to(torch.int64), device),
            _to_device(host_data['seq_lengths'].to(torch.int64), device),
            _to_device(host_data['seq_offsets'].to(torch.int64), device),
        ), ['non_seq_full', 'seq_values', 'seq_prefix', 'seq_lengths', 'seq_offsets']

    if mode == 'seq_only':
        return infer_models['seq_only'], (
            _to_device(host_data['non_seq_compact'], device),
            _to_device(host_data['seq_values'], device),
            _to_device(host_data['seq_prefix'].to(torch.int64), device),
            _to_device(host_data['seq_lengths'].to(torch.int64), device),
            _to_device(host_data['seq_offsets'].to(torch.int64), device),
        ), ['non_seq_compact', 'seq_values', 'seq_prefix', 'seq_lengths', 'seq_offsets']

    if mode == 'joint':
        return infer_models['joint'], (
            _to_device(host_data['vec_values'], device),
            _to_device(host_data['vec_prefix'].to(torch.int64), device),
            _to_device(host_data['vec_indices'].to(torch.int64), device),
            _to_device(host_data['seq_values'], device),
            _to_device(host_data['seq_prefix'].to(torch.int64), device),
            _to_device(host_data['seq_lengths'].to(torch.int64), device),
            _to_device(host_data['seq_offsets'].to(torch.int64), device),
        ), ['vec_values', 'vec_prefix', 'vec_indices', 'seq_values', 'seq_prefix', 'seq_lengths', 'seq_offsets']

    if mode == 'joint_v2':
        return infer_models['joint_v2'], (
            _to_device(host_data['user_vec_values'], device),
            _to_device(host_data['user_vec_indices'].to(torch.int64), device),
            _to_device(host_data['item_vec_values'], device),
            _to_device(host_data['item_vec_prefix'].to(torch.int64), device),
            _to_device(host_data['item_vec_indices'].to(torch.int64), device),
            _to_device(host_data['seq_values'], device),
            _to_device(host_data['seq_prefix'].to(torch.int64), device),
            _to_device(host_data['seq_lengths'].to(torch.int64), device),
            _to_device(host_data['seq_offsets'].to(torch.int64), device),
        ), ['user_vec_values', 'user_vec_indices', 'item_vec_values', 'item_vec_prefix', 'item_vec_indices', 'seq_values', 'seq_prefix', 'seq_lengths', 'seq_offsets']

    raise ValueError(f'unknown mode for export: {mode}')


def export_mode_onnx(mode, infer_models, host_data, device, scheduler_side_concat, onnx_path, opset, export_no_custom_ops):
    module, export_args, input_names = _build_export_payload(
        mode=mode,
        infer_models=infer_models,
        host_data=host_data,
        device=device,
        scheduler_side_concat=scheduler_side_concat,
        export_no_custom_ops=export_no_custom_ops,
    )
    os.makedirs(os.path.dirname(os.path.abspath(onnx_path)), exist_ok=True)
    module.eval()
    with torch.no_grad():
        torch.onnx.export(
            module,
            export_args,
            onnx_path,
            input_names=input_names,
            output_names=['output'],
            opset_version=opset,
            do_constant_folding=True,
        )


def benchmark_mode(mode, model, infer_models, host_data, device, iters, warmup, include_h2d, scheduler_side_concat):
    if not include_h2d:
        dev_inputs = _prepare_inputs(mode, host_data, device, scheduler_side_concat)

    _sync_if_cuda(device)
    with torch.no_grad():
        for _ in range(warmup):
            if include_h2d:
                dev_inputs = _prepare_inputs(mode, host_data, device, scheduler_side_concat)
            _ = _forward_mode(mode, model, infer_models, dev_inputs, scheduler_side_concat)

    _sync_if_cuda(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            if include_h2d:
                dev_inputs = _prepare_inputs(mode, host_data, device, scheduler_side_concat)
            y = _forward_mode(mode, model, infer_models, dev_inputs, scheduler_side_concat)
    _sync_if_cuda(device)
    t1 = time.perf_counter()

    e2e_ms = (t1 - t0) * 1000.0 / iters
    return e2e_ms, y


def h2d_only_ms(mode, host_data, device, iters, warmup, scheduler_side_concat):
    if not device.startswith('cuda'):
        return 0.0

    for _ in range(warmup):
        _ = _prepare_inputs(mode, host_data, device, scheduler_side_concat)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = _prepare_inputs(mode, host_data, device, scheduler_side_concat)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / iters


def pure_h2d_only_ms(mode, host_data, device, iters, warmup, scheduler_side_concat):
    if not device.startswith('cuda'):
        return 0.0

    host_inputs = _prepare_host_inputs_for_copy(mode, host_data, scheduler_side_concat)
    pinned_inputs = {}
    for key, tensor in host_inputs.items():
        pinned_inputs[key] = tensor.pin_memory()

    host_tensors = tuple(pinned_inputs.values())

    for _ in range(warmup):
        for tensor in host_tensors:
            _ = tensor.to(device, non_blocking=True)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        for tensor in host_tensors:
            _ = tensor.to(device, non_blocking=True)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / iters


def h2d_bytes_breakdown(mode, host_data, scheduler_side_concat):
    host_inputs = _prepare_host_inputs_for_copy(mode, host_data, scheduler_side_concat)
    total_bytes = 0
    data_bytes = 0
    meta_bytes = 0

    for tensor in host_inputs.values():
        bytes_cur = int(tensor.numel() * tensor.element_size())
        total_bytes += bytes_cur
        if tensor.dtype.is_floating_point:
            data_bytes += bytes_cur
        else:
            meta_bytes += bytes_cur

    return {
        'total': int(total_bytes),
        'data': int(data_bytes),
        'meta': int(meta_bytes),
    }


def main():
    parser = argparse.ArgumentParser(description='Unified 5-stage benchmark for DIN Emblayer optimizations')
    parser.add_argument('--mode', type=str, choices=['baseline', 'multislice', 'multislice_seq', 'seq_only', 'joint', 'joint_v2', 'all'], default='all')
    parser.add_argument('--iters', type=int, default=1000)
    parser.add_argument('--warmup', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--max_seq_len', type=int, default=256)
    parser.add_argument('--avg_seq_len', type=float, default=8.0)
    parser.add_argument('--num_sparse', type=int, default=200)
    parser.add_argument('--num_seq', type=int, default=100)
    parser.add_argument('--user_sparse_count', type=int, default=0,
                        help='Number of leading sparse_i treated as user-side shared features for emblayerV2')
    parser.add_argument('--even_vocab_size', type=int, default=500000)
    parser.add_argument('--odd_vocab_size', type=int, default=4)
    parser.add_argument('--scheduler_side_concat', dest='scheduler_side_concat', action='store_true')
    parser.add_argument('--rebuild_from_emblayer', dest='scheduler_side_concat', action='store_false')
    parser.add_argument('--export_onnx', type=str, default='')
    parser.add_argument('--export_opset', type=int, default=18)
    parser.add_argument('--export_no_custom_ops', action='store_true')
    parser.add_argument('--include_h2d', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    parser.set_defaults(scheduler_side_concat=True, export_no_custom_ops=True)
    args = parser.parse_args()

    device = 'cpu'
    if not args.cpu and torch.cuda.is_available():
        device = 'cuda:0'

    modes = ['baseline', 'multislice', 'multislice_seq', 'seq_only', 'joint', 'joint_v2'] if args.mode == 'all' else [args.mode]

    if args.export_onnx and args.mode == 'all':
        raise ValueError('--export_onnx requires a single --mode (not all)')

    allow_cpu_optimized_export = bool(args.export_onnx and args.export_no_custom_ops)
    if device == 'cpu' and any(m != 'baseline' for m in modes) and not allow_cpu_optimized_export:
        raise RuntimeError('optimized modes require CUDA device')

    model, feature_columns = build_din_model(
        device=device,
        max_seq_len=args.max_seq_len,
        num_sparse=args.num_sparse,
        num_seq=args.num_seq,
        even_vocab_size=args.even_vocab_size,
        odd_vocab_size=args.odd_vocab_size,
    )
    if args.user_sparse_count < 0 or args.user_sparse_count > args.num_sparse:
        raise ValueError('--user_sparse_count must be in [0, num_sparse]')
    model.user_sparse_count_for_emblayer_v2 = int(args.user_sparse_count)
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
    print('scheduler_side_concat:', args.scheduler_side_concat)
    print('user_sparse_count(emblayerV2):', args.user_sparse_count)
    for key in ['baseline', 'multislice', 'multislice_seq', 'seq_only', 'joint', 'joint_v2']:
        if args.scheduler_side_concat and key in ('multislice_seq', 'seq_only', 'joint', 'joint_v2'):
            print(f'input_bytes {key}:', host_data['bytes']['baseline'])
        else:
            print(f'input_bytes {key}:', host_data['bytes'][key])

    multislice_ext = None
    seq_ext = None
    vec_ext = None
    infer_models = {}

    if 'baseline' in modes or args.export_onnx or (args.scheduler_side_concat and any(m in ('multislice_seq', 'seq_only', 'joint') for m in modes)):
        input_spans = [model.feature_index[name] for name in model.feature_index]
        input_spans = sorted(input_spans, key=lambda x: x[0])
        infer_models['baseline'] = DINBaselinePerInputSlice(
            model=model,
            input_spans=input_spans,
        ).to(device).eval()
        infer_models['baseline_raw'] = model

    skip_custom_for_export = bool(args.export_onnx and args.export_no_custom_ops)

    if any(m in ('multislice', 'multislice_seq') for m in modes) and not (args.scheduler_side_concat and 'multislice_seq' in modes and 'multislice' not in modes) and not skip_custom_for_export:
        multislice_ext = load_multislice_extension()
    if any(m in ('multislice_seq', 'seq_only', 'joint', 'joint_v2') for m in modes) and not args.scheduler_side_concat and not skip_custom_for_export:
        seq_ext = load_emblayer_seq_extension()
    if any(m in ('joint', 'joint_v2') for m in modes) and not args.scheduler_side_concat and not skip_custom_for_export:
        vec_ext = load_emblayer_vec_extension()

    if 'multislice' in modes and not skip_custom_for_export:
        infer_models['multislice'] = DINMultiSliceInfer(
            model=model,
            compact_layout=host_data['compact_layout'],
            starts=host_data['starts'],
            span_lengths=host_data['span_lengths'],
            multislice_ext=multislice_ext,
        ).to(device).eval()

    if 'multislice_seq' in modes and not args.scheduler_side_concat and not skip_custom_for_export:
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

    if 'seq_only' in modes and not args.scheduler_side_concat and not skip_custom_for_export:
        infer_models['seq_only'] = DINSeqOnlyInfer(
            model=model,
            seq_feature_info=host_data['seq_feature_info'],
            compact_layout=host_data['compact_layout'],
            input_dim=host_data['input_dim'],
            num_seq_per_sample=host_data['num_seq_per_sample'],
            max_seq_len=host_data['max_seq_len'],
            emblayer_seq_ext=seq_ext,
        ).to(device).eval()

    if 'joint' in modes and not args.scheduler_side_concat and not skip_custom_for_export:
        infer_models['joint'] = DINJointInfer(
            model=model,
            seq_feature_info=host_data['seq_feature_info'],
            input_dim=host_data['input_dim'],
            num_seq_per_sample=host_data['num_seq_per_sample'],
            max_seq_len=host_data['max_seq_len'],
            emblayer_seq_ext=seq_ext,
            emblayer_vec_ext=vec_ext,
        ).to(device).eval()

    if 'joint_v2' in modes and not args.scheduler_side_concat and not skip_custom_for_export:
        infer_models['joint_v2'] = DINJointInferV2(
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
    h2d_pure = {}

    if args.export_onnx:
        export_mode_onnx(
            mode=args.mode,
            infer_models=infer_models,
            host_data=host_data,
            device=device,
            scheduler_side_concat=args.scheduler_side_concat,
            onnx_path=args.export_onnx,
            opset=args.export_opset,
            export_no_custom_ops=args.export_no_custom_ops,
        )
        print('exported onnx:', os.path.abspath(args.export_onnx))
        return

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
            scheduler_side_concat=args.scheduler_side_concat,
        )
        outputs[mode] = y
        e2e[mode] = e2e_ms
        print(f'{mode} avg latency (ms):', round(e2e_ms, 6))

        h2d_bytes = h2d_bytes_breakdown(mode, host_data, args.scheduler_side_concat)
        print(
            f"{mode} h2d-bytes total(B): {h2d_bytes['total']} "
            f"data(B): {h2d_bytes['data']} meta(B): {h2d_bytes['meta']}"
        )

        if args.include_h2d:
            h2d_ms = h2d_only_ms(mode, host_data, device, args.iters, args.warmup, args.scheduler_side_concat)
            h2d[mode] = h2d_ms
            print(f'{mode} h2d-only (ms):', round(h2d_ms, 6))
            h2d_pure_ms = pure_h2d_only_ms(mode, host_data, device, args.iters, args.warmup, args.scheduler_side_concat)
            h2d_pure[mode] = h2d_pure_ms
            print(f'{mode} h2d-pure (ms):', round(h2d_pure_ms, 6))
            print(f'{mode} h2d-prepare-overhead(ms):', round(max(h2d_ms - h2d_pure_ms, 0.0), 6))
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
    if 'baseline' in h2d_pure:
        for mode in modes:
            if mode == 'baseline' or mode not in h2d_pure:
                continue
            print(f'h2d-pure speedup baseline/{mode}:', round(h2d_pure['baseline'] / h2d_pure[mode], 6))


if __name__ == '__main__':
    main()