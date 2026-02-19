# -*- coding: utf-8 -*-
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

from deepctr_torch.inputs import DenseFeat, SparseFeat, VarLenSparseFeat
from deepctr_torch.models.din import DIN


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


def build_joint_compressed_inputs(model, feature_columns, batch_size=8, max_seq_len=64, avg_seq_len=8, seed=2026):
    rng = np.random.default_rng(seed)
    input_dim = max(end for _, end in model.feature_index.values())
    full = np.zeros((batch_size, input_dim), dtype=np.float32)

    sparse_feats = [f for f in feature_columns if isinstance(f, SparseFeat)]
    dense_feats = [f for f in feature_columns if isinstance(f, DenseFeat)]
    varlen_feats = [f for f in feature_columns if isinstance(f, VarLenSparseFeat)]

    lengths = rng.poisson(lam=max(1.0, float(avg_seq_len)), size=(batch_size,))
    lengths = np.clip(lengths, 1, max_seq_len).astype(np.int32)

    seq_length_name = None
    for feat in varlen_feats:
        if feat.length_name is not None:
            seq_length_name = feat.length_name

    # non-seq columns for emblayerVec
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
    if seq_length_name is not None:
        s, e = model.feature_index[seq_length_name]
        full[:, s:e] = lengths.reshape(batch_size, 1).astype(np.float32)
        non_seq_spans.append((s, e))

    # seq columns for emblayerSeq
    seq_feature_info = []
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
        seq_feature_info.append({'name': feat.name, 'col_start': s, 'col_end': e, 'maxlen': feat.maxlen})

    # build emblayerVec compressed tensors
    vec_values = []
    vec_indices = []
    vec_prefix = [0]
    sorted_spans = sorted(non_seq_spans, key=lambda x: x[0])
    for b in range(batch_size):
        for s, e in sorted_spans:
            for col in range(s, e):
                vec_values.append(full[b, col])
                vec_indices.append(col)
        vec_prefix.append(len(vec_values))

    vec_values = np.asarray(vec_values, dtype=np.float32)
    vec_indices = np.asarray(vec_indices, dtype=np.int32)
    vec_prefix = np.asarray(vec_prefix, dtype=np.int32)

    # build emblayerSeq compressed tensors (single iobuffer for all seq feats)
    num_seq_per_sample = len(seq_feature_info)
    seq_prefix = np.arange(0, (batch_size + 1) * num_seq_per_sample, num_seq_per_sample, dtype=np.int32)
    seq_lengths = np.repeat(lengths, repeats=num_seq_per_sample).astype(np.int32)
    seq_values = []
    for b in range(batch_size):
        for info in seq_feature_info:
            seq_values.append(seq_tokens_by_feature[info['name']][b])
    seq_values = np.concatenate(seq_values, axis=0).astype(np.int32)

    return {
        'full': torch.from_numpy(full),
        'vec_values': torch.from_numpy(vec_values),
        'vec_prefix': torch.from_numpy(vec_prefix),
        'vec_indices': torch.from_numpy(vec_indices),
        'seq_values': torch.from_numpy(seq_values),
        'seq_prefix': torch.from_numpy(seq_prefix),
        'seq_lengths': torch.from_numpy(seq_lengths),
        'seq_feature_info': seq_feature_info,
        'input_dim': input_dim,
        'num_seq_per_sample': num_seq_per_sample,
        'max_seq_len': max_seq_len,
    }


class DINEmblayerJointInfer(nn.Module):
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
        x = self.emblayer_vec_ext.emblayer_vec(
            vec_values,
            vec_prefix,
            vec_indices,
            self.input_dim,
            0.0,
        )

        seq_padded = self.emblayer_seq_ext.emblayer_seq(
            seq_values,
            seq_prefix,
            seq_lengths,
            0,
            self.num_seq_per_sample,
            self.max_seq_len,
        )

        for seq_idx, info in enumerate(self.seq_feature_info):
            seq_2d = seq_padded[:, seq_idx, :].to(x.dtype)
            x[:, info['col_start']:info['col_end']] = seq_2d

        return self.model(x)


if __name__ == '__main__':
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for joint Emblayer inference demo.')

    device = 'cuda:0'
    model, feature_columns = build_din_model(device=device, max_seq_len=64)
    data = build_joint_compressed_inputs(model, feature_columns, batch_size=16, max_seq_len=64, avg_seq_len=8)

    emblayer_seq_ext = load_emblayer_seq_extension()
    emblayer_vec_ext = load_emblayer_vec_extension()

    joint_model = DINEmblayerJointInfer(
        model=model,
        seq_feature_info=data['seq_feature_info'],
        input_dim=data['input_dim'],
        num_seq_per_sample=data['num_seq_per_sample'],
        max_seq_len=data['max_seq_len'],
        emblayer_seq_ext=emblayer_seq_ext,
        emblayer_vec_ext=emblayer_vec_ext,
    ).to(device).eval()

    full = data['full'].to(device)
    vec_values = data['vec_values'].to(device)
    vec_prefix = data['vec_prefix']
    vec_indices = data['vec_indices']
    seq_values = data['seq_values'].to(device)
    seq_prefix = data['seq_prefix']
    seq_lengths = data['seq_lengths']

    with torch.no_grad():
        y_base = model(full)
        y_joint = joint_model(vec_values, vec_prefix, vec_indices, seq_values, seq_prefix, seq_lengths)

    max_diff = torch.max(torch.abs(y_base - y_joint)).item()

    print('Joint Emblayer inference done')
    print('batch size:', full.size(0))
    print('baseline output shape:', tuple(y_base.shape))
    print('joint output shape:', tuple(y_joint.shape))
    print('max |baseline - joint|:', max_diff)
    print('sample baseline:', y_base[:5].detach().cpu().numpy().reshape(-1).tolist())
    print('sample joint:   ', y_joint[:5].detach().cpu().numpy().reshape(-1).tolist())

    if max_diff > 1e-6:
        raise RuntimeError('Joint Emblayer output mismatch against baseline')
