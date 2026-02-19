# -*- coding: utf-8 -*-
import argparse
import os

import numpy as np
import onnx
import torch

from deepctr_torch.inputs import DenseFeat, SparseFeat, VarLenSparseFeat
from deepctr_torch.models.din import DIN
from onnx_multislice_converter import convert_onnx_to_multislice


def get_din_feature_config(max_seq_len, num_sparse, num_seq, even_vocab_size, odd_vocab_size):
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
    behavior_feature_list = [f'sparse_{idx}' for idx in range(num_seq)]
    return feature_columns, behavior_feature_list


def build_model_and_input(device, batch_size=1, max_seq_len=128, avg_seq_len=64, num_sparse=200, num_seq=100, even_vocab_size=4096, odd_vocab_size=64, seed=2026):
    feature_columns, behavior_feature_list = get_din_feature_config(
        max_seq_len=max_seq_len,
        num_sparse=num_sparse,
        num_seq=num_seq,
        even_vocab_size=even_vocab_size,
        odd_vocab_size=odd_vocab_size,
    )
    model = DIN(feature_columns, behavior_feature_list, device=device, att_weight_normalization=True)
    model.eval()

    rng = np.random.default_rng(seed)
    input_dim = max(end for _, end in model.feature_index.values())
    sample_input = np.zeros((batch_size, input_dim), dtype=np.float32)

    sparse_features = [f for f in feature_columns if isinstance(f, SparseFeat)]
    dense_features = [f for f in feature_columns if isinstance(f, DenseFeat)]
    varlen_features = [f for f in feature_columns if isinstance(f, VarLenSparseFeat)]

    lengths = rng.poisson(lam=max(1.0, float(avg_seq_len)), size=(batch_size,))
    lengths = np.clip(lengths, 1, max_seq_len).astype(np.int32)

    for feat in sparse_features:
        s, e = model.feature_index[feat.name]
        vals = rng.integers(1, feat.vocabulary_size, size=(batch_size,), endpoint=False).astype(np.float32)
        sample_input[:, s:e] = vals.reshape(batch_size, 1)

    for feat in dense_features:
        s, e = model.feature_index[feat.name]
        vals = rng.random((batch_size, feat.dimension), dtype=np.float32)
        sample_input[:, s:e] = vals

    seq_length_name = None
    for feat in varlen_features:
        if feat.length_name is not None:
            seq_length_name = feat.length_name
    if seq_length_name is not None:
        s, e = model.feature_index[seq_length_name]
        sample_input[:, s:e] = lengths.reshape(batch_size, 1).astype(np.float32)

    for feat in varlen_features:
        s, e = model.feature_index[feat.name]
        padded = np.zeros((batch_size, feat.maxlen), dtype=np.float32)
        for row in range(batch_size):
            cur_len = int(lengths[row])
            seq_vals = rng.integers(1, feat.vocabulary_size, size=(cur_len,), endpoint=False).astype(np.float32)
            padded[row, :cur_len] = seq_vals
        sample_input[:, s:e] = padded

    sample_input = torch.from_numpy(sample_input).to(device)

    sparse_dense_names = [
        feat.name for feat in feature_columns if isinstance(feat, (SparseFeat, DenseFeat))
    ]
    sparse_dense_start = min(model.feature_index[name][0] for name in sparse_dense_names)
    sparse_dense_end = max(model.feature_index[name][1] for name in sparse_dense_names)

    return model, sample_input, sparse_dense_start, sparse_dense_end


def export_onnx(model, sample_input, output_path, opset):
    torch.onnx.export(
        model,
        sample_input,
        output_path,
        input_names=['input'],
        output_names=['output'],
        do_constant_folding=True,
        opset_version=opset,
    )


def export_torchscript_frozen(model, sample_input, output_path):
    traced = torch.jit.trace(model, sample_input)
    frozen = torch.jit.freeze(traced)
    frozen.save(output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export DIN frozen graph and convert sparse+dense slices to MultiSlice')
    parser.add_argument('--onnx_output', type=str, default='./din_frozen.onnx')
    parser.add_argument('--torchscript_output', type=str, default='./din_frozen.ts')
    parser.add_argument('--skip_torchscript', action='store_true')
    parser.add_argument('--multislice_onnx_output', type=str, default='./din_frozen_multislice.onnx')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_seq_len', type=int, default=128)
    parser.add_argument('--avg_seq_len', type=float, default=64.0)
    parser.add_argument('--num_sparse', type=int, default=200)
    parser.add_argument('--num_seq', type=int, default=100)
    parser.add_argument('--even_vocab_size', type=int, default=4096)
    parser.add_argument('--odd_vocab_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--opset', type=int, default=18)
    parser.add_argument('--multislice_min_group_size', type=int, default=2)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    device = 'cpu'
    if not args.cpu and torch.cuda.is_available():
        device = 'cuda:0'

    model, sample_input, sparse_dense_start, sparse_dense_end = build_model_and_input(
        device=device,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        avg_seq_len=args.avg_seq_len,
        num_sparse=args.num_sparse,
        num_seq=args.num_seq,
        even_vocab_size=args.even_vocab_size,
        odd_vocab_size=args.odd_vocab_size,
        seed=args.seed,
    )

    if args.checkpoint:
        state_dict = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

    os.makedirs(os.path.dirname(os.path.abspath(args.onnx_output)), exist_ok=True)
    if not args.skip_torchscript:
        os.makedirs(os.path.dirname(os.path.abspath(args.torchscript_output)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.multislice_onnx_output)), exist_ok=True)

    export_onnx(model, sample_input, args.onnx_output, args.opset)
    if not args.skip_torchscript:
        export_torchscript_frozen(model, sample_input, args.torchscript_output)

    onnx_model = onnx.load(args.onnx_output)
    onnx_model, removed, inserted = convert_onnx_to_multislice(
        onnx_model,
        min_group_size=args.multislice_min_group_size,
        domain='com.deepctr',
        version=1,
        include_col_start=sparse_dense_start,
        include_col_end=sparse_dense_end,
    )
    onnx.save(onnx_model, args.multislice_onnx_output)

    print('Export done!')
    print('ONNX:', os.path.abspath(args.onnx_output))
    if not args.skip_torchscript:
        print('TorchScript frozen:', os.path.abspath(args.torchscript_output))
    print('MultiSlice ONNX:', os.path.abspath(args.multislice_onnx_output))
    print('sparse+dense column range:', (sparse_dense_start, sparse_dense_end))
    print('removed Slice nodes:', removed)
    print('inserted MultiSlice nodes:', inserted)
