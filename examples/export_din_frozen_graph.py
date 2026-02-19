# -*- coding: utf-8 -*-
import argparse
import os

import numpy as np
import onnx
import torch

from deepctr_torch.inputs import DenseFeat, SparseFeat, VarLenSparseFeat
from deepctr_torch.models.din import DIN
from onnx_multislice_converter import convert_onnx_to_multislice


def get_din_feature_config():
    feature_columns = [
        SparseFeat('user', 3, embedding_dim=8),
        SparseFeat('gender', 2, embedding_dim=8),
        SparseFeat('item', 3 + 1, embedding_dim=8),
        SparseFeat('item_gender', 2 + 1, embedding_dim=8),
        DenseFeat('score', 1),
    ]

    feature_columns += [
        VarLenSparseFeat(SparseFeat('hist_item', 3 + 1, embedding_dim=8), 4, length_name='seq_length'),
        VarLenSparseFeat(SparseFeat('hist_item_gender', 2 + 1, embedding_dim=8), 4, length_name='seq_length'),
    ]
    behavior_feature_list = ['item', 'item_gender']
    return feature_columns, behavior_feature_list


def get_din_feature_dict(sample_size=3):
    uid = np.array([0, 1, 2])[:sample_size]
    ugender = np.array([0, 1, 0])[:sample_size]
    iid = np.array([1, 2, 3])[:sample_size]
    igender = np.array([1, 2, 1])[:sample_size]
    score = np.array([0.1, 0.2, 0.3], dtype='float32')[:sample_size]

    hist_iid = np.array([[1, 2, 3, 0], [1, 2, 3, 0], [1, 2, 0, 0]])[:sample_size]
    hist_igender = np.array([[1, 1, 2, 0], [2, 1, 1, 0], [2, 1, 0, 0]])[:sample_size]
    behavior_length = np.array([3, 3, 2])[:sample_size]

    return {
        'user': uid,
        'gender': ugender,
        'item': iid,
        'item_gender': igender,
        'score': score,
        'hist_item': hist_iid,
        'hist_item_gender': hist_igender,
        'seq_length': behavior_length,
    }


def build_model_and_input(device, batch_size=1):
    feature_columns, behavior_feature_list = get_din_feature_config()
    model = DIN(feature_columns, behavior_feature_list, device=device, att_weight_normalization=True)
    model.eval()

    feature_dict = get_din_feature_dict(sample_size=max(batch_size, 1))

    sample_input_list = []
    for name in model.feature_index:
        values = feature_dict[name][:batch_size]
        if len(values.shape) == 1:
            values = np.expand_dims(values, axis=1)
        sample_input_list.append(values)

    sample_input = np.concatenate(sample_input_list, axis=-1).astype('float32')
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
    parser.add_argument('--multislice_onnx_output', type=str, default='./din_frozen_multislice.onnx')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=1)
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
    )

    if args.checkpoint:
        state_dict = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

    os.makedirs(os.path.dirname(os.path.abspath(args.onnx_output)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.torchscript_output)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.multislice_onnx_output)), exist_ok=True)

    export_onnx(model, sample_input, args.onnx_output, args.opset)
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
    print('TorchScript frozen:', os.path.abspath(args.torchscript_output))
    print('MultiSlice ONNX:', os.path.abspath(args.multislice_onnx_output))
    print('sparse+dense column range:', (sparse_dense_start, sparse_dense_end))
    print('removed Slice nodes:', removed)
    print('inserted MultiSlice nodes:', inserted)
