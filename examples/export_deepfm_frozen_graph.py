# -*- coding: utf-8 -*-
import argparse
import os

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr_torch.inputs import DenseFeat, SparseFeat
from deepctr_torch.models import DeepFM
from onnx_multislice_converter import convert_onnx_to_multislice

import onnx


def build_deepfm_and_input(data_path, device, batch_size=1):
    data = pd.read_csv(data_path)

    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]

    data[sparse_features] = data[sparse_features].fillna('-1')
    data[dense_features] = data[dense_features].fillna(0)

    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    fixlen_feature_columns = [
        SparseFeat(feat, vocabulary_size=data[feat].max() + 1, embedding_dim=4)
        for feat in sparse_features
    ] + [DenseFeat(feat, 1) for feat in dense_features]

    model = DeepFM(
        linear_feature_columns=fixlen_feature_columns,
        dnn_feature_columns=fixlen_feature_columns,
        task='binary',
        l2_reg_embedding=1e-5,
        device=device,
    )
    model.eval()

    sample_dict = {
        name: data[name].values[:batch_size]
        for name in model.feature_index
    }
    sample_input = [np.expand_dims(sample_dict[name], axis=1) for name in model.feature_index]
    sample_input = np.concatenate(sample_input, axis=-1).astype('float32')
    sample_input = torch.from_numpy(sample_input).to(device)

    return model, sample_input


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
    parser = argparse.ArgumentParser(description='Export DeepFM frozen graph for Netron')
    parser.add_argument('--data_path', type=str, default='./criteo_sample.txt')
    parser.add_argument('--onnx_output', type=str, default='./deepfm_frozen.onnx')
    parser.add_argument('--torchscript_output', type=str, default='./deepfm_frozen.ts')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--opset', type=int, default=18)
    parser.add_argument('--multislice_onnx_output', type=str, default='')
    parser.add_argument('--multislice_min_group_size', type=int, default=2)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    device = 'cpu'
    if not args.cpu and torch.cuda.is_available():
        device = 'cuda:0'

    model, sample_input = build_deepfm_and_input(args.data_path, device, batch_size=args.batch_size)

    if args.checkpoint:
        state_dict = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

    os.makedirs(os.path.dirname(os.path.abspath(args.onnx_output)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.torchscript_output)), exist_ok=True)

    export_onnx(model, sample_input, args.onnx_output, args.opset)
    export_torchscript_frozen(model, sample_input, args.torchscript_output)

    if args.multislice_onnx_output:
        onnx_model = onnx.load(args.onnx_output)
        onnx_model, removed, inserted = convert_onnx_to_multislice(
            onnx_model,
            min_group_size=args.multislice_min_group_size,
            domain='com.deepctr',
            version=1,
        )
        os.makedirs(os.path.dirname(os.path.abspath(args.multislice_onnx_output)), exist_ok=True)
        onnx.save(onnx_model, args.multislice_onnx_output)
        print('MultiSlice ONNX:', os.path.abspath(args.multislice_onnx_output))
        print('removed Slice nodes:', removed)
        print('inserted MultiSlice nodes:', inserted)

    print('Export done!')
    print('ONNX:', os.path.abspath(args.onnx_output))
    print('TorchScript frozen:', os.path.abspath(args.torchscript_output))
