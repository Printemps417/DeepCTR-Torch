# -*- coding: utf-8 -*-
import argparse
import os

import onnx
import torch

from export_din_frozen_graph import build_model_and_input, export_onnx
from onnx_multislice_converter import convert_onnx_to_multislice


def main():
    parser = argparse.ArgumentParser(
        description='Three-step DIN pipeline: export ONNX -> convert MultiSlice -> run one forward'
    )
    parser.add_argument('--onnx_output', type=str, default='./din_frozen.onnx')
    parser.add_argument('--multislice_onnx_output', type=str, default='./din_frozen_multislice.onnx')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--opset', type=int, default=18)
    parser.add_argument('--multislice_min_group_size', type=int, default=2)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    device = 'cpu'
    if not args.cpu and torch.cuda.is_available():
        device = 'cuda:0'

    os.makedirs(os.path.dirname(os.path.abspath(args.onnx_output)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.multislice_onnx_output)), exist_ok=True)

    model, sample_input, sparse_dense_start, sparse_dense_end = build_model_and_input(
        device=device,
        batch_size=args.batch_size,
    )

    print('[Step 1/3] Export initial DIN ONNX...')
    export_onnx(model, sample_input, args.onnx_output, args.opset)
    print('initial ONNX:', os.path.abspath(args.onnx_output))

    print('[Step 2/3] Convert ONNX Slice -> MultiSlice (sparse+dense only)...')
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
    print('converted ONNX:', os.path.abspath(args.multislice_onnx_output))
    print('sparse+dense column range:', (sparse_dense_start, sparse_dense_end))
    print('removed Slice nodes:', removed)
    print('inserted MultiSlice nodes:', inserted)

    print('[Step 3/3] Run one forward on DIN (PyTorch)...')
    model.eval()
    with torch.no_grad():
        y_pred = model(sample_input)
    print('forward input shape:', tuple(sample_input.shape))
    print('forward output shape:', tuple(y_pred.shape))
    print('forward output value:', y_pred.detach().cpu().numpy().reshape(-1).tolist())

    print('Done.')


if __name__ == '__main__':
    main()
