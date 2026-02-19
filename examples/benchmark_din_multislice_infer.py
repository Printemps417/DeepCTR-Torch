# -*- coding: utf-8 -*-
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn

from deepctr_torch.inputs import DenseFeat, combined_dnn_input, embedding_lookup, get_varlen_pooling_list, maxlen_lookup, varlen_embedding_lookup
from export_din_frozen_graph import build_model_and_input
from torch.utils.cpp_extension import load


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


class DINMultiSliceInfer(nn.Module):
    def __init__(self, model, multislice_ext):
        super().__init__()
        self.model = model
        self.multislice_ext = multislice_ext

        self.sparse_feature_columns = model.sparse_feature_columns
        self.dense_feature_columns = [
            feat for feat in model.dnn_feature_columns if isinstance(feat, DenseFeat)
        ]

        sparse_starts = [model.feature_index[feat.name][0] for feat in self.sparse_feature_columns]
        sparse_lengths = [model.feature_index[feat.name][1] - model.feature_index[feat.name][0]
                          for feat in self.sparse_feature_columns]

        dense_starts = [model.feature_index[feat.name][0] for feat in self.dense_feature_columns]
        dense_lengths = [feat.dimension for feat in self.dense_feature_columns]

        self.sparse_starts = torch.tensor(sparse_starts, dtype=torch.int64)
        self.sparse_lengths = torch.tensor(sparse_lengths, dtype=torch.int64)
        self.dense_starts = torch.tensor(dense_starts, dtype=torch.int64)
        self.dense_lengths = torch.tensor(dense_lengths, dtype=torch.int64)

    def forward(self, x):
        sparse_block = self.multislice_ext.multislice(x, self.sparse_starts, self.sparse_lengths)

        if len(self.dense_feature_columns) > 0:
            dense_block = self.multislice_ext.multislice(x, self.dense_starts, self.dense_lengths)
        else:
            dense_block = None

        sparse_emb_by_name = {}
        for i, feat in enumerate(self.sparse_feature_columns):
            ids = sparse_block[:, i:i + 1].long()
            sparse_emb_by_name[feat.name] = self.model.embedding_dict[feat.embedding_name](ids)

        dense_value_list = []
        if dense_block is not None:
            cursor = 0
            for feat in self.dense_feature_columns:
                next_cursor = cursor + feat.dimension
                dense_value_list.append(dense_block[:, cursor:next_cursor].float())
                cursor = next_cursor

        query_emb_list = [sparse_emb_by_name[name] for name in self.model.history_feature_list]

        keys_emb_list = embedding_lookup(
            x,
            self.model.embedding_dict,
            self.model.feature_index,
            self.model.history_feature_columns,
            return_feat_list=self.model.history_fc_names,
            to_list=True,
        )

        dnn_input_emb_list = [sparse_emb_by_name[feat.name] for feat in self.sparse_feature_columns]

        sequence_embed_dict = varlen_embedding_lookup(
            x,
            self.model.embedding_dict,
            self.model.feature_index,
            self.model.sparse_varlen_feature_columns,
        )
        sequence_embed_list = get_varlen_pooling_list(
            sequence_embed_dict,
            x,
            self.model.feature_index,
            self.model.sparse_varlen_feature_columns,
            self.model.device,
        )
        dnn_input_emb_list += sequence_embed_list

        deep_input_emb = torch.cat(dnn_input_emb_list, dim=-1)
        query_emb = torch.cat(query_emb_list, dim=-1)
        keys_emb = torch.cat(keys_emb_list, dim=-1)

        keys_length_feature_name = [feat.length_name for feat in self.model.varlen_sparse_feature_columns
                                    if feat.length_name is not None]
        keys_length = torch.squeeze(maxlen_lookup(x, self.model.feature_index, keys_length_feature_name), 1)

        hist = self.model.attention(query_emb, keys_emb, keys_length)

        deep_input_emb = torch.cat((deep_input_emb, hist), dim=-1)
        deep_input_emb = deep_input_emb.view(deep_input_emb.size(0), -1)

        dnn_input = combined_dnn_input([deep_input_emb], dense_value_list)
        dnn_output = self.model.dnn(dnn_input)
        dnn_logit = self.model.dnn_linear(dnn_output)
        return self.model.out(dnn_logit)


def build_variable_length_padded_input(model, batch_size, device, seed=2026):
    rng = np.random.default_rng(seed)
    input_dim = max(end for _, end in model.feature_index.values())
    x_np = np.zeros((batch_size, input_dim), dtype=np.float32)

    sparse_feature_columns = model.sparse_feature_columns
    dense_feature_columns = [feat for feat in model.dnn_feature_columns if isinstance(feat, DenseFeat)]
    varlen_feature_columns = model.varlen_sparse_feature_columns

    for feat in sparse_feature_columns:
        start, end = model.feature_index[feat.name]
        vals = rng.integers(low=1, high=feat.vocabulary_size, size=(batch_size,), endpoint=False)
        x_np[:, start:end] = vals.reshape(batch_size, 1)

    for feat in dense_feature_columns:
        start, end = model.feature_index[feat.name]
        vals = rng.random((batch_size, feat.dimension), dtype=np.float32)
        x_np[:, start:end] = vals

    if len(varlen_feature_columns) > 0:
        maxlen = varlen_feature_columns[0].maxlen
        if batch_size == 1:
            seq_lengths = np.array([maxlen], dtype=np.int64)
        else:
            seq_lengths = (np.arange(batch_size) % maxlen) + 1
            rng.shuffle(seq_lengths)

        for feat in varlen_feature_columns:
            start, end = model.feature_index[feat.name]
            seq_block = np.zeros((batch_size, feat.maxlen), dtype=np.float32)
            for i in range(batch_size):
                valid_len = int(seq_lengths[i])
                seq_vals = rng.integers(low=1, high=feat.vocabulary_size, size=(valid_len,), endpoint=False)
                seq_block[i, :valid_len] = seq_vals.astype(np.float32)
            x_np[:, start:end] = seq_block

        length_name_to_values = {}
        for feat in varlen_feature_columns:
            if feat.length_name is not None:
                if feat.length_name not in length_name_to_values:
                    length_name_to_values[feat.length_name] = seq_lengths.astype(np.float32)

        for length_name, length_values in length_name_to_values.items():
            start, end = model.feature_index[length_name]
            x_np[:, start:end] = length_values.reshape(batch_size, 1)

    x = torch.from_numpy(x_np).to(device).contiguous()
    return x


def main():
    parser = argparse.ArgumentParser(description='Benchmark DIN inference baseline vs MultiSlice')
    parser.add_argument('--mode', type=str, choices=['baseline', 'multislice'], required=True)
    parser.add_argument('--iters', type=int, default=2000)
    parser.add_argument('--warmup', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    device = 'cpu'
    if not args.cpu and torch.cuda.is_available():
        device = 'cuda:0'

    model, _, _, _ = build_model_and_input(device=device, batch_size=1)
    sample_input = build_variable_length_padded_input(
        model=model,
        batch_size=args.batch_size,
        device=device,
    )

    if args.mode == 'multislice':
        if device == 'cpu':
            raise RuntimeError('multislice mode requires CUDA')
        multislice_ext = load_multislice_extension()
        infer_model = DINMultiSliceInfer(model, multislice_ext).to(device).eval()
    else:
        infer_model = model.eval()

    if device.startswith('cuda'):
        torch.cuda.synchronize()

    with torch.no_grad():
        for _ in range(args.warmup):
            _ = infer_model(sample_input)

    if device.startswith('cuda'):
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(args.iters):
            y = infer_model(sample_input)
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    ms = (t1 - t0) * 1000.0 / args.iters
    print('mode:', args.mode)
    print('device:', device)
    print('batch_size:', args.batch_size)
    print('avg latency (ms):', round(ms, 6))
    print('output shape:', tuple(y.shape))
    if len(model.varlen_sparse_feature_columns) > 0:
        length_name = model.varlen_sparse_feature_columns[0].length_name
        if length_name is not None:
            s, e = model.feature_index[length_name]
            seq_len_vals = sample_input[:, s:e].detach().cpu().numpy().reshape(-1)
            print('seq_length unique values in batch:', np.unique(seq_len_vals).astype(int).tolist())


if __name__ == '__main__':
    main()
