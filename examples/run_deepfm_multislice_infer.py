# -*- coding: utf-8 -*-
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch.utils.cpp_extension import load

from deepctr_torch.inputs import DenseFeat, SparseFeat, combined_dnn_input
from deepctr_torch.models import DeepFM


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
        verbose=True,
    )


class DeepFMMultiSliceInfer(nn.Module):
    def __init__(self, model, multislice_ext):
        super().__init__()
        self.model = model
        self.multislice_ext = multislice_ext

        self.linear_sparse_features = model.linear_model.sparse_feature_columns
        self.linear_dense_features = model.linear_model.dense_feature_columns

        self.dnn_sparse_features = [
            feat for feat in model.dnn_feature_columns if isinstance(feat, SparseFeat)
        ]
        self.dnn_dense_features = [
            feat for feat in model.dnn_feature_columns if isinstance(feat, DenseFeat)
        ]

        self.sparse_starts = torch.arange(0, len(self.dnn_sparse_features), dtype=torch.int64)
        self.sparse_lengths = torch.ones(len(self.dnn_sparse_features), dtype=torch.int64)

        dense_start = len(self.dnn_sparse_features)
        self.dense_starts = torch.tensor([dense_start], dtype=torch.int64)
        self.dense_lengths = torch.tensor([len(self.dnn_dense_features)], dtype=torch.int64)

    def forward(self, x):
        sparse_x = self.multislice_ext.multislice(x, self.sparse_starts, self.sparse_lengths)
        dense_x = self.multislice_ext.multislice(x, self.dense_starts, self.dense_lengths)

        linear_sparse_embedding_list = []
        for i, feat in enumerate(self.linear_sparse_features):
            feat_ids = sparse_x[:, i:i + 1].long()
            linear_sparse_embedding_list.append(self.model.linear_model.embedding_dict[feat.embedding_name](feat_ids))

        linear_logit = torch.zeros([x.shape[0], 1], device=x.device)
        if len(linear_sparse_embedding_list) > 0:
            sparse_embedding_cat = torch.cat(linear_sparse_embedding_list, dim=-1)
            sparse_feat_logit = torch.sum(sparse_embedding_cat, dim=-1, keepdim=False)
            linear_logit += sparse_feat_logit
        if len(self.linear_dense_features) > 0:
            linear_logit += dense_x.matmul(self.model.linear_model.weight)

        sparse_embedding_list = []
        for i, feat in enumerate(self.dnn_sparse_features):
            feat_ids = sparse_x[:, i:i + 1].long()
            sparse_embedding_list.append(self.model.embedding_dict[feat.embedding_name](feat_ids))

        dense_value_list = [dense_x[:, i:i + 1] for i in range(dense_x.shape[1])]

        logit = linear_logit
        if self.model.use_fm and len(sparse_embedding_list) > 0:
            fm_input = torch.cat(sparse_embedding_list, dim=1)
            logit += self.model.fm(fm_input)

        if self.model.use_dnn:
            dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
            dnn_output = self.model.dnn(dnn_input)
            dnn_logit = self.model.dnn_linear(dnn_output)
            logit += dnn_logit

        return self.model.out(logit)


def build_input_and_model(device):
    data = pd.read_csv('./criteo_sample.txt')

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

    sample_input = np.concatenate(
        [np.expand_dims(data[name].values[:1], axis=1) for name in model.feature_index],
        axis=-1,
    ).astype('float32')
    sample_input = torch.from_numpy(sample_input).to(device)
    return model, sample_input


if __name__ == '__main__':
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this MultiSlice demo, but no CUDA device was found.')

    device = 'cuda:0'
    multislice_ext = load_multislice_extension()
    model, sample_input = build_input_and_model(device)

    infer_model = DeepFMMultiSliceInfer(model, multislice_ext).to(device).eval()

    with torch.no_grad():
        y_pred = infer_model(sample_input)

    print('Inference done with MultiSlice CUDA op.')
    print('input shape:', tuple(sample_input.shape))
    print('output shape:', tuple(y_pred.shape))
    print('output value:', y_pred.detach().cpu().numpy().reshape(-1).tolist())
