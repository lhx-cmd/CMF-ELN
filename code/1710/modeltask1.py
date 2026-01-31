import torch
import torch.nn as nn
import torch.nn.functional as F
# from keras import backend as K
from collections import defaultdict
import numpy as np
import copy
import os
from tqdm import tqdm
from torch_geometric.nn import VGAE

from torch_geometric.nn import GCNConv, RGCNConv
import math

# from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder, BertPooler
# from transformers import AutoConfig, BertModel

EPS = 1e-15
MAX_LOGSTD = 10

device = 'cuda:0'


def rm_forward(key_str, j):
    key_list = key_str.split('.')
    key_list = key_list[j:]
    return '.'.join(key_list)


class Model1(torch.nn.Module):
    def __init__(self, tail_len, args, feats, edge_index, in_channels, out_channels, hidden_dim, output_dim,
                 num_classes, real_edge):
        super(Model1, self).__init__()
        self.autoencoder = GCNEncoder(args, tail_len, feats, edge_index, in_channels, out_channels, hidden_dim,
                                      output_dim, num_classes, real_edge)
        self.Linear1 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )

    def forward(self, idx):
        drug_f, z = self.autoencoder()
        return drug_f, z, idx


class Model2(torch.nn.Module):
    def __init__(self, tail_len, args, feats, edge_index, in_channels, out_channels, hidden_dim, output_dim,
                 num_classes, real_edge):
        super(Model2, self).__init__()
        self.autoencoder = GCNEncoder(args, tail_len, feats, edge_index, in_channels, out_channels, hidden_dim,
                                      output_dim, num_classes, real_edge)
        self.Linear1 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )

    def forward(self, arguments):
        gae1_embedding, gae1_z, idx = arguments
        drug_f, z = self.autoencoder()

        return drug_f, z, gae1_embedding, gae1_z, idx


class Model3(torch.nn.Module):
    def __init__(self, tail_len, args, feats, edge_index, in_channels, out_channels, hidden_dim, output_dim,
                 num_classes, real_edge):
        super(Model3, self).__init__()
        self.autoencoder = GCNEncoder(args, tail_len, feats, edge_index, in_channels, out_channels, hidden_dim,
                                      output_dim, num_classes, real_edge)
        self.Linear1 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )

    def forward(self, arguments):
        gae2_embedding, gae2_z, gae1_embedding, gae1_z, idx = arguments
        drug_f, z = self.autoencoder()

        return drug_f, z, gae2_embedding, gae2_z, gae1_embedding, gae1_z, idx


class Model4(torch.nn.Module):
    def __init__(self, tail_len, args, feats, edge_index, in_channels, out_channels, hidden_dim, output_dim,
                 num_classes, real_edge):
        super(Model4, self).__init__()
        self.autoencoder = GCNEncoder(args, tail_len, feats, edge_index, in_channels, out_channels, hidden_dim,
                                      output_dim, num_classes, real_edge)
        self.Linear1 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )

    def forward(self, arguments):
        gae3_embedding, gae3_z, gae2_embedding, gae2_z, gae1_embedding, gae1_z, idx = arguments
        drug_f, z = self.autoencoder()

        return drug_f, z, gae3_embedding, gae3_z, gae2_embedding, gae2_z, gae1_embedding, gae1_z, idx


class Model5(torch.nn.Module):
    def __init__(self, tail_len, args, feats, edge_index, in_channels, out_channels, hidden_dim, output_dim,
                 num_classes, real_edge):
        super(Model5, self).__init__()
        self.autoencoder = GCNEncoder(args, tail_len, feats, edge_index, in_channels, out_channels, hidden_dim,
                                      output_dim, num_classes, real_edge)
        self.Linear1 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )

    def forward(self, arguments):
        gae4_embedding, gae4_z, gae3_embedding, gae3_z, gae2_embedding, gae2_z, gae1_embedding, gae1_z, idx = arguments
        drug_f, z = self.autoencoder()

        return drug_f, z, gae4_embedding, gae4_z, gae3_embedding, gae3_z, gae2_embedding, gae2_z, gae1_embedding, gae1_z, idx


class FusionLayer1(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.fullConnectionLayer = nn.Sequential(
            nn.Linear(args.embedding_num * 8, args.embedding_num * 4),
            nn.ReLU(),
            nn.BatchNorm1d(args.embedding_num * 4),
            nn.Dropout(args.dropout),
            nn.Linear(args.embedding_num * 4, args.embedding_num * 2),
            nn.ReLU(),
            nn.BatchNorm1d(args.embedding_num * 2),
            nn.Dropout(args.dropout),
            nn.Linear(args.embedding_num * 2, args.event_num))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, arguments):
        gae4_embedding, gae4_z, gae3_embedding, gae3_z, gae2_embedding, gae2_z, gae1_embedding, gae1_z, idx = arguments
        drugA = idx[:, 0]  # 取第0列，起点
        drugB = idx[:, 1]  # 取第1列，终点

        Embedding = torch.cat(
            [gae1_embedding[drugA], gae2_embedding[drugA],  gae3_embedding[drugA],gae4_embedding[drugA], \
             gae1_embedding[drugB], gae2_embedding[drugB],  gae3_embedding[drugB],gae4_embedding[drugB], ], 1).float()
        return self.fullConnectionLayer(Embedding), gae4_z, gae3_z, gae2_z, gae1_z


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))




class feature_encoder(nn.Module):
    def __init__(self, tail_len, args, feats, edge_index, in_channels, out_channels, hidden_dim, output_dim,
                 num_classes, real_edge):
        super(feature_encoder, self).__init__()
        hidden_dim = 512
        dropout = 0.1
        # 定义网络层
        self.layer1 = nn.Linear(tail_len, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        # 权重初始化可以提高训练稳定性
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # 第一层
        x = self.layer1(x)
        x = self.layer_norm(x)
        x = self.relu(x)

        # 第二层
        x = self.layer2(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.dropout(x)

        return x


class GCNEncoder(torch.nn.Module):
    def __init__(self, tail_len, args, feats, edge_index, in_channels, out_channels, hidden_dim, output_dim,
                 num_classes, real_edge):
        super(GCNEncoder, self).__init__()
        # in_channels 是特征数量, out_channels * 2 是因为我们有两个GCNConv, 最后我们得到embedding大小的向量

        self.drug_features = torch.tensor(feats).to(device)
        self.chemical_features = torch.tensor(feats.T).to(device)
        self.edge_index = edge_index.to(device)
        self.real_edge = real_edge.to(device)

        self.drug_projector = feature_encoder(tail_len, args, feats, edge_index, in_channels, out_channels, hidden_dim, output_dim, num_classes, real_edge)
        self.chemical_projector = feature_encoder(args.n_drug, args, feats, edge_index, in_channels, out_channels, hidden_dim, output_dim, num_classes, real_edge)
        self.relu = nn.ReLU()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

        if hidden_dim is not None:
            self.proj_z = True
            self.lin = nn.Linear(hidden_dim, output_dim)

        self.edge_classifier = nn.Linear(2 * output_dim, num_classes)

    def forward(self):
        drug_x = self.drug_projector(self.drug_features.float())
        chemical_x = self.chemical_projector(self.chemical_features.float())

        feat_final = torch.cat([drug_x, chemical_x], dim=0)

        x1 = self.conv1(feat_final, self.edge_index).relu()
        drug_f = self.conv2(x1, self.edge_index)

        z = self.lin(drug_f)

        src_emb = z[self.edge_index[0]]  # 起点节点的embedding
        dst_emb = z[self.edge_index[1]]  # 终点节点的embedding
        edge_feat = torch.cat([src_emb, dst_emb], dim=1)
        return drug_f, edge_feat
