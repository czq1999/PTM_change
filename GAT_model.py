"""
@Project ：code-transformer
@File    ：model.py
@IDE     ：PyCharm
@Author  ：orange_czqqq
@Date    ：2022/7/14 14:50
@Function：GAT model
"""

from cmath import atan
from re import A, S
from cv2 import normalize
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# from 批量预处理数据.data import data1


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True, bs=4):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(bs, in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(bs, 2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.bmm(h, self.W)  # h.shape: (B, N, in_features), Wh.shape: (B, N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.bmm(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:, :self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[:, self.out_features:, :])
        # broadcast add
        e = torch.matmul(Wh1, Wh2.transpose(1, 2))
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, bs=4):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True, bs=bs) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False, bs=bs)

    def forward(self, x, adj):
        # x:图中节点的表示，adj邻接矩阵
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=-1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        # return F.log_softmax(x, dim=1)
        return x


class GraphModel(nn.Module):
    def __init__(self, bs=4):
        """图神经网络模型"""
        super(GraphModel, self).__init__()
        self.n_d = 384
        # 每个节点的表示：开始的行（256），开始的列（256），结束的行（256），结束的列（256），节点类型(23)；
        # 节点的领接矩阵，ppr距离，最短距离，祖先距离，兄弟距离

        self.nodes_start_line_embedder = nn.Embedding(256, self.n_d)
        self.nodes_start_column_embedder = nn.Embedding(256, self.n_d)
        self.nodes_end_line_embedder = nn.Embedding(256, self.n_d)
        self.nodes_end_column_embedder = nn.Embedding(256, self.n_d)
        self.nodes_type_embedder = nn.Embedding(79, 2 * self.n_d)  # ruby 79个

        # self.adj_embedder = nn.Embedding(256, self.n_d)
        self.ppr_embedder = nn.Embedding(256, self.n_d)
        self.short_path_embedder = nn.Embedding(256, self.n_d)
        self.ancestor_embedder = nn.Embedding(256, self.n_d)
        self.sibling_embedder = nn.Embedding(256, self.n_d)

        self.GAT = GAT(nfeat=768,
                       nhid=64,
                       nclass=768,
                       dropout=0.1,
                       alpha=0.2,
                       nheads=12,
                       bs=bs
                       )

        self.linear = nn.Sequential(nn.Linear(768, 1024),
                                    nn.ReLU(),
                                    nn.Linear(1024, 768))

        self.linear_query = nn.Sequential(nn.Linear(768, 1024),
                                          nn.ReLU(),
                                          nn.Linear(1024, 768))

        self.linear_key = nn.Sequential(nn.Linear(768, 1024),
                                        nn.ReLU(),
                                        nn.Linear(1024, 768))

        self.linear_value = nn.Sequential(nn.Linear(768, 1024),
                                          nn.ReLU(),
                                          nn.Linear(1024, 768))

        self.linear_attention = nn.Sequential(nn.Linear(768, 256),
                                              nn.ReLU(),
                                              nn.Linear(256, 256))
        self.dropout = nn.Dropout(0.2)

        # ratio
        self.ratio = nn.Sequential(
            nn.Linear(bs * 768, 768),
            nn.ReLU(),
            nn.Linear(768, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, *args):
        nodes_start_line, nodes_start_column, nodes_end_line, nodes_end_column, nodes_type, adj = args
        nodes_start_line = self.nodes_start_line_embedder(nodes_start_line)
        nodes_start_column = self.nodes_start_column_embedder(nodes_start_column)
        nodes_end_line = self.nodes_end_column_embedder(nodes_end_line)
        nodes_end_column = self.nodes_end_column_embedder(nodes_end_column)
        nodes_type = self.nodes_type_embedder(nodes_type)

        # ppr = self.ppr_embedder(ppr)
        # short_path = self.short_path_embedder(short_path)
        # ancestor = self.ancestor_embedder(ancestor)
        # sibiling = self.sibiling_embedder(sibiling)

        emb = torch.cat([nodes_start_line, nodes_start_column], dim=-1) - torch.cat([nodes_end_line, nodes_end_column],
                                                                                    dim=-1)
        emb = emb + nodes_type

        # mean版本
        # gat_out = self.GAT(emb, adj)
        # gat_out = self.linear(gat_out)
        # gat_out = gat_out.mean(dim=1).unsqueeze(dim=1)
        # gat_out = gat_out.expand(-1, 256, -1)

        # attention 版本
        gat_out = self.GAT(emb, adj)
        query = self.linear_query(gat_out)
        key = self.linear_key(gat_out)

        value = self.linear_value(gat_out)
        attention = self.linear_attention(query)
        # attention = torch.bmm(query, key.transpose(2, 1)) / np.sqrt(value.size(2))
        # attention = self.dropout(attention)
        # attention = torch.softmax(attention, dim=-1)
        attention = self.softmax(self.dropout(attention))
        gat_out = torch.bmm(attention, value)
        gat_out = self.dropout(self.linear(gat_out))
        gat_out = gat_out.mean(dim=1)
        ratio = torch.sigmoid(self.ratio(torch.softmax(torch.flatten(gat_out), dim=-1)))
        # ratio = torch.tensor([1]).to(gat_out.device)
        gat_out = gat_out.unsqueeze(dim=1).expand(-1, 256, -1)

        return gat_out, ratio


class GraphModel2(nn.Module):
    def __init__(self, bs=4):
        """图神经网络模型"""
        super(GraphModel, self).__init__()
        self.n_d = 384
        # 每个节点的表示：开始的行（256），开始的列（256），结束的行（256），结束的列（256），节点类型(23)；
        # 节点的领接矩阵，ppr距离，最短距离，祖先距离，兄弟距离

        self.nodes_start_line_embedder = nn.Embedding(256, self.n_d)
        self.nodes_start_column_embedder = nn.Embedding(256, self.n_d)
        self.nodes_end_line_embedder = nn.Embedding(256, self.n_d)
        self.nodes_end_column_embedder = nn.Embedding(256, self.n_d)
        self.nodes_type_embedder = nn.Embedding(79, 2 * self.n_d)  # ruby 79个

        # self.adj_embedder = nn.Embedding(256, self.n_d)
        self.ppr_embedder = nn.Embedding(256, self.n_d)
        self.short_path_embedder = nn.Embedding(256, self.n_d)
        self.ancestor_embedder = nn.Embedding(256, self.n_d)
        self.sibling_embedder = nn.Embedding(256, self.n_d)

        self.embdder1 = nn.Embedding(256, self.n_d)
        self.embdder2 = nn.Embedding(256, self.n_d)

        self.GAT = GAT(nfeat=768,
                       nhid=64,
                       nclass=768,
                       dropout=0.1,
                       alpha=0.2,
                       nheads=12,
                       bs=bs
                       )

        self.linear = nn.Linear(768, 768)
        self.linear_query = nn.Linear(768, 768)
        self.linear_key = nn.Linear(768, 768)
        self.linear_value = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.2)

        # ratio
        # self.ratio = nn.Linear(bs*768, 1)

    def forward(self, *args):
        line_tokens, line_ast_type, adj = args
        line_toknes = self.embdder1(line_tokens)
        line_ast_type = self.embdder2(line_ast_type)
        adj = self.normalize_adj(adj)

        emb = torch.cat([line_tokens, line_ast_type])
        # emb = torch.cat([nodes_start_line, nodes_start_column], dim=-1) - torch.cat([nodes_end_line, nodes_end_column],
        #                                                                             dim=-1)
        # emb = emb + nodes_type

        # mean版本
        # gat_out = self.GAT(emb, adj)
        # gat_out = self.linear(gat_out)
        # gat_out = gat_out.mean(dim=1).unsqueeze(dim=1)
        # gat_out = gat_out.expand(-1, 256, -1)

        # attention 版本
        gat_out = self.GAT(emb, adj)
        query = self.linear_query(gat_out)
        key = self.linear_key(gat_out)
        value = self.linear_value(gat_out)
        attention = torch.bmm(query, key.transpose(2, 1)) / np.sqrt(value.size(2))
        attention = self.dropout(attention)
        attention = torch.softmax(attention, dim=-1)
        gat_out = torch.bmm(attention, value)
        gat_out = self.dropout(self.linear(gat_out))
        gat_out = gat_out.mean(dim=1)
        # ratio = torch.sigmoid(self.ratio(torch.flatten(gat_out)))
        # ratio = torch.tensor([1]).to(gat_out.device)
        gat_out = gat_out.unsqueeze(dim=1).expand(-1, 256, -1)

        return gat_out

    def normalize_adj(adj):
        return adj

# if __name__ == '__main__':
#     data = data1
#     model = GraphModel()
#     y = model(*data.values())
#     print(GraphModel)
