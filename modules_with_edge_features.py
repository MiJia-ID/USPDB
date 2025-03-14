# essential modules of model

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from dgl import function as fn

from torch import nn, Tensor
from torch.nn import LayerNorm, Linear
import torch.nn.functional as F
from einops import rearrange, repeat
import dgl
from einops.layers.torch import Rearrange
import sys

from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_batch
from typing import List, Optional, Tuple, Type

import torch
from torch import nn
from torch.cuda.amp import autocast

import torch
import math
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"


class EGNNConv(nn.Module):
    # Equivariant graph neural network

    def __init__(self, in_size, hidden_size, out_size, edge_feat_size=2):
        super(EGNNConv, self).__init__()

        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.edge_feat_size = edge_feat_size
        act_fn = nn.SiLU()

        # \phi_e
        self.edge_mlp = nn.Sequential(
            # +1 for the radial feature: ||x_i - x_j||^2
            nn.Linear(in_size * 2 + edge_feat_size + 1, hidden_size),
            act_fn,
            nn.Linear(hidden_size, hidden_size),
            act_fn,
        )

        # \phi_h
        self.node_mlp = nn.Sequential(
            nn.Linear(in_size + hidden_size, hidden_size),
            act_fn,
            nn.Linear(hidden_size, out_size),
        )

        # \phi_x
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            act_fn,
            nn.Linear(hidden_size, 1, bias=False),
        )

    def message(self, edges):
        """message function for EGNN"""
        # concat features for edge mlp
        if self.edge_feat_size > 0:
            f = torch.cat(
                [
                    edges.src["h"],
                    edges.dst["h"],
                    edges.data["radial"],
                    edges.data["a"],
                ],
                dim=-1,
            )
        else:
            f = torch.cat(
                [edges.src["h"], edges.dst["h"], edges.data["radial"]], dim=-1
            )

        msg_h = self.edge_mlp(f)
        msg_x = self.coord_mlp(msg_h) * edges.data["x_diff"]

        return {"msg_x": msg_x, "msg_h": msg_h}

    def forward(self, graph, node_feat, coord_feat, edge_feat=None):

        with graph.local_scope():
            # node feature
            graph.ndata["h"] = node_feat
            # coordinate feature
            graph.ndata["x"] = coord_feat
            # edge feature
            if self.edge_feat_size > 0:
                assert edge_feat is not None, "Edge features must be provided."
                graph.edata["a"] = edge_feat
            # get coordinate diff & radial features
            graph.apply_edges(fn.u_sub_v("x", "x", "x_diff"))
            graph.edata["radial"] = (
                graph.edata["x_diff"].square().sum(dim=1).unsqueeze(-1)
            )
            # normalize coordinate difference
            graph.edata["x_diff"] = graph.edata["x_diff"] / (
                    graph.edata["radial"].sqrt() + 1e-30
            )
            graph.apply_edges(self.message)
            graph.update_all(fn.copy_e("msg_x", "m"), fn.mean("m", "x_neigh"))
            graph.update_all(fn.copy_e("msg_h", "m"), fn.sum("m", "h_neigh"))

            h_neigh, x_neigh = graph.ndata["h_neigh"], graph.ndata["x_neigh"]

            h = self.node_mlp(torch.cat([node_feat, h_neigh], dim=-1))
            x = coord_feat + x_neigh

            return h, x


class BaseModule_EGNN(nn.Module):
    # Equivariant graph neural network with residue connection and identity mapping

    def __init__(self, in_size, hidden_size, out_size, edge_feat_size=2):
        super(BaseModule_EGNN, self).__init__()
        self.EGNN = EGNNConv(in_size, hidden_size, out_size, edge_feat_size=edge_feat_size)
        self.in_features = 2 * in_size
        self.out_features = out_size
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    # graph, node_feat, coord_feat, edge_feat=None
    def forward(self, input, coord_feat, h0, lamda, alpha, l, adj_matrix=None, graph=None, efeats=None):
        input = input.to(device, dtype=torch.float32)
        # coord_feat = torch.from_numpy(coord_feat).to(device, dtype=torch.float32)
        coord_feat = coord_feat.to(device, dtype=torch.float32)

        # efeats = efeats.to(device, dtype=torch.float32)
        self.EGNN = self.EGNN.to(input.device, dtype=input.dtype)

        theta = min(1, math.log(lamda / l + 1))
        if adj_matrix is not None:
            hi = torch.sparse.mm(adj_matrix, input)
        elif graph is not None:

            hi, _ = self.EGNN(graph, input, coord_feat,efeats)
        else:
            print(
                'ERROR:adj_matrix, graph and efeats must not be None at the same time! Please input the value of adj_matrix or the value of graph and efeats.')
            raise ValueError

        support = torch.cat([hi, h0], 1)
        r = (1 - alpha) * hi + alpha * h0
        output = theta * torch.mm(support, self.weight) + (1 - theta) * r
        output = output + input
        return output


class RGN_EGNN(nn.Module):
    # construct multilayer EGNN

    def __init__(self, nlayers, nfeat, nhidden, nclass, dropout, lamda, alpha, variant, heads):
        super(RGN_EGNN, self).__init__()
        self.convs = nn.ModuleList()
        self.ln = nn.LayerNorm(nhidden)

        for _ in range(nlayers):
            self.convs.append(BaseModule_EGNN(in_size=nhidden, hidden_size=nhidden, out_size=nhidden, edge_feat_size=2))

        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

    def forward(self, G, h, x,efeats):
        _layers = []
        h = F.dropout(h, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](h))
        _layers.append(layer_inner)

        for i, con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(
                # self, input, coord_feat, h0, lamda, alpha, l, adj_matrix=None, graph=None, efeats=None
                self.ln(con(input=layer_inner, coord_feat=x, h0=_layers[0], lamda=self.lamda, alpha=self.alpha, l=i + 1,adj_matrix=None, graph=G,efeats=efeats)))

        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)

        return layer_inner


class U_Module_EGNN(nn.Module):
    # Equivariant graph neural network with residue connection and identity mapping

    def __init__(self, in_size, hidden_size, out_size, edge_feat_size=2):
        super(U_Module_EGNN, self).__init__()
        self.EGNN = EGNNConv(in_size, hidden_size, out_size, edge_feat_size=edge_feat_size)
        self.in_features = 2 * in_size
        self.out_features = out_size
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    # graph, node_feat, coord_feat, edge_feat=None
    def forward(self, input, coord_feat, h0, lamda, alpha, l, adj_matrix=None, graph=None, efeats=None):
        input = input.to(device, dtype=torch.float32)
        # coord_feat = torch.from_numpy(coord_feat).to(device, dtype=torch.float32)
        coord_feat = coord_feat.to(device, dtype=torch.float32)

        # efeats = efeats.to(device, dtype=torch.float32)
        self.EGNN = self.EGNN.to(input.device, dtype=input.dtype)

        theta = min(1, math.log(lamda / l + 1))
        if adj_matrix is not None:
            hi = torch.sparse.mm(adj_matrix, input)
        elif graph is not None:
            # print("图特征:")
            # print(graph)
            hi, _ = self.EGNN(graph, input, coord_feat,efeats)
        else:
            print(
                'ERROR:adj_matrix, graph and efeats must not be None at the same time! Please input the value of adj_matrix or the value of graph and efeats.')
            raise ValueError

        support = torch.cat([hi, h0], 1)
        r = (1 - alpha) * hi + alpha * h0
        output = theta * torch.mm(support, self.weight) + (1 - theta) * r
        output = output + input
        return output


class U_GCN_EGNN(nn.Module):
    # construct multilayer EGNN
    ######################################层数消融实验#######################################################
    def __init__(self, nlayers, nfeat, nhidden, nclass, dropout, lamda, alpha, attention, variant, heads,
                 args = {'drop_n':0.0 ,'ks': [1,0.8,0.8], 'l_dim':128,'h_dim':128},drop_p =0.0):
        super(U_GCN_EGNN, self).__init__()
        self.convs = nn.ModuleList()
        self.ln = nn.LayerNorm(nhidden)
        self.subgraph_gcns_1 = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(U_Module_EGNN(in_size=nhidden, hidden_size=nhidden, out_size=nhidden, edge_feat_size=2))
            #self.subgraph_gcns_1.append(EGNN(dim=nhidden, edge_dim=1, m_dim=17))

        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda
        self.ks = args['ks']
        self.linear = nn.Linear(6717, 6205)

        self.pools = nn.ModuleList()
        self.unpools = nn.ModuleList()
        self.rgn_gcn2 = nn.ModuleList()

        self.l_n = len(args['ks'])
        self.drop_p = 0.0
        self.rgn_gcn2 = RGN_GCN(nlayers=nlayers, nfeat=nfeat, nhidden=128, nclass=1,
                                        dropout=dropout,lamda=lamda, alpha=alpha, variant=True, heads=1)

        for i in range(self.l_n):
            self.pools.append(Pool(args['ks'][i],nfeat, drop_p))
            self.unpools.append(Unpool(args['l_dim'], args['l_dim'], drop_p))


    def readout(self, hs):
        h_max = torch.Tensor([torch.max(h, 0)[0] for h in hs])

        return h_max

    def forward(self, G, H, X, adj, Efeats): # G:graph_batch, h:node_features_batch, x:coors_batch, adj:adj_batch, efeats: efeat_batch
        hs = []
        #egnn
        for i in range(self.l_n):
                if i==0:
                    h=H
                    x=X
                    g=G
                    efeats=Efeats
                if i==1:
                    fea1 = torch.cat([return_h1, H], dim=1)
                    fea1 = self.linear(fea1)
                    g, h, x, efeats ,idx= self.pools[i](G,fea1,X,Efeats)
                if i==2:
                    g, h, x, efeats, idx = self.pools[i](G, fea1, X,Efeats)
                    fea = torch.cat([return_h2, h], dim=1)
                    fea = self.linear(fea)
                    g, h, x, efeats, idx = self.pools[i](g, fea, x, efeats)
                # h = F.dropout(h, self.dropout, training=self.training)
                _layers = []
                layer_inner = self.act_fn(self.fcs[0](h))
                _layers.append(layer_inner)
                for j, con in enumerate(self.convs):
                    h = F.dropout(layer_inner, self.dropout, training=self.training)
                    h = self.act_fn(
                        # self, input, coord_feat, h0, lamda, alpha, l, adj_matrix=None, graph=None, efeats=None
                        self.ln(con(input=h, coord_feat=x, h0=_layers[0], lamda=self.lamda, alpha=self.alpha,
                                    l=j + 1, adj_matrix=None, graph=g, efeats=efeats)))

                if i == 0:
                    return_h1 = h
                if i == 1:
                    return_h2 = h
                #unpooling
                if i!= 0:
                    g, h = self.unpools[i](G, h, idx)

                hs.append(h)


        for h_i in hs:
            h = torch.max(h,h_i)

        return h


class BaseModule_GCN(nn.Module):
    # GCNII (orignally has residue connection and identity mapping)

    def __init__(self, in_features, out_features, residual=True, variant=False):
        super(BaseModule_GCN, self).__init__()
        self.variant = variant
        if self.variant:
            self.in_features = 2 * in_features
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, h0, lamda, alpha, l):

        theta = math.log(lamda / l + 1)
        hi = torch.spmm(adj, input)
        if self.variant:
            support = torch.cat([hi, h0], 1)
            r = (1 - alpha) * hi + alpha * h0
        else:
            support = (1 - alpha) * hi + alpha * h0
            r = support
        output = theta * torch.mm(support, self.weight) + (1 - theta) * r
        if self.residual:
            output = output + input
        return output


class RGN_GCN(nn.Module):
    # construct multilayer GCNII

    def __init__(self, nlayers, nfeat, nhidden, nclass, dropout, lamda, alpha, variant, heads):
        super(RGN_GCN, self).__init__()
        self.convs = nn.ModuleList()
        self.ln = nn.LayerNorm(nhidden)

        for _ in range(nlayers):
            self.convs.append(
                BaseModule_GCN(in_features=nhidden, out_features=nhidden, residual=True, variant=False)
            )

        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

    def forward(self, h, adj):
        _layers = []
        h = F.dropout(h, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](h))
        _layers.append(layer_inner)

        for i, con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(
                self.ln(con(input=layer_inner, h0=_layers[0], lamda=self.lamda, alpha=self.alpha, l=i + 1, adj=adj)))

        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)

        return layer_inner


# Gated self-attention mechanism
class Attention_1(nn.Module):
    # Multi-headed Self-Attention Mechanism

    def __init__(self, hidden_size, num_attention_heads):
        super(Attention_1, self).__init__()

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)

        self.sigmoid = torch.nn.Sigmoid()
        self.hidden_size = hidden_size

        # self.layer_norm = torch.nn.LayerNorm(hidden_size).to(device)

        self.query = nn.Linear(hidden_size, self.hidden_size, bias=False).to(device)
        self.key = nn.Linear(hidden_size, self.hidden_size, bias=False).to(device)
        self.value = nn.Linear(hidden_size, self.hidden_size, bias=False).to(device)
        self.gate = nn.Linear(hidden_size, self.hidden_size).to(device)

    def transpose_for_scores(self, x):  # Divide the vector into two heads
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads, self.attention_head_size)  # [:-1]Left closed and right open not included-1
        x = x.view(*new_x_shape)  # * The purpose of the asterisk is probably to remove the tuple attribute (automatic unpacking)
        return x.permute(0, 2, 1, 3)

    def forward(self, batch_hidden):
        # batch_hidden: b x len x hidden_size (2 * hidden_size of lstm)
        # batch_masks:  b x len

        # linear
        # key = torch.matmul(batch_hidden, self.weight) # b x len x hidden
        # batch_hidden=self.layer_norm(batch_hidden)
        query = self.query(batch_hidden)
        key = self.key(batch_hidden)
        value = self.key(batch_hidden)
        gate = self.sigmoid(self.gate(batch_hidden))
        #         key=batch_hidden
        #         query=batch_hidden
        #         print(key.shape)
        #         print(query.shape)
        # compute attention
        query = self.transpose_for_scores(query)  # batch,num_attention_heads,len,attention_head_size

        key = self.transpose_for_scores(key)
        value = self.transpose_for_scores(value)

        outputs = torch.matmul(key, query.transpose(-1, -2))  # b x num_attention_heads*len*len

        attention_scores = outputs / math.sqrt(self.attention_head_size)  # (batch,num_attention_heads,len,len)
        attn_scores = F.softmax(attention_scores, dim=-1)  #

        # For an all-zero vector, -1e32 results in 1/len, -inf is nan, and the extra complement is 0
        #         masked_attn_scores = attn_scores.masked_fill((1 - batch_masks).bool(), 0.0)

        # sum weighted sources
        context_layer = torch.matmul(attn_scores, value)  # (batch,num_attention_heads,len,attention_head_size

        context_layer = context_layer.permute(0, 2, 1,
                                              3).contiguous()  # (batch,n,num_attention_heads,attention_head_size)
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size, 1)
        batch_outputs = context_layer.view(*new_context_layer_shape)  # (batch,n,all_head_size)
        # print(gate.shape)#32,33,128
        # print(batch_outputs.shape)#32,33,128,1
        batch_outputs = gate * batch_outputs.squeeze(3)

        batch_outputs = batch_outputs

        # batch_outputs = torch.sum(batch_outputs, dim=1)        
        # batch_outputs = batch_outputs[:,0]+batch_outputs[:,-1]      

        return batch_outputs, attn_scores




def top_k_dgl_graph(scores, g, h, k, coords, efeats):
    # 获取图的节点数量
    num_nodes = g.number_of_nodes()
    # 获取得分最高的前 k% 的节点
    values, idx = torch.topk(scores, max(2, int(k * num_nodes)))
    # 更新节点特征
    new_h = h[idx, :]
    values = torch.unsqueeze(values, -1)
    new_h = new_h * values

    # 选择保留的节点对应的坐标
    new_coords = coords[idx, :]

    # 将 idx 转换到设备上
    idx = idx.to(g.device)

    # 使用 DGL 的 subgraph 方法来获取新的子图
    subgraph = dgl.node_subgraph(g, idx)

    # 选择保留的边特征
    edge_mask = subgraph.edata[dgl.EID]
    new_edge_feats = efeats[edge_mask]

    # 返回子图、更新后的特征、新的节点坐标、保留的边特征和节点索引
    return subgraph, new_h, new_coords, new_edge_feats, idx

class Pool(nn.Module):

    def __init__(self, k, in_dim, p):
        super(Pool, self).__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

    def forward(self, g, h, x, efeats):
        Z = self.drop(h)
        weights = self.proj(Z).squeeze()
        scores = self.sigmoid(weights)

        return top_k_dgl_graph(scores, g, h, self.k, x, efeats)


class Unpool(nn.Module):

    def __init__(self, *args):
        super(Unpool, self).__init__()

    def forward(self, g, h, idx):
        # 获取图的节点数量
        num_nodes = g.number_of_nodes()

        # 创建一个全零矩阵，形状为[节点数量, 特征维度]
        new_h = h.new_zeros([num_nodes, h.shape[1]])
        new_h[idx] = h
        return g, new_h


from get_clean import GET, fully_connect_edges, knn_edges
#from get_clean_old import GET
import torch

class GETModelRunner(nn.Module):
    def __init__(self,
                 d_hidden=128,       # hidden size
                 d_radial=16,        # mapped size of key/value in attention
                 n_channel=1,        # number of channels for coordinates
                 d_edge=2,           # edge feature size
                 n_rbf=16,           # RBF kernel size
                 n_head=4,           # number of heads for multi-head attention
                 ):   # device to use
        super(GETModelRunner, self).__init__()

        self.device = torch.device(device)
        self.d_hidden = d_hidden
        self.d_radial = d_radial
        self.n_channel = n_channel
        self.d_edge = d_edge
        self.n_rbf = n_rbf
        self.n_head = n_head

        self.model = GET(d_hidden, d_radial, n_channel, n_rbf, d_edge=d_edge, n_head=n_head)
        self.model.to(self.device)

    def forward(self, G, H, X, Efeats,device):
        n_atoms = H.shape[0]
        block_ids = torch.arange(n_atoms, device=device)
        batch_ids = torch.zeros(n_atoms, dtype=torch.long, device=device)
        src, dst = G.edges()
        src_dst = torch.tensor(list(zip(src.tolist(), dst.tolist())))
        H_out, X_out = self.model(H, X, block_ids, batch_ids, src_dst,device, Efeats)

        return H_out


