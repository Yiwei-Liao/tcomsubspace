#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script contains the sheaf Hypergraph Neural Network models.
Refactored for readability while preserving original logic.
"""

# =============================================================================
# Section 1: Imports
# =============================================================================
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from helper import *
# 导入新的 sheaf 卷积层
from sheafRHNN_conv import sheafRHNNConv 
# from RHNN_conv import RHNNConv # 旧的可以注释掉或保留用于对比
import time
import torch
import torch.nn as nn
from torch.nn import Parameter
import numpy as np

# Graph & Scatter Imports
from torch_scatter import scatter, scatter_mean, scatter_add
from torch_geometric.utils import softmax
import torch_sparse

# Local Imports (Dependencies)
import utils
from layers import * # Note: 'MLP' and 'Linear' are expected to be imported from 'layers'

# sheaf Builders & Laplacians
from sheaf_builder import (
    sheafBuilderDiag, 
    sheafBuilderOrtho, 
    sheafBuilderGeneral, 
    sheafBuilderLowRank,
    HGCNsheafBuilderDiag, 
    HGCNsheafBuilderGeneral, 
    HGCNsheafBuilderOrtho, 
    HGCNsheafBuilderLowRank
)
from hgcn_sheaf_laplacians import *


# =============================================================================
# Section 2: sheafHyperGNN Model
# =============================================================================

class sheafHyperGNN(nn.Module):
    """
    Hypergraph sheaf Model where the d x d blocks in H_BIG associated 
    to each pair (node, hyperedge) are modeled via specific sheaf builders.
    """
    def __init__(self, args, sheaf_type):
        super(sheafHyperGNN, self).__init__()

        # --- Configuration ---
        self.args = args
        self.num_layers = args.All_num_layers
        self.dropout = args.dropout
        self.num_features = args.num_features
        self.MLP_hidden = args.MLP_hidden 
        self.d = args.heads  # dimension of the stalks
        self.init_hedge = args.init_hedge # 'avg' or 'rand'
        self.norm_type = args.sheaf_normtype 
        self.act = args.sheaf_act 
        self.left_proj = args.sheaf_left_proj 
        self.norm = args.AllSet_input_norm
        self.dynamic_sheaf = args.dynamic_sheaf 
        self.residual = args.residual_HCHA
        self.n_sub = args.n_sub
        
        self.hyperedge_attr = None

        # Device setup
        if args.cuda in [0, 1]:
            self.device = torch.device('cuda:'+str(args.cuda) if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')

        # --- Input Projection ---
        self.lin = MLP(in_channels=self.num_features, 
                        hidden_channels=args.MLP_hidden,
                        out_channels=self.MLP_hidden*self.d,
                        num_layers=1,
                        dropout=0.0,
                        Normalization='ln',
                        InputNorm=False)
        
        # --- sheaf Logic Selection ---
        if sheaf_type == 'sheafHyperGNNDiag':
            Modelsheaf, ModelConv = sheafBuilderDiag, HyperDiffusionDiagsheafConv
        elif sheaf_type == 'sheafHyperGNNOrtho':
            Modelsheaf, ModelConv = sheafBuilderOrtho, HyperDiffusionOrthosheafConv
        elif sheaf_type == 'sheafHyperGNNGeneral':
            Modelsheaf, ModelConv = sheafBuilderGeneral, HyperDiffusionGeneralsheafConv
        elif sheaf_type == 'sheafHyperGNNLowRank':
            Modelsheaf, ModelConv = sheafBuilderLowRank, HyperDiffusionGeneralsheafConv
        
        # --- Layer Construction ---
        self.convs = nn.ModuleList()
        self.sheaf_builder = nn.ModuleList()

        # Initial sheaf Diffusion Layer
        self.convs.append(ModelConv(self.MLP_hidden, self.MLP_hidden, d=self.d, device=self.device, 
                                    norm_type=self.norm_type, left_proj=self.left_proj, 
                                    norm=self.norm, residual=self.residual))
        
        # Initial sheaf Builder
        self.sheaf_builder.append(Modelsheaf(args))

        # Stacking Layers
        for _ in range(self.num_layers - 1):
            self.convs.append(ModelConv(self.MLP_hidden, self.MLP_hidden, d=self.d, device=self.device, 
                                        norm_type=self.norm_type, left_proj=self.left_proj, 
                                        norm=self.norm, residual=self.residual))
            if self.dynamic_sheaf:
                self.sheaf_builder.append(Modelsheaf(args))
                
        # Output Linear Layer
        self.lin2 = Linear(self.MLP_hidden * self.d, args.num_classes, bias=False)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for sheaf_builder in self.sheaf_builder:
            sheaf_builder.reset_parameters()
        self.lin.reset_parameters()
        self.lin2.reset_parameters()    

    def init_hyperedge_attr(self, type, num_edges=None, x=None, hyperedge_index=None):
        """Initialize hyperedge attributes either randomly or as the average of the nodes."""
        if type == 'rand':
            hyperedge_attr = torch.randn((num_edges, self.num_features)).to(self.device)
        elif type == 'avg':
            hyperedge_attr = scatter_mean(x[hyperedge_index[0]], hyperedge_index[1], dim=0)
        else:
            hyperedge_attr = None
        return hyperedge_attr

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        num_nodes = data.x.shape[0]
        num_edges = data.edge_index[1].max().item() + 1

        # 1. Initialize hyperedge attributes if needed
        if self.hyperedge_attr is None:
            self.hyperedge_attr = self.init_hyperedge_attr(self.init_hedge, num_edges=num_edges, x=x, hyperedge_index=edge_index)

        # 2. Expand Input: N x num_features -> (N*d) x num_features
        x = self.lin(x)
        x = x.view((x.shape[0]*self.d, self.MLP_hidden)) 

        hyperedge_attr = self.lin(self.hyperedge_attr)
        hyperedge_attr = hyperedge_attr.view((hyperedge_attr.shape[0]*self.d, self.MLP_hidden))

        # 3. Message Passing Layers
        for i, conv in enumerate(self.convs[:-1]):
            # Infer sheaf
            if i == 0:
                h_sheaf_index, h_sheaf_attributes = self.sheaf_builder[i](x, hyperedge_attr, edge_index)
            elif (self.dynamic_sheaf and i <= self.n_sub):
                h_sheaf_index, h_sheaf_attributes = self.sheaf_builder[i % self.n_sub](x, hyperedge_attr, edge_index)
            
            # Diffusion
            x = F.elu(conv(x, hyperedge_index=h_sheaf_index, alpha=h_sheaf_attributes, num_nodes=num_nodes, num_edges=num_edges))
            x = F.dropout(x, p=self.dropout, training=self.training)

        # 4. Final Layer
        if len(self.convs) == 1 or self.dynamic_sheaf:
            h_sheaf_index, h_sheaf_attributes = self.sheaf_builder[self.n_sub-1](x, hyperedge_attr, edge_index) 
        
        x = self.convs[self.n_sub-1](x, hyperedge_index=h_sheaf_index, alpha=h_sheaf_attributes, num_nodes=num_nodes, num_edges=num_edges)
        
        # 5. Output Projection
        x = x.view(num_nodes, -1) # Nd x out_channels -> N x (d*out_channels)
        x = self.lin2(x)          # -> N x num_classes
        
        return x


# =============================================================================
# Section 3: sheafHyperGCN Model
# =============================================================================

class sheafHyperGCN(nn.Module):
    """
    HyperGCN variant using sheaf Diffusion.
    Constructs a Laplacian based on the sheaf structure.
    """
    def __init__(self, V, num_features, num_layers, num_classses, args, sheaf_type):
        super(sheafHyperGCN, self).__init__()
        
        # --- Configuration ---
        self.args = args
        self.num_nodes = V
        self.num_features = num_features  # <--- 【修复】这里补上了缺失的属性定义
        
        d, l, c = num_features, num_layers, num_classses
        cuda = args.cuda 

        # Hidden dimension calculation
        h = [args.MLP_hidden]
        for i in range(l-1):
            power = l - i + 2
            if (getattr(args, 'dname', None) is not None) and args.dname == 'citeseer':
                power = l - i + 4
            h.append(2**power)
        h.append(c)

        reapproximate = False 

        self.MLP_hidden = args.MLP_hidden
        self.d = args.heads
        self.num_layers = args.All_num_layers
        self.dropout = args.dropout 
        self.init_hedge = args.init_hedge 
        self.norm_type = args.sheaf_normtype 
        self.act = args.sheaf_act 
        self.left_proj = args.sheaf_left_proj 
        self.norm = args.AllSet_input_norm
        self.dynamic_sheaf = args.dynamic_sheaf 
        self.sheaf_type = sheaf_type 
        self.hyperedge_attr = None
        self.residual = args.residual_HCHA
        self.do, self.l = args.dropout, num_layers
        self.m = args.HyperGCN_mediators

        # --- sheaf Builder Selection ---
        if sheaf_type == 'Diagsheafs':
            Modelsheaf, self.Laplacian = HGCNsheafBuilderDiag, sheafLaplacianDiag
        elif sheaf_type == 'Orthosheafs':
            Modelsheaf, self.Laplacian = HGCNsheafBuilderOrtho, sheafLaplacianOrtho
        elif sheaf_type == 'Generalsheafs':
            Modelsheaf, self.Laplacian = HGCNsheafBuilderGeneral, sheafLaplacianGeneral
        elif sheaf_type == 'LowRanksheafs':
            Modelsheaf, self.Laplacian = HGCNsheafBuilderLowRank, sheafLaplacianGeneral

        # --- Layer Construction ---
        
        # Left Projection (Optional)
        if self.left_proj:
            self.lin_left_proj = nn.ModuleList([
                MLP(in_channels=self.d, 
                    hidden_channels=self.d,
                    out_channels=self.d,
                    num_layers=1,
                    dropout=0.0,
                    Normalization='ln',
                    InputNorm=self.norm) for i in range(l)])

        # Input Linear
        self.lin = MLP(in_channels=self.num_features, 
                       hidden_channels=self.MLP_hidden,
                       out_channels=self.MLP_hidden*self.d,
                       num_layers=1,
                       dropout=0.0,
                       Normalization='ln',
                       InputNorm=False)

        # sheaf Builders
        self.sheaf_builder = nn.ModuleList()
        self.sheaf_builder.append(Modelsheaf(args, args.MLP_hidden))

        if self.dynamic_sheaf:
            for i in range(1, l):
                self.sheaf_builder.append(Modelsheaf(args, h[i]))

        # HyperGraph Convolution Layers (from utils)
        self.layers = nn.ModuleList([utils.HyperGraphConvolution(
            h[i], h[i+1], reapproximate, cuda) for i in range(l)])

        # Output Linear
        self.lin2 = Linear(h[-1]*self.d, args.num_classes, bias=False)

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        if self.left_proj:
            for lin_layer in self.lin_left_proj:
                lin_layer.reset_parameters()
        self.lin.reset_parameters()
        self.lin2.reset_parameters()
        for sheaf_builder in self.sheaf_builder:
            sheaf_builder.reset_parameters()

    def init_hyperedge_attr(self, type, num_edges=None, x=None, hyperedge_index=None):
        if type == 'rand':
            hyperedge_attr = torch.randn((num_edges, self.num_features)).to(self.device)
        elif type == 'avg':
            hyperedge_attr = scatter_mean(x[hyperedge_index[0]], hyperedge_index[1], dim=0)
        else:
            hyperedge_attr = None
        return hyperedge_attr

    def normalise(self, A, hyperedge_index, num_nodes, d):
        """
        Normalizes the Laplacian matrix based on the selected sheaf_normtype.
        Supports: degree_norm, sym_degree_norm, block_norm, sym_block_norm.
        """
        # 1. Degree Normalization
        if self.args.sheaf_normtype == 'degree_norm':
            D = scatter_add(hyperedge_index.new_ones(hyperedge_index.size(1)), hyperedge_index[0], dim=0, dim_size=num_nodes*d) 
            D = torch.pow(D, -1.0)
            D[D == float("inf")] = 0
            D = utils.sparse_diagonal(D, (D.shape[0], D.shape[0]))
            D = D.coalesce()

            # Compute D^-1 A
            A = torch_sparse.spspmm(D.indices(), D.values(), A.indices(), A.values(), D.shape[0], D.shape[1], A.shape[1])
            A = torch.sparse_coo_tensor(A[0], A[1], size=(num_nodes*d, num_nodes*d)).to(D.device)
        
        # 2. Symmetric Degree Normalization
        elif self.args.sheaf_normtype == 'sym_degree_norm':
            D = scatter_add(hyperedge_index.new_ones(hyperedge_index.size(1)), hyperedge_index[0], dim=0, dim_size=num_nodes*d) 
            D = torch.pow(D, -0.5)
            D[D == float("inf")] = 0
            D = utils.sparse_diagonal(D, (D.shape[0], D.shape[0]))
            D = D.coalesce()

            # Compute D^-0.5 A D^-0.5
            A = torch_sparse.spspmm(D.indices(), D.values(), A.indices(), A.values(), D.shape[0], D.shape[1], A.shape[1], coalesced=True)
            A = torch_sparse.spspmm(A[0], A[1], D.indices(), D.values(), D.shape[0], D.shape[1], D.shape[1], coalesced=True)
            A = torch.sparse_coo_tensor(A[0], A[1], size=(num_nodes*d, num_nodes*d)).to(D.device)
        
        # 3. Block Normalization
        elif self.args.sheaf_normtype == 'block_norm':
            D = A.to_dense().view((num_nodes, d, num_nodes, d))
            D = torch.permute(D, (0,2,1,3)) 
            D = torch.diagonal(D, dim1=0, dim2=1) # block diagonals
            D = torch.permute(D, (2,0,1))

            if self.sheaf_type in ["Generalsheafs", "LowRanksheafs"]:
                D = utils.batched_sym_matrix_pow(D, -1.0)
            else:
                D = torch.pow(D, -1.0)
                D[D == float("inf")] = 0
            D = torch.block_diag(*torch.unbind(D,0))
            D = D.to_sparse()

            # Compute D^-1 A
            A = torch.sparse.mm(D, A)
            if self.sheaf_type in ["Generalsheafs", "LowRanksheafs"]:
                A = A.to_dense().clamp(-1,1).to_sparse()
        
        # 4. Symmetric Block Normalization
        elif self.args.sheaf_normtype == 'sym_block_norm':
            D = A.to_dense().view((num_nodes, d, num_nodes, d))
            D = torch.permute(D, (0,2,1,3))
            D = torch.diagonal(D, dim1=0, dim2=1)
            D = torch.permute(D, (2,0,1))

            if self.sheaf_type in ["Generalsheafs", "LowRanksheafs"]:
                D = utils.batched_sym_matrix_pow(D, -0.5)
            else:
                D = torch.pow(D, -0.5)
                D[D == float("inf")] = 0
            D = torch.block_diag(*torch.unbind(D,0))
            D = D.to_sparse()

            # Compute D^-0.5 A D^-0.5
            A = torch.sparse.mm(D, A) 
            A = torch.sparse.mm(A, D) 
            if self.sheaf_type in ["Generalsheafs", "LowRanksheafs"]:
                A = A.to_dense().clamp(-1,1).to_sparse()
        return A

    def forward(self, data):
        """
        Forward pass for l-layer GCN
        """
        do, l, m = self.do, self.l, self.m
        H = data.x
        num_nodes = data.x.shape[0]
        num_edges = data.edge_index[1].max().item() + 1
        edge_index= data.edge_index

        # 1. Initialize Attributes
        if self.hyperedge_attr is None:
            self.hyperedge_attr = self.init_hyperedge_attr(self.init_hedge, num_edges=num_edges, x=H, hyperedge_index=edge_index)
        
        # 2. Input Projections
        H = self.lin(H)
        hyperedge_attr = self.lin(self.hyperedge_attr)

        H = H.view((H.shape[0]*self.d, self.MLP_hidden)) # (N * d) x MLP_hidden
        hyperedge_attr = hyperedge_attr.view((hyperedge_attr.shape[0]*self.d, self.MLP_hidden))

        # 3. Layer Loop
        for i, hidden in enumerate(self.layers):
            if i == 0 or self.dynamic_sheaf:
                # A. Compute the sheaf
                sheaf = self.sheaf_builder[i](H, hyperedge_attr, edge_index) # N x E x d x d
                
                # B. Build Laplacian
                h_sheaf_index, h_sheaf_attributes = self.Laplacian(H, m, self.d, edge_index, sheaf)
                
                A = torch.sparse.FloatTensor(h_sheaf_index, h_sheaf_attributes, (num_nodes*self.d, num_nodes*self.d))
                A = A.coalesce()
                
                # C. Normalize
                A = self.normalise(A, h_sheaf_index, num_nodes, self.d)
                
                # D. Compute I - A
                eye_diag = torch.ones((num_nodes*self.d))
                A = utils.sparse_diagonal(eye_diag, (num_nodes*self.d, num_nodes*self.d)).to(A.device) - A 

            # E. Left Projection (Optional)
            if self.left_proj:
                H = H.t().reshape(-1, self.d)
                H = self.lin_left_proj[i](H)
                H = H.reshape(-1,num_nodes * self.d).t()
            
            # F. Convolution and Activation
            H = F.relu(hidden(A, H, m))
            if i < l - 1:
                H = F.dropout(H, do, training=self.training)

        # 4. Final Projection
        H = H.view(self.num_nodes, -1) # Nd x out_channels -> Nx(d*out_channels)
        H = self.lin2(H)               # -> N x num_classes
        
        return H
def add_embedding_noise(embedding: torch.FloatTensor, 
                        noise_type: str = 'gaussian', 
                        snr_db: float = 20.0) -> torch.FloatTensor:
    """
    给 embedding 添加噪声，支持 Gaussian 或 Rayleigh，通过 SNR 控制噪声强度。
    """
    # 计算信号功率
    signal_power = embedding.pow(2).mean()

    # 根据 SNR（以 dB 为单位）计算噪声功率
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise_std = noise_power.sqrt()  # 对 Gaussian 是 std，对 Rayleigh 是 scale

    if noise_type == 'gaussian':
        noise = torch.randn_like(embedding) * noise_std
    elif noise_type == 'rayleigh':
        uniform_noise = torch.rand_like(embedding).clamp(min=1e-6)
        noise = noise_std * torch.sqrt(-2.0 * torch.log(uniform_noise))
    else:
        raise ValueError(f"Unsupported noise type: {noise_type}. Use 'gaussian' or 'rayleigh'.")

    return embedding + noise

class BaseModel(torch.nn.Module):
    """
    Base class for all other models, including the implementation of the loss function
    """
    def __init__(self, params):
        super(BaseModel, self).__init__()

        self.p      = params
        self.act    = torch.tanh
        self.bceloss    = torch.nn.BCELoss()
        self.margin = self.p.margin
        self.neg = self.p.neg

        if self.margin:
            self.loss_func = torch.nn.MarginRankingLoss(margin=self.margin)

    def loss(self, pred, true_label):
        if not self.margin:
            return self.bceloss(pred, true_label)
        else:
            if self.neg != -1:
                neg_list = []
                pos_list = []

                for i, data in enumerate(true_label):
                    pos = pred[i][data > 0.1]
                    neg = pred[i][data < 0.1]
                    pos_ = pos.repeat_interleave(self.neg)
                    neg_index = torch.randint_like(pos_, len(neg)).long()
                    neg_ = neg[neg_index]
                    neg_list.append(neg_)
                    pos_list.append(pos_)
                neg_list = torch.cat(neg_list, 0)
                pos_list = torch.cat(pos_list, 0)
                y = -torch.ones(len(pos_list)).to(pred.device) # 动态设备
                loss = self.loss_func(neg_list, pos_list, y)
                return loss

class RHKHBase(BaseModel):
    """
    The base class of our RHKH model, from which all other RHKH model implementations inherit
    """
    def __init__(self, edge_index, edge_type, num_rel, params=None):
        super(RHKHBase, self).__init__(params)

        self.edge_index, self.edge_order = edge_index
        # 这里的 swap 可能是为了适配 MessagePassing 的 source/target 约定
        self.edge_index = self.edge_index[[1, 0]]

        self.edge_type = edge_type
        # 确保 gcn_dim (Encoder 输出) 与 embed_dim (Decoder 输入) 的一致性
        self.p.gcn_dim = self.p.embed_dim if self.p.gcn_layer == 1 else self.p.gcn_dim
        
        self.init_embed = get_param((self.p.num_ent, self.p.init_dim))
        
        # 不要硬编码 cuda，留空在 forward 中处理或默认 cpu
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        if self.p.num_bases > 0:
            self.init_rel = get_param((self.p.num_bases, self.p.init_dim))
        else:
            if self.p.score_func == 'transe':
                self.init_rel = get_param((num_rel, self.p.init_dim))
            else:
                self.init_rel = get_param((num_rel, self.p.init_dim))

        if self.p.multi_r:
            multi_rel = get_param((num_rel * 6, self.p.init_dim))
            self.multi_rel = (multi_rel).view(num_rel, 6, self.p.init_dim)
        else:
            multi_rel = get_param((num_rel * 6, self.p.init_dim * 2))
            self.multi_rel = (multi_rel).view(num_rel, 6, self.p.init_dim * 2)

        # ================== 修改核心：使用 sheafRHNNConv ==================
        # 无论 num_bases 如何，我们现在都使用 sheaf 卷积作为编码器
        self.conv1 = sheafRHNNConv(self.p.init_dim, self.p.gcn_dim, num_rel, act=self.act, params=self.p)
        
        # 如果是两层，第二层也使用 sheaf 或根据需要保留 RHNNConv
        # 这里统一改为 sheaf 以保持架构一致性
        if self.p.gcn_layer == 2:
            self.conv2 = sheafRHNNConv(self.p.gcn_dim, self.p.embed_dim, num_rel, act=self.act, params=self.p)
        else:
            self.conv2 = None
        # ===================================================================

        self.register_parameter('bias', Parameter(torch.zeros(self.p.num_ent)))
        self.arity = 5
        self.shift_network = torch.nn.Linear(self.p.embed_dim + self.arity, self.p.embed_dim)

    def forward_base(self, sub, rel, drop1, drop2):
        # 自动将 embedding 移到与输入相同的设备
        self.init_embed = self.init_embed.to(self.device)
        if self.p.score_func != 'transe':
            self.init_rel = self.init_rel.to(self.device)
            r = self.init_rel
        else:
            self.init_rel = self.init_rel.to(self.device)
            r = torch.cat([self.init_rel, -self.init_rel], dim=0)

        # 确保 edge_index 等也在正确的 device
        self.edge_index = self.edge_index.to(self.device)
        self.edge_type = self.edge_type.to(self.device)
        self.edge_order = self.edge_order.to(self.device)

        # 调用 Conv1
        x, r, _ = self.conv1(self.init_embed, self.edge_index, self.edge_order, self.edge_type, rel_embed=r)
        
        x = drop1(x)
        if self.p.gcn_layer == 2:
            x, r, _ = self.conv2(x, self.edge_index, self.edge_order, self.edge_type, rel_embed=r)
            x = drop2(x)

        rel_emb = torch.index_select(r, 0, rel)
        return rel_emb, x

    def forward_base_multi(self, sub, rel, drop1, drop2):
        self.init_embed = self.init_embed.to(self.device)
        self.init_rel = self.init_rel.to(self.device)
        
        r = self.init_rel
        multi_r = self.multi_rel.to(self.device) # 动态移动
        
        self.edge_index = self.edge_index.to(self.device)
        self.edge_type = self.edge_type.to(self.device)
        self.edge_order = self.edge_order.to(self.device)

        x, r, multi_r = self.conv1(self.init_embed, self.edge_index, self.edge_order, self.edge_type, rel_embed=r, multi_rel_embed=multi_r)
        x = drop1(x)
        
        if self.p.gcn_layer == 2:
            x, r, multi_r = self.conv2(x, self.edge_index, self.edge_order, self.edge_type, rel_embed=r, multi_rel_embed=multi_r)
            x = drop2(x)

        rel_emb = torch.index_select(r, 0, rel)
        multi_r_emb = torch.index_select(multi_r, 0, rel)
        return rel_emb, x, multi_r_emb

    def shift_onehot(self, entity_embed):
        e_onehot = torch.eye(5).unsqueeze(0).repeat(entity_embed.size()[0], 1, 1).to(entity_embed.device)
        e = self.shift_network(torch.cat((entity_embed, e_onehot), 2))
        return e

    def shift(self, v, dim, sh):
        y = torch.cat((v[:, dim:dim + 1, sh:], v[:, dim:dim + 1, :sh]), 2)
        return y

    def shift_rotate(self, entity_embed):
        emb_dim = entity_embed.size()[2]
        e1 = self.shift(entity_embed, 0, int(1 * emb_dim / 6))
        e2 = self.shift(entity_embed, 1, int(2 * emb_dim / 6))
        e3 = self.shift(entity_embed, 2, int(3 * emb_dim / 6))
        e4 = self.shift(entity_embed, 3, int(4 * emb_dim / 6))
        e5 = self.shift(entity_embed, 4, int(5 * emb_dim / 6))
        return torch.cat((e1, e2, e3, e4, e5), 1)


class RHKH_ConvE(RHKHBase):
    """
    Implementation of RHKH model with N-ConvE as score function
    """
    def __init__(self, edge_index, edge_type, params=None):
        super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params)

        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(self.p.num_filt)
        self.bn2 = torch.nn.BatchNorm1d(self.p.embed_dim)

        self.hidden_drop = torch.nn.Dropout(self.p.id_drop)
        self.hidden_drop2 = torch.nn.Dropout(self.p.id_drop2)
        self.feature_drop = torch.nn.Dropout(self.p.feat_drop)
        self.m_conv1 = torch.nn.Conv2d(1, out_channels=self.p.num_filt, kernel_size=(self.p.ker_sz, self.p.ker_sz), stride=1, padding=0, bias=self.p.bias)

        flat_sz_h = int(2 * self.p.k_w) - self.p.ker_sz + 1
        flat_sz_w = self.p.k_h - self.p.ker_sz + 1
        self.flat_sz = flat_sz_h * flat_sz_w * self.p.num_filt
        self.fc = torch.nn.Linear(self.flat_sz, self.p.embed_dim)

    def concat(self, e1_embed, rel_embed):
        e1_embed = e1_embed.view(-1, 1, self.p.embed_dim)
        rel_embed = rel_embed.view(-1, 1, self.p.embed_dim)
        stack_inp = torch.cat([e1_embed, rel_embed], 1)
        stack_inp = torch.transpose(stack_inp, 2, 1).reshape((-1, 1, 2 * self.p.k_w, self.p.k_h))
        return stack_inp

    def forward(self, sub, rel):
        # 1. 动态确定 device
        self.device = sub.device if isinstance(sub, torch.Tensor) else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 确保父类里的 device 也更新
        super().to(self.device)
        self.init_embed = self.init_embed.to(self.device)
        if hasattr(self, 'init_rel'): self.init_rel = self.init_rel.to(self.device)

        if self.p.position:
            sub_list = []
            pos_list = []
            for s in sub:
                sub0 = self.p.id2sub_pos[s]
                sub1, pos1 = sub0[0], sub0[1]
                sub_list.append(sub1)
                pos_list.append(list(pos1))

            sub = torch.LongTensor(sub_list).to(self.device)
            pos_sub_obj = torch.from_numpy(np.array(pos_list)).to(self.device)
            pos, obj_pos = pos_sub_obj[:, :-1], pos_sub_obj[:, -1:]

        # 2. 调用 sheaf 编码器 (通过 RHKHBase)
        if self.p.multi_r:
            rel_emb, all_ent, multi_rel_embed = self.forward_base_multi(sub, rel, self.hidden_drop, self.feature_drop)
        else:
            # 修改：去掉硬编码的 .cuda()，multi_rel 已经在 forward_base 里处理了
            if hasattr(self, 'multi_rel'):
                self.multi_rel = self.multi_rel.to(self.device)
            rel_emb, all_ent = self.forward_base(sub, rel, self.hidden_drop, self.feature_drop)

        # 3. 构造 Entity Embeddings
        entity_embed_add1 = torch.cat((all_ent, torch.zeros(1, all_ent.size()[1]).to(self.device)), 0)
        
        # 确保 id2entity_instance 在正确设备
        id2entity_instance = self.p.id2entity_instance.to(self.device)

        if self.p.position:
            if not self.p.multi_r:
                multi_rel_embed = torch.index_select(self.multi_rel, 0, rel)
            multi_rel_embed_add1 = torch.cat((multi_rel_embed, torch.zeros(multi_rel_embed.size()[0], 1, multi_rel_embed.size()[2]).to(self.device)), 1)
            multi_rel_embed = torch.gather(multi_rel_embed_add1, 1, pos.unsqueeze(2).repeat(1, 1, self.p.embed_dim)).squeeze(1) # 注意这里如果是200请改为 embed_dim
            entity_embed_all = entity_embed_add1[id2entity_instance[sub]] * multi_rel_embed
        else:
            entity_embed_all = entity_embed_add1[id2entity_instance[sub]]

        # 4. 噪声与 Shift
        if self.p.add_noise:
            entity_embed_all = add_embedding_noise(entity_embed_all, noise_type=self.p.noise_type, snr_db=self.p.snr)
        
        if self.p.shift == 1:
            entity_embed_all = self.shift_onehot(entity_embed_all)
        elif self.p.shift == 2:
            entity_embed_all = self.shift_rotate(entity_embed_all)

        sub_emb = torch.sum(entity_embed_all, 1)

        # 5. ConvE 评分
        stk_inp = self.concat(sub_emb, rel_emb)
        x = self.bn0(stk_inp)
        x = self.m_conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_drop(x)
        x = x.view(-1, self.flat_sz)
        x = self.fc(x)
        x = self.hidden_drop2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = torch.mm(x, all_ent.transpose(1, 0))
        x += self.bias.expand_as(x)

        score = torch.sigmoid(x)
        return score