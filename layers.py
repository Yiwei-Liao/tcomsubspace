#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Neural Network Layers Module for Hypergraph Neural Networks.
Contains implementations for:
1. sheaf Diffusion Layers (Diagonal, Orthogonal, General)
2. Standard Hypergraph Convolution Layers (HGNN, HNHN, HCHA)
3. Multi-Layer Perceptrons (MLP)
4. AllSet Transformer Layers (PMA, HalfNLHconv)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn import Linear, Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
from torch_scatter import scatter_add, scatter
from torch_geometric.nn.inits import glorot as pyg_glorot, zeros as pyg_zeros
from torch_geometric.typing import Adj, Size, OptTensor
from typing import Optional

# Local Imports
import utils
import torch_sparse


# =============================================================================
# Section 1: Utility Functions & Initialization
# =============================================================================

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)

def normalisation_matrices(x, hyperedge_index, alpha, num_nodes, num_edges, d, norm_type='degree_norm'):
    """
    Computes normalization matrices D and B for sheaf Diffusion.
    Returns inverse (D^-1, B^-1) or square root inverse (D^-0.5, B^-1) based on norm_type.
    """
    # 1. Degree Normalization
    if norm_type == 'degree_norm':
        D = scatter_add(x.new_ones(hyperedge_index.size(1)), hyperedge_index[0], dim=0, dim_size=num_nodes*d) 
        D = 1.0 / D
        D[D == float("inf")] = 0

        B = scatter_add(x.new_ones(hyperedge_index.size(1)), hyperedge_index[1], dim=0, dim_size=num_edges*d)
        B = 1.0 / B
        B[B == float("inf")] = 0
        return D, B

    # 2. Symmetric Degree Normalization
    elif norm_type == 'sym_degree_norm':
        D = scatter_add(x.new_ones(hyperedge_index.size(1)), hyperedge_index[0], dim=0, dim_size=num_nodes*d) 
        D = D ** (-0.5)
        D[D == float("inf")] = 0

        B = scatter_add(x.new_ones(hyperedge_index.size(1)), hyperedge_index[1], dim=0, dim_size=num_edges*d)
        B = 1.0 / B
        B[B == float("inf")] = 0
        return D, B

    # 3. Block Normalization (Diag(HHT))
    elif norm_type == 'block_norm':
        D = scatter_add(alpha*alpha, hyperedge_index[0], dim=0, dim_size=num_nodes*d)
        D = 1.0 / D 
        D[D == float("inf")] = 0

        B = scatter_add(x.new_ones(hyperedge_index.size(1)), hyperedge_index[1], dim=0, dim_size=num_edges*d)
        B = 1.0 / B
        B[B == float("inf")] = 0
        return D, B
    
    # 4. Symmetric Block Normalization
    elif norm_type == 'sym_block_norm':
        D = scatter_add(alpha*alpha, hyperedge_index[0], dim=0, dim_size=num_nodes*d)
        D = D ** (-0.5) 
        D[D == float("inf")] = 0

        B = scatter_add(x.new_ones(hyperedge_index.size(1)), hyperedge_index[1], dim=0, dim_size=num_edges*d)
        B = 1.0 / B
        B[B == float("inf")] = 0
        return D, B
    
    return None, None


# =============================================================================
# Section 2: General Components (MLP)
# =============================================================================

class MLP(nn.Module):
    """ 
    Multi-Layer Perceptron with optional Batch/Layer Normalization.
    Adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py 
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=.5, Normalization='bn', InputNorm=False):
        super(MLP, self).__init__()
        self.lins = nn.ModuleList()
        self.normalizations = nn.ModuleList()
        self.InputNorm = InputNorm
        self.dropout = dropout

        assert Normalization in ['bn', 'ln', 'None']
        
        # Helper to select normalization layer
        def get_norm_layer(channels):
            if Normalization == 'bn': return nn.BatchNorm1d(channels)
            if Normalization == 'ln': return nn.LayerNorm(channels)
            return nn.Identity()

        # Input Layer
        self.normalizations.append(get_norm_layer(in_channels) if InputNorm else nn.Identity())
        
        if num_layers == 1:
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            self.normalizations.append(get_norm_layer(hidden_channels))
            
            # Hidden Layers
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                self.normalizations.append(get_norm_layer(hidden_channels))
            
            # Output Layer
            self.lins.append(nn.Linear(hidden_channels, out_channels))

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for normalization in self.normalizations:
            if not isinstance(normalization, nn.Identity):
                normalization.reset_parameters()

    def forward(self, x):
        x = self.normalizations[0](x)
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.normalizations[i+1](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x


# =============================================================================
# Section 3: sheaf Diffusion Layers
# =============================================================================

class HyperDiffusionDiagsheafConv(MessagePassing):
    """
    One layer of sheaf Diffusion with diagonal Laplacian.
    Y = (I - D^-1/2 L D^-1/2) X  or similar variants.
    """
    def __init__(self, in_channels, out_channels, d, device, dropout=0, bias=True, norm_type='degree_norm', 
                left_proj=None, norm=None, residual=False, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(flow='source_to_target', node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.d = d
        self.norm_type = norm_type
        self.left_proj = left_proj
        self.norm = norm
        self.residual = residual
        self.device = device

        if self.left_proj:
            self.lin_left_proj = MLP(in_channels=d, hidden_channels=d, out_channels=d,
                                     num_layers=1, dropout=0.0, Normalization='ln', InputNorm=self.norm)

        self.lin = MLP(in_channels=in_channels, hidden_channels=out_channels, out_channels=out_channels,
                       num_layers=1, dropout=0.0, Normalization='ln', InputNorm=self.norm)
        
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.I_mask = None
        self.Id = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.left_proj:
            self.lin_left_proj.reset_parameters()
        self.lin.reset_parameters()
        zeros(self.bias)

    def forward(self, x: Tensor, hyperedge_index: Tensor, alpha, num_nodes, num_edges) -> Tensor:
        if self.left_proj:
            x = x.t().reshape(-1, self.d)
            x = self.lin_left_proj(x)
            x = x.reshape(-1,num_nodes * self.d).t()
        
        x = self.lin(x)
        data_x = x

        D_inv, B_inv = normalisation_matrices(x, hyperedge_index, alpha, num_nodes, num_edges, self.d, self.norm_type)

        if self.norm_type in ['sym_degree_norm', 'sym_block_norm']:
            x = D_inv.unsqueeze(-1) * x

        H = torch.sparse.FloatTensor(hyperedge_index, alpha, size=(num_nodes*self.d, num_edges*self.d))
        H_t = torch.sparse.FloatTensor(hyperedge_index.flip([0]), alpha, size=(num_edges*self.d, num_nodes*self.d))

        B_inv = utils.sparse_diagonal(B_inv, shape=(num_edges*self.d, num_edges*self.d))
        D_inv = utils.sparse_diagonal(D_inv, shape=(num_nodes*self.d, num_nodes*self.d))

        # Coalesce sparse tensors
        B_inv = B_inv.coalesce()
        H_t = H_t.coalesce()
        H = H.coalesce()
        D_inv = D_inv.coalesce()

        # Compute Laplacian L = D^-1 H B^-1 H^T
        minus_L = torch_sparse.spspmm(B_inv.indices(), B_inv.values(), H_t.indices(), H_t.values(), B_inv.shape[0], B_inv.shape[1], H_t.shape[1])
        minus_L = torch_sparse.spspmm(H.indices(), H.values(), minus_L[0], minus_L[1], H.shape[0], H.shape[1], H_t.shape[1])
        minus_L = torch_sparse.spspmm(D_inv.indices(), D_inv.values(), minus_L[0], minus_L[1], D_inv.shape[0], D_inv.shape[1], H_t.shape[1])
        minus_L = torch.sparse_coo_tensor(minus_L[0], minus_L[1], size=(num_nodes*self.d, num_nodes*self.d)).to(self.device)

        # Prepare Identity masks
        if self.I_mask is None:
            I_mask_indices = torch.stack([torch.arange(num_nodes), torch.arange(num_nodes)], dim=0)
            I_mask_indices = utils.generate_indices_general(I_mask_indices, self.d)
            I_mask_values = torch.ones((I_mask_indices.shape[1]))
            self.I_mask = torch.sparse_coo_tensor(I_mask_indices, I_mask_values).to(self.device)
            self.Id = utils.sparse_diagonal(torch.ones(num_nodes*self.d), shape=(num_nodes*self.d, num_nodes * self.d)).to(self.device)

        # Compute I - L (Diffusion Operator)
        minus_L = minus_L.coalesce()
        minus_L = torch.sparse_coo_tensor(minus_L.indices(), minus_L.values(), minus_L.size())
        minus_L = minus_L - 2 * minus_L.mul(self.I_mask)
        minus_L = self.Id + minus_L
        minus_L = minus_L.coalesce()

        out = torch_sparse.spmm(minus_L.indices(), minus_L.values(), minus_L.shape[0], minus_L.shape[1], x)
        
        if self.bias is not None:
            out = out + self.bias
        if self.residual:
            out = out + data_x
        return out


class HyperDiffusionOrthosheafConv(MessagePassing):
    """
    One layer of sheaf Diffusion with orthogonal Laplacian.
    """
    def __init__(self, in_channels, out_channels, d, device, dropout=0, bias=True, norm_type='degree_norm', 
                left_proj=None, norm=None, residual=False, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(flow='source_to_target', node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.d = d
        self.norm = norm
        self.left_proj = left_proj
        self.residual = residual
        self.device = device
        
        # Adjust norm_type mapping for ortho block
        if norm_type == 'block_norm':
            self.norm_type = 'degree_norm'
        elif norm_type == 'sym_block_norm':
            self.norm_type = 'sym_degree_norm'
        else:
            self.norm_type = norm_type

        if self.left_proj:
            self.lin_left_proj = MLP(in_channels=d, hidden_channels=d, out_channels=d,
                                     num_layers=1, dropout=0.0, Normalization='ln', InputNorm=self.norm)
        
        self.lin = MLP(in_channels=in_channels, hidden_channels=d, out_channels=out_channels,
                       num_layers=1, dropout=0.0, Normalization='ln', InputNorm=self.norm)
        
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.I_mask = None
        self.Id = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.left_proj:
            self.lin_left_proj.reset_parameters()
        self.lin.reset_parameters()
        zeros(self.bias)

    def forward(self, x: Tensor, hyperedge_index: Tensor, alpha, num_nodes, num_edges) -> Tensor:
        if self.left_proj:
            x = x.t().reshape(-1, self.d)
            x = self.lin_left_proj(x)
            x = x.reshape(-1,num_nodes * self.d).t()
        
        x = self.lin(x)    
        data_x = x

        if self.I_mask is None:
            I_mask_indices = torch.stack([torch.arange(num_nodes), torch.arange(num_nodes)], dim=0)
            I_mask_indices = utils.generate_indices_general(I_mask_indices, self.d)
            I_mask_values = -1 * torch.ones((I_mask_indices.shape[1]))
            self.I_mask = torch.sparse.FloatTensor(I_mask_indices, I_mask_values).to(self.device)
            self.Id = utils.sparse_diagonal(torch.ones(num_nodes*self.d), shape=(num_nodes*self.d, num_nodes * self.d)).to(self.device)
        
        D_inv, B_inv = normalisation_matrices(x, hyperedge_index, alpha, num_nodes, num_edges, self.d, norm_type=self.norm_type)

        if self.norm_type in ['sym_degree_norm', 'sym_block_norm']:
            x = D_inv.unsqueeze(-1) * x

        H = torch.sparse.FloatTensor(hyperedge_index, alpha, size=(num_nodes*self.d, num_edges*self.d))
        H_t = torch.sparse.FloatTensor(hyperedge_index.flip([0]), alpha, size=(num_edges*self.d, num_nodes*self.d))

        B_inv = utils.sparse_diagonal(B_inv, shape=(num_edges*self.d, num_edges*self.d))
        D_inv = utils.sparse_diagonal(D_inv, shape=(num_nodes*self.d, num_nodes*self.d))

        B_inv = B_inv.coalesce()
        H_t = H_t.coalesce()
        H = H.coalesce()
        D_inv = D_inv.coalesce()

        minus_L = torch_sparse.spspmm(B_inv.indices(), B_inv.values(), H_t.indices(), H_t.values(), B_inv.shape[0], B_inv.shape[1], H_t.shape[1])
        minus_L = torch_sparse.spspmm(H.indices(), H.values(), minus_L[0], minus_L[1], H.shape[0], H.shape[1], H_t.shape[1])
        minus_L = torch_sparse.spspmm(D_inv.indices(), D_inv.values(), minus_L[0], minus_L[1], D_inv.shape[0], D_inv.shape[1], H_t.shape[1])
        minus_L = torch.sparse_coo_tensor(minus_L[0], minus_L[1], size=(num_nodes*self.d, num_nodes*self.d)).to(self.device)

        minus_L = minus_L * self.I_mask
        minus_L = self.Id + minus_L
        minus_L = minus_L.coalesce()

        out = torch_sparse.spmm(minus_L.indices(), minus_L.values(), minus_L.shape[0], minus_L.shape[1], x)

        if self.bias is not None:
            out = out + self.bias
        if self.residual:
            out = out + data_x
        return out


class HyperDiffusionGeneralsheafConv(MessagePassing):
    """
    One layer of sheaf Diffusion with general/lowrank Laplacian.
    """
    def __init__(self, in_channels, out_channels, d, device, dropout=0, bias=True, norm_type='degree_norm', 
                left_proj=None, norm=None, residual=False, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(flow='source_to_target', node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.d = d
        self.norm = norm
        self.left_proj = left_proj
        self.residual = residual
        self.norm_type = norm_type
        self.device = device

        if self.left_proj:
            self.lin_left_proj = MLP(in_channels=d, hidden_channels=d, out_channels=d,
                                     num_layers=1, dropout=0.0, Normalization='ln', InputNorm=self.norm)

        self.lin = MLP(in_channels=in_channels, hidden_channels=d, out_channels=out_channels,
                       num_layers=1, dropout=0.0, Normalization='ln', InputNorm=self.norm)
        
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.I_mask = None
        self.Id = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.left_proj:
            self.lin_left_proj.reset_parameters()
        self.lin.reset_parameters()
        zeros(self.bias)

    def normalise(self, h_general_sheaf, hyperedge_index, norm_type, num_nodes, num_edges):
        """Helper for block normalization in general sheaf"""
        row_small = hyperedge_index[0].view(-1,self.d,self.d)[:,0,0] // self.d
        h_general_sheaf_1 = h_general_sheaf.reshape(row_small.shape[0], self.d, self.d)

        to_be_inv_nodes = torch.bmm(h_general_sheaf_1, h_general_sheaf_1.permute(0,2,1)) 
        to_be_inv_nodes = scatter_add(to_be_inv_nodes, row_small, dim=0, dim_size=num_nodes)

        if norm_type == 'block_norm':
            return utils.batched_sym_matrix_pow(to_be_inv_nodes, -1.0)
        elif norm_type == 'sym_block_norm':
            return utils.batched_sym_matrix_pow(to_be_inv_nodes, -0.5)

    def forward(self, x: Tensor, hyperedge_index: Tensor, alpha, num_nodes, num_edges) -> Tensor:
        if self.left_proj:
            x = x.t().reshape(-1, self.d)
            x = self.lin_left_proj(x)
            x = x.reshape(-1,num_nodes * self.d).t()

        x = self.lin(x)
        data_x = x

        if self.I_mask is None:
            I_mask_indices = torch.stack([torch.arange(num_nodes), torch.arange(num_nodes)], dim=0)
            I_mask_indices = utils.generate_indices_general(I_mask_indices, self.d)
            I_mask_values = -1 * torch.ones((I_mask_indices.shape[1]))
            self.I_mask = torch.sparse.FloatTensor(I_mask_indices, I_mask_values).to(self.device)
            self.Id = utils.sparse_diagonal(torch.ones(num_nodes*self.d), shape=(num_nodes*self.d, num_nodes * self.d)).to(self.device)

        # Compute Normalization
        if self.norm_type in ['block_norm', 'sym_block_norm']:
            B_inv_flat = scatter_add(x.new_ones(hyperedge_index.size(1)), hyperedge_index[1], dim=0, dim_size=num_edges*self.d)
            B_inv_flat = 1.0 / B_inv_flat
            B_inv_flat[B_inv_flat == float("inf")] = 0
            B_inv = utils.sparse_diagonal(B_inv_flat, shape=(num_edges*self.d, num_edges*self.d))

            D_inv = self.normalise(alpha, hyperedge_index, self.norm_type, num_nodes, num_edges) 
            diag_indices_D = torch.stack([torch.arange(num_nodes), torch.arange(num_nodes)], dim=0)
            D_inv_indices = utils.generate_indices_general(diag_indices_D, self.d).to(x.device)
            D_inv_flat = D_inv.reshape(-1)
            D_inv = torch.sparse.FloatTensor(D_inv_indices, D_inv_flat)
        else:
            D_inv, B_inv = normalisation_matrices(x, hyperedge_index, alpha, num_nodes, num_edges, self.d, norm_type=self.norm_type)
        
        # Apply Normalization
        if self.norm_type == 'sym_degree_norm':
            x = D_inv.unsqueeze(-1) * x
        elif self.norm_type == 'sym_block_norm':
            D_inv = D_inv.coalesce()
            x = torch_sparse.spmm(D_inv.indices(), D_inv.values(), D_inv.shape[0], D_inv.shape[1], x)

        if self.norm_type in ['sym_degree_norm', 'degree_norm']:
            B_inv = utils.sparse_diagonal(B_inv, shape=(num_edges*self.d, num_edges*self.d))
            D_inv = utils.sparse_diagonal(D_inv, shape=(num_nodes*self.d, num_nodes*self.d))

        H = torch.sparse.FloatTensor(hyperedge_index, alpha, size=(num_nodes*self.d, num_edges*self.d))
        H_t = torch.sparse.FloatTensor(hyperedge_index.flip([0]), alpha, size=(num_edges*self.d, num_nodes*self.d))

        B_inv = B_inv.coalesce()
        H_t = H_t.coalesce()
        H = H.coalesce()
        D_inv = D_inv.coalesce()

        minus_L = torch_sparse.spspmm(B_inv.indices(), B_inv.values(), H_t.indices(), H_t.values(), B_inv.shape[0], B_inv.shape[1], H_t.shape[1])
        minus_L = torch_sparse.spspmm(H.indices(), H.values(), minus_L[0], minus_L[1], H.shape[0], H.shape[1], H_t.shape[1])
        minus_L = torch_sparse.spspmm(D_inv.indices(), D_inv.values(), minus_L[0], minus_L[1], D_inv.shape[0], D_inv.shape[1], H_t.shape[1])
        minus_L = torch.sparse_coo_tensor(minus_L[0], minus_L[1], size=(num_nodes*self.d, num_nodes*self.d)).to(self.device)
        
        minus_L = minus_L * self.I_mask
        minus_L = self.Id + minus_L
        minus_L = minus_L.coalesce()

        out = torch_sparse.spmm(minus_L.indices(), minus_L.values(), minus_L.shape[0], minus_L.shape[1], x)
        
        if self.bias is not None:
            out = out + self.bias
        if self.residual:
            out = out + data_x

        return out


# =============================================================================
# Section 4: AllSet / PMA Layers
# =============================================================================

class PMA(MessagePassing):
    """
    PMA (Perceiver-like Multi-head Attention) Layer.
    Modified from GATConv logic.
    """
    _alpha: OptTensor

    def __init__(self, in_channels, hid_dim, out_channels, num_layers, heads=1, concat=True,
                 negative_slope=0.2, dropout=0.0, bias=False, **kwargs):
        super(PMA, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.hidden = hid_dim // heads
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = 0.
        self.aggr = 'add'

        self.lin_K = Linear(in_channels, self.heads*self.hidden)
        self.lin_V = Linear(in_channels, self.heads*self.hidden)
        self.att_r = Parameter(torch.Tensor(1, heads, self.hidden)) 
        
        self.rFF = MLP(in_channels=self.heads*self.hidden,
                       hidden_channels=self.heads*self.hidden,
                       out_channels=out_channels,
                       num_layers=num_layers,
                       dropout=.0, Normalization='None')
        
        self.ln0 = nn.LayerNorm(self.heads*self.hidden)
        self.ln1 = nn.LayerNorm(self.heads*self.hidden)
        
        self.register_parameter('bias', None)
        self._alpha = None
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_K.weight)
        glorot(self.lin_V.weight)
        self.rFF.reset_parameters()
        self.ln0.reset_parameters()
        self.ln1.reset_parameters()
        nn.init.xavier_uniform_(self.att_r)

    def forward(self, x, edge_index: Adj, size: Size = None, return_attention_weights=None):
        H, C = self.heads, self.hidden

        x_K = self.lin_K(x).view(-1, H, C)
        x_V = self.lin_V(x).view(-1, H, C)
        alpha_r = (x_K * self.att_r).sum(dim=-1)

        out = self.propagate(edge_index, x=x_V, alpha=alpha_r, aggr=self.aggr)

        alpha = self._alpha
        self._alpha = None

        out += self.att_r 
        out = self.ln0(out.view(-1, self.heads * self.hidden))
        out = self.ln1(out + F.relu(self.rFF(out)))

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            return out, (edge_index, alpha)
        else:
            return out

    def message(self, x_j, alpha_j, index, ptr, size_j):
        alpha = F.leaky_relu(alpha_j, self.negative_slope)
        alpha = softmax(alpha, index, ptr, index.max()+1)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def aggregate(self, inputs, index, dim_size=None, aggr=None):
        if aggr is None: raise ValueError("aggr was not passed!")
        return scatter(inputs, index, dim=self.node_dim, reduce=aggr)


class HalfNLHconv(MessagePassing):
    """
    AllSet Transformer Block.
    Composition: MLP -> Prop (PMA) -> MLP
    """
    def __init__(self, in_dim, hid_dim, out_dim, num_layers, dropout,
                 Normalization='bn', InputNorm=False, heads=1, attention=True):
        super(HalfNLHconv, self).__init__()

        self.attention = attention
        self.dropout = dropout

        if self.attention:
            self.prop = PMA(in_dim, hid_dim, out_dim, num_layers, heads=heads)
        else:
            if num_layers > 0:
                self.f_enc = MLP(in_dim, hid_dim, hid_dim, num_layers, dropout, Normalization, InputNorm)
                self.f_dec = MLP(hid_dim, hid_dim, out_dim, num_layers, dropout, Normalization, InputNorm)
            else:
                self.f_enc = nn.Identity()
                self.f_dec = nn.Identity()

    def reset_parameters(self):
        if self.attention:
            self.prop.reset_parameters()
        else:
            if not isinstance(self.f_enc, nn.Identity): self.f_enc.reset_parameters()
            if not isinstance(self.f_dec, nn.Identity): self.f_dec.reset_parameters()

    def forward(self, x, edge_index, norm, aggr='add'):
        if self.attention:
            x = self.prop(x, edge_index)
        else:
            x = F.relu(self.f_enc(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.propagate(edge_index, x=x, norm=norm, aggr=aggr)
            x = F.relu(self.f_dec(x))
        return x

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def aggregate(self, inputs, index, dim_size=None, aggr=None):
        if aggr is None: raise ValueError("aggr was not passed!")
        return scatter(inputs, index, dim=self.node_dim, reduce=aggr)


# =============================================================================
# Section 5: Standard Hypergraph Layers
# =============================================================================

class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()
        self.lin = Linear(in_ft, out_ft, bias=bias)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, x, G):
        x = self.lin(x)
        x = torch.matmul(G, x)
        return x


class HNHNConv(MessagePassing):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1, nonlinear_inbetween=True,
                 concat=True, bias=True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(HNHNConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.nonlinear_inbetween = nonlinear_inbetween
        self.heads = heads
        self.concat = True

        self.weight_v2e = Linear(in_channels, hidden_channels)
        self.weight_e2v = Linear(hidden_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.weight_v2e.reset_parameters()
        self.weight_e2v.reset_parameters()

    def forward(self, x, data):
        hyperedge_index = data.edge_index
        num_nodes = x.size(0)
        num_edges = int(hyperedge_index[1].max()) + 1 if hyperedge_index.numel() > 0 else 0

        x = self.weight_v2e(x)
        x = data.D_v_beta.unsqueeze(-1) * x

        self.flow = 'source_to_target'
        out = self.propagate(hyperedge_index, x=x, norm=data.D_e_beta_inv, size=(num_nodes, num_edges))
        
        if self.nonlinear_inbetween:
            out = F.relu(out)
        
        out = torch.squeeze(out, dim=1)
        out = self.weight_e2v(out)
        out = data.D_e_alpha.unsqueeze(-1) * out

        self.flow = 'target_to_source'
        out = self.propagate(hyperedge_index, x=out, norm=data.D_v_alpha_inv, size=(num_edges, num_nodes))
        
        return out

    def message(self, x_j, norm_i):
        return norm_i.view(-1, 1) * x_j


class HypergraphConv(MessagePassing):
    r"""The hypergraph convolutional operator from the `"Hypergraph Convolution and Hypergraph Attention" paper"""
    def __init__(self, in_channels, out_channels, use_attention=False, heads=1,
                 concat=True, negative_slope=0.2, dropout=0, bias=True, residual=False, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(flow='source_to_target', node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention
        self.residual = residual
        self.negative_slope = negative_slope
        self.dropout = dropout

        if self.use_attention:
            self.heads = heads
            self.concat = concat
            self.lin = Linear(in_channels, heads * out_channels, bias=False)
            self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))
        else:
            self.heads = 1
            self.concat = True
            self.lin = Linear(in_channels, out_channels, bias=False)

        if bias and concat:
            self.bias = Parameter(torch.Tensor(self.heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        if self.use_attention:
            glorot(self.att)
        zeros(self.bias)

    def forward(self, x: Tensor, hyperedge_index: Tensor,
                hyperedge_weight: Optional[Tensor] = None,
                hyperedge_attr: Optional[Tensor] = None) -> Tensor:
        
        num_nodes = hyperedge_index[0].max().item() + 1
        num_edges = int(hyperedge_index[1].max()) + 1 if hyperedge_index.numel() > 0 else 0

        if hyperedge_weight is None:
            hyperedge_weight = x.new_ones(num_edges)

        x = self.lin(x)
        data_x = x
        alpha = None

        if self.use_attention:
            assert hyperedge_attr is not None
            x = x.view(-1, self.heads, self.out_channels)
            hyperedge_attr = self.lin(hyperedge_attr)
            hyperedge_attr = hyperedge_attr.view(-1, self.heads, self.out_channels)
            
            x_i = x[hyperedge_index[0]]
            x_j = hyperedge_attr[hyperedge_index[1]]
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
            alpha = F.leaky_relu(alpha, self.negative_slope)
            alpha = softmax(alpha, hyperedge_index[0], num_nodes=x.size(0))
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        D = scatter_add(hyperedge_weight[hyperedge_index[1]], hyperedge_index[0], dim=0, dim_size=num_nodes)
        D = 1.0 / D
        D[D == float("inf")] = 0

        B = scatter_add(x.new_ones(hyperedge_index.size(1)), hyperedge_index[1], dim=0, dim_size=num_edges)
        B = 1.0 / B
        B[B == float("inf")] = 0

        out = self.propagate(hyperedge_index, x=x, norm=B, alpha=alpha, size=(num_nodes, num_edges))
        out = self.propagate(hyperedge_index.flip([0]), x=out, norm=D, alpha=alpha, size=(num_edges, num_nodes))

        if self.concat is True:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        if self.residual:
            out = out + data_x
        return out

    def message(self, x_j: Tensor, norm_i: Tensor, alpha: Tensor) -> Tensor:
        H, F = self.heads, self.out_channels
        out = norm_i.view(-1, 1, 1) * x_j.view(-1, H, F)
        if alpha is not None:
            out = alpha.view(-1, self.heads, 1) * out
        return out