#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
sheaf Builder Module.
Contains classes and helper functions to construct restriction maps (Sheaves) 
for different constraint types (Diagonal, General, Orthogonal, LowRank).
"""

# =============================================================================
# Section 1: Imports
# =============================================================================
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter, scatter_mean, scatter_add

# Local Imports
import utils
from layers import MLP
from orthogonal import Orthogonal


# =============================================================================
# Section 2: Prediction Helper Functions
# =============================================================================

def apply_activation(x, act_type):
    """Helper to apply activation function."""
    if act_type == 'sigmoid':
        return torch.sigmoid(x)
    elif act_type == 'tanh':
        return torch.tanh(x)
    return x

def predict_blocks(x, e, hyperedge_index, sheaf_lin, args):
    """
    Standard prediction: sigma(MLP(x_v || h_e))
    """
    row, col = hyperedge_index
    xs = torch.index_select(x, dim=0, index=row)
    es = torch.index_select(e, dim=0, index=col)

    # Cat: NxEx2f -> Proj: NxExd
    h_sheaf = torch.cat((xs, es), dim=-1) 
    h_sheaf = sheaf_lin(h_sheaf)
    
    return apply_activation(h_sheaf, args.sheaf_act)

def predict_blocks_var2(x, hyperedge_index, sheaf_lin, args):
    """
    Variation 2: Uses scatter_mean of nodes to approximate edge features.
    e_j = avg(h_v)
    """
    row, col = hyperedge_index
    # Re-compute edge features based on mean of connected nodes
    e = scatter_mean(x[row], col, dim=0)
    
    xs = torch.index_select(x, dim=0, index=row)
    es = torch.index_select(e, dim=0, index=col)

    h_sheaf = torch.cat((xs, es), dim=-1)
    h_sheaf = sheaf_lin(h_sheaf)
    
    return apply_activation(h_sheaf, args.sheaf_act)

def predict_blocks_var3(x, hyperedge_index, sheaf_lin, sheaf_lin2, args):
    """
    Variation 3: Universal approximation logic.
    e_j = sum(phi(x_v))
    """
    row, col = hyperedge_index
    xs = torch.index_select(x, dim=0, index=row)

    # phi(x_v)
    x_e = sheaf_lin2(x)
    # sum(phi(x_v))
    e = scatter_add(x_e[row], col, dim=0)  
    es = torch.index_select(e, dim=0, index=col)

    h_sheaf = torch.cat((xs, es), dim=-1)
    h_sheaf = sheaf_lin(h_sheaf)
    
    return apply_activation(h_sheaf, args.sheaf_act)

def predict_blocks_cp_decomp(x, hyperedge_index, cp_W, cp_V, sheaf_lin, args):
    """
    CP Decomposition based prediction.
    """
    row, col = hyperedge_index
    xs = torch.index_select(x, dim=0, index=row)

    # Append 1s for bias handling in CP decomp context
    xs_ones = torch.cat((xs, torch.ones(xs.shape[0], 1).to(xs.device)), dim=-1) 
    xs_ones_proj = torch.tanh(cp_W(xs_ones)) 
    
    # Aggregation
    xs_prod = scatter(xs_ones_proj, col, dim=0, reduce="mul") 
    e = torch.relu(cp_V(xs_prod))
    e = e + torch.relu(scatter_add(x[row], col, dim=0))
    es = torch.index_select(e, dim=0, index=col)

    h_sheaf = torch.cat((xs, es), dim=-1)
    h_sheaf = sheaf_lin(h_sheaf)
    
    return apply_activation(h_sheaf, args.sheaf_act)


# =============================================================================
# Section 3: Standard sheaf Builders (Return Indices & Values)
# Used for standard sheafHyperGNN
# =============================================================================

class sheafBuilderDiag(nn.Module):
    """Builds diagonal dxd blocks."""
    def __init__(self, args):
        super(sheafBuilderDiag, self).__init__()
        self.args = args
        self.prediction_type = args.sheaf_pred_block
        self.sheaf_dropout = args.sheaf_dropout
        self.special_head = args.sheaf_special_head
        self.d = args.heads
        self.MLP_hidden = args.MLP_hidden
        self.norm = args.AllSet_input_norm
        self.dropout = args.dropout
        
        self.sheaf_lin = MLP(in_channels=2*self.MLP_hidden, hidden_channels=args.MLP_hidden,
                             out_channels=self.d, num_layers=1, dropout=0.0,
                             Normalization='ln', InputNorm=self.norm)

        if self.prediction_type == 'MLP_var3':
            self.sheaf_lin2 = MLP(in_channels=self.MLP_hidden, hidden_channels=args.MLP_hidden,
                                  out_channels=args.MLP_hidden, num_layers=1, dropout=0.0,
                                  Normalization='ln', InputNorm=self.norm)
        elif self.prediction_type == 'cp_decomp':
            self.cp_W = MLP(in_channels=self.MLP_hidden+1, hidden_channels=args.MLP_hidden,
                            out_channels=args.MLP_hidden, num_layers=1, dropout=0.0,
                            Normalization='ln', InputNorm=self.norm)
            self.cp_V = MLP(in_channels=args.MLP_hidden, hidden_channels=args.MLP_hidden,
                            out_channels=args.MLP_hidden, num_layers=1, dropout=0.0,
                            Normalization='ln', InputNorm=self.MLP_hidden)

    def reset_parameters(self):
        self.sheaf_lin.reset_parameters()
        if self.prediction_type == 'MLP_var3':
            self.sheaf_lin2.reset_parameters()
        elif self.prediction_type == 'cp_decomp':
            self.cp_W.reset_parameters()
            self.cp_V.reset_parameters()

    def forward(self, x, e, hyperedge_index):
        # Flatten heads for processing: N x d x f -> N x f
        num_nodes = x.shape[0] // self.d
        num_edges = hyperedge_index[1].max().item() + 1
        x = x.view(num_nodes, self.d, x.shape[-1]).mean(1) 
        e = e.view(num_edges, self.d, e.shape[-1]).mean(1)

        # Predict Values
        if self.prediction_type == 'MLP_var1':
            h_sheaf = predict_blocks(x, e, hyperedge_index, self.sheaf_lin, self.args)
        elif self.prediction_type == 'MLP_var2':
            h_sheaf = predict_blocks_var2(x, hyperedge_index, self.sheaf_lin, self.args)
        elif self.prediction_type == 'MLP_var3':
            h_sheaf = predict_blocks_var3(x, hyperedge_index, self.sheaf_lin, self.sheaf_lin2, self.args)
        elif self.prediction_type == 'cp_decomp':
            h_sheaf = predict_blocks_cp_decomp(x, hyperedge_index, self.cp_W, self.cp_V, self.sheaf_lin, self.args)
        
        if self.sheaf_dropout:
            h_sheaf = F.dropout(h_sheaf, p=self.dropout, training=self.training)
        
        if self.special_head:
            new_head_mask = torch.tensor([1]*(self.d-1) + [0], device=x.device)
            new_head = torch.tensor([0]*(self.d-1) + [1], device=x.device)
            h_sheaf = h_sheaf * new_head_mask + new_head
        
        self.h_sheaf = h_sheaf # Stored for potential debug/testing
        h_sheaf_attributes = h_sheaf.reshape(-1) #(d*K)

        # Generate Indices for the large sparse matrix (Nd x Ed)
        # Broadcasting logic: [i, j] -> block diagonal [d*i+k, d*j+k]
        d_range = torch.arange(self.d, device=x.device).view(1,-1,1).repeat(2,1,1) # 2xdx1
        hyperedge_index = hyperedge_index.unsqueeze(1) # 2x1xK
        hyperedge_index = self.d * hyperedge_index + d_range # 2xdxK
        h_sheaf_index = hyperedge_index.permute(0,2,1).reshape(2,-1) # 2x(d*K)

        return h_sheaf_index, h_sheaf_attributes


class sheafBuilderGeneral(nn.Module):
    """Builds full general dxd blocks."""
    def __init__(self, args):
        super(sheafBuilderGeneral, self).__init__()
        self.args = args
        self.prediction_type = args.sheaf_pred_block
        self.sheaf_dropout = args.sheaf_dropout
        self.d = args.heads
        self.MLP_hidden = args.MLP_hidden
        self.norm = args.AllSet_input_norm
        self.dropout = args.dropout

        self.general_sheaf_lin = MLP(in_channels=2*self.MLP_hidden, hidden_channels=args.MLP_hidden,
                                     out_channels=self.d*self.d, num_layers=1, dropout=0.0,
                                     Normalization='ln', InputNorm=self.norm)
        
        if self.prediction_type == 'MLP_var3':
            self.general_sheaf_lin2 = MLP(in_channels=self.MLP_hidden, hidden_channels=args.MLP_hidden,
                                          out_channels=args.MLP_hidden, num_layers=1, dropout=0.0,
                                          Normalization='ln', InputNorm=self.norm)
        if self.prediction_type == 'cp_decomp':
            self.cp_W = MLP(in_channels=self.MLP_hidden+1, hidden_channels=args.MLP_hidden,
                            out_channels=args.MLP_hidden, num_layers=1, dropout=0.0,
                            Normalization='ln', InputNorm=self.norm)
            self.cp_V = MLP(in_channels=args.MLP_hidden, hidden_channels=args.MLP_hidden,
                            out_channels=args.MLP_hidden, num_layers=1, dropout=0.0,
                            Normalization='ln', InputNorm=self.MLP_hidden)

    def reset_parameters(self):
        self.general_sheaf_lin.reset_parameters()
        if self.prediction_type == 'MLP_var3':
            self.general_sheaf_lin2.reset_parameters()
        elif self.prediction_type == 'cp_decomp':
            self.cp_W.reset_parameters()
            self.cp_V.reset_parameters()

    def forward(self, x, e, hyperedge_index, debug=False):
        num_nodes =  x.shape[0] // self.d
        num_edges = hyperedge_index[1].max().item() + 1
        x = x.view(num_nodes, self.d, x.shape[-1]).mean(1)
        e = e.view(num_edges, self.d, e.shape[-1]).mean(1)

        if self.prediction_type == 'MLP_var1':
            h_general_sheaf = predict_blocks(x, e, hyperedge_index, self.general_sheaf_lin, self.args)
        elif self.prediction_type == 'MLP_var2':
            h_general_sheaf = predict_blocks_var2(x, hyperedge_index, self.general_sheaf_lin, self.args)
        elif self.prediction_type == 'MLP_var3':
            h_general_sheaf = predict_blocks_var3(x, hyperedge_index, self.general_sheaf_lin, self.general_sheaf_lin2, self.args)
        elif self.prediction_type == 'cp_decomp':
            h_general_sheaf = predict_blocks_cp_decomp(x, hyperedge_index, self.cp_W, self.cp_V, self.general_sheaf_lin, self.args)
        
        if debug: self.h_general_sheaf = h_general_sheaf

        if self.sheaf_dropout:
            h_general_sheaf = F.dropout(h_general_sheaf, p=self.dropout, training=self.training)

        # Generate Indices for General Blocks (Full dxd)
        d_range = torch.arange(self.d, device=x.device)
        d_range_edges = d_range.repeat(self.d).view(-1,1) # [0,1..d, 0,1..d...]
        d_range_nodes = d_range.repeat_interleave(self.d).view(-1,1) # [0,0..0, 1,1..1...]
        
        hyperedge_index = hyperedge_index.unsqueeze(1) 
        hyperedge_index_0 = self.d * hyperedge_index[0] + d_range_nodes
        hyperedge_index_0 = hyperedge_index_0.permute((1,0)).reshape(1,-1)
        
        hyperedge_index_1 = self.d * hyperedge_index[1] + d_range_edges
        hyperedge_index_1 = hyperedge_index_1.permute((1,0)).reshape(1,-1)
        
        h_general_sheaf_index = torch.concat((hyperedge_index_0, hyperedge_index_1), 0)
        h_general_sheaf_attributes = h_general_sheaf.reshape(-1)
        
        return h_general_sheaf_index, h_general_sheaf_attributes


class sheafBuilderOrtho(nn.Module):
    """Builds Orthogonal dxd blocks."""
    def __init__(self, args):
        super(sheafBuilderOrtho, self).__init__()
        self.args = args
        self.prediction_type = args.sheaf_pred_block
        self.sheaf_dropout = args.sheaf_dropout
        self.special_head = args.sheaf_special_head
        self.d = args.heads
        self.MLP_hidden = args.MLP_hidden
        self.norm = args.AllSet_input_norm
        self.dropout = args.dropout

        self.orth_transform = Orthogonal(d=self.d, orthogonal_map='householder')

        self.orth_sheaf_lin = MLP(in_channels=2*self.MLP_hidden, hidden_channels=args.MLP_hidden,
                                  out_channels=self.d*(self.d-1)//2, num_layers=1, dropout=0.0,
                                  Normalization='ln', InputNorm=self.norm)
        
        if self.prediction_type == 'MLP_var3':
            self.orth_sheaf_lin2 = MLP(in_channels=self.MLP_hidden, hidden_channels=args.MLP_hidden,
                                       out_channels=args.MLP_hidden, num_layers=1, dropout=0.0,
                                       Normalization='ln', InputNorm=self.norm)
        if self.prediction_type == 'cp_decomp':
            self.cp_W = MLP(in_channels=self.MLP_hidden+1, hidden_channels=args.MLP_hidden,
                            out_channels=args.MLP_hidden, num_layers=1, dropout=0.0,
                            Normalization='ln', InputNorm=self.norm)
            self.cp_V = MLP(in_channels=args.MLP_hidden, hidden_channels=args.MLP_hidden,
                            out_channels=args.MLP_hidden, num_layers=1, dropout=0.0,
                            Normalization='ln', InputNorm=self.MLP_hidden)

    def reset_parameters(self):
        self.orth_sheaf_lin.reset_parameters()
        if self.prediction_type == 'MLP_var3':
            self.orth_sheaf_lin2.reset_parameters()
        elif self.prediction_type == 'cp_decomp':
            self.cp_W.reset_parameters()
            self.cp_V.reset_parameters()

    def forward(self, x, e, hyperedge_index, debug=False):
        num_nodes = x.shape[0] // self.d
        num_edges = hyperedge_index[1].max().item() + 1
        x = x.view(num_nodes, self.d, x.shape[-1]).mean(1)
        e = e.view(num_edges, self.d, e.shape[-1]).mean(1)

        if self.prediction_type == 'MLP_var1':
            h_orth_sheaf = predict_blocks(x, e, hyperedge_index, self.orth_sheaf_lin, self.args)
        elif self.prediction_type == 'MLP_var2':
            h_orth_sheaf = predict_blocks_var2(x, hyperedge_index, self.orth_sheaf_lin, self.args)
        elif self.prediction_type == 'MLP_var3':
            h_orth_sheaf = predict_blocks_var3(x, hyperedge_index, self.orth_sheaf_lin, self.orth_sheaf_lin2, self.args)
        elif self.prediction_type == 'cp_decomp':
            h_orth_sheaf = predict_blocks_cp_decomp(x, hyperedge_index, self.cp_W, self.cp_V, self.orth_sheaf_lin, self.args)
        
        # Apply Householder transform
        h_orth_sheaf = self.orth_transform(h_orth_sheaf) 

        if self.sheaf_dropout:
            h_orth_sheaf = F.dropout(h_orth_sheaf, p=self.dropout, training=self.training)
        
        if self.special_head:
            # Add identity-like structure to the last head
            new_head_mask = torch.ones((self.d, self.d), device=x.device)
            new_head_mask[:,-1] = 0
            new_head_mask[-1,:] = 0
            
            new_head = torch.zeros((self.d, self.d), device=x.device)
            new_head[-1,-1] = 1
            
            h_orth_sheaf = h_orth_sheaf * new_head_mask + new_head
            h_orth_sheaf = h_orth_sheaf.float()

        # Generate Indices (Same as General sheaf)
        d_range = torch.arange(self.d, device=x.device)
        d_range_edges = d_range.repeat(self.d).view(-1,1)
        d_range_nodes = d_range.repeat_interleave(self.d).view(-1,1)
        
        hyperedge_index = hyperedge_index.unsqueeze(1) 
        hyperedge_index_0 = self.d * hyperedge_index[0] + d_range_nodes
        hyperedge_index_0 = hyperedge_index_0.permute((1,0)).reshape(1,-1)
        hyperedge_index_1 = self.d * hyperedge_index[1] + d_range_edges
        hyperedge_index_1 = hyperedge_index_1.permute((1,0)).reshape(1,-1)
        
        h_orth_sheaf_index = torch.concat((hyperedge_index_0, hyperedge_index_1), 0)
        h_orth_sheaf_attributes = h_orth_sheaf.reshape(-1)
        
        return h_orth_sheaf_index, h_orth_sheaf_attributes


class sheafBuilderLowRank(nn.Module):
    """Builds Low-Rank dxd blocks via product AB^T + diag(c)."""
    def __init__(self, args):
        super(sheafBuilderLowRank, self).__init__()
        self.args = args
        self.prediction_type = args.sheaf_pred_block
        self.sheaf_dropout = args.sheaf_dropout
        self.special_head = args.sheaf_special_head
        self.d = args.heads
        self.MLP_hidden = args.MLP_hidden
        self.norm = args.AllSet_input_norm
        self.norm_type = args.sheaf_normtype
        self.rank = args.rank
        self.dropout = args.dropout

        # Output dims: A(d*rank) + B(d*rank) + C(d)
        out_dims = 2 * self.d * self.rank + self.d

        self.general_sheaf_lin = MLP(in_channels=2*self.MLP_hidden, hidden_channels=args.MLP_hidden,
                                     out_channels=out_dims, num_layers=1, dropout=0.0,
                                     Normalization='ln', InputNorm=self.norm)

        if self.prediction_type == 'MLP_var3':
            self.general_sheaf_lin2 = MLP(in_channels=self.MLP_hidden, hidden_channels=args.MLP_hidden,
                                          out_channels=args.MLP_hidden, num_layers=1, dropout=0.0,
                                          Normalization='ln', InputNorm=self.norm)
        if self.prediction_type == 'cp_decomp':
            self.cp_W = MLP(in_channels=self.MLP_hidden+1, hidden_channels=args.MLP_hidden,
                            out_channels=args.MLP_hidden, num_layers=1, dropout=0.0,
                            Normalization='ln', InputNorm=self.norm)
            self.cp_V = MLP(in_channels=args.MLP_hidden, hidden_channels=args.MLP_hidden,
                            out_channels=args.MLP_hidden, num_layers=1, dropout=0.0,
                            Normalization='ln', InputNorm=self.MLP_hidden)

    def reset_parameters(self):
        self.general_sheaf_lin.reset_parameters()
        if self.prediction_type == 'MLP_var3':
            self.general_sheaf_lin2.reset_parameters()
        elif self.prediction_type == 'cp_decomp':
            self.cp_W.reset_parameters()
            self.cp_V.reset_parameters()

    def forward(self, x, e, hyperedge_index, debug=False):
        num_nodes =  x.shape[0] // self.d
        num_edges = hyperedge_index[1].max().item() + 1
        x = x.view(num_nodes, self.d, x.shape[-1]).mean(1)
        e = e.view(num_edges, self.d, e.shape[-1]).mean(1)

        if self.prediction_type == 'MLP_var1':
            h_general_sheaf = predict_blocks(x, e, hyperedge_index, self.general_sheaf_lin, self.args)
        elif self.prediction_type == 'MLP_var2':
            h_general_sheaf = predict_blocks_var2(x, hyperedge_index, self.general_sheaf_lin, self.args)
        elif self.prediction_type == 'MLP_var3':
            h_general_sheaf = predict_blocks_var3(x, hyperedge_index, self.general_sheaf_lin, self.general_sheaf_lin2, self.args)
        elif self.prediction_type == 'cp_decomp':
            h_general_sheaf = predict_blocks_cp_decomp(x, hyperedge_index, self.cp_W, self.cp_V, self.general_sheaf_lin, self.args)

        # Reconstruct from Low Rank parts
        # h_general_sheaf is nnz x (2*d*r + d)
        split_idx_1 = self.d * self.rank
        split_idx_2 = 2 * self.d * self.rank
        
        part_A = h_general_sheaf[:, :split_idx_1].reshape(-1, self.d, self.rank)
        part_B = h_general_sheaf[:, split_idx_1:split_idx_2].reshape(-1, self.d, self.rank)
        part_C = h_general_sheaf[:, split_idx_2:].reshape(-1, self.d)

        # AB^T + diag(C)
        h_sheaf = torch.bmm(part_A, part_B.transpose(2,1)) 
        diag = torch.diag_embed(part_C)
        h_sheaf = h_sheaf + diag
        
        # Flatten back to nnz x (d*d)
        h_sheaf = h_sheaf.reshape(h_sheaf.shape[0], self.d*self.d)
        
        if debug: self.h_general_sheaf = h_sheaf
        if self.sheaf_dropout:
            h_sheaf = F.dropout(h_sheaf, p=self.dropout, training=self.training)

        # Generate Indices (Same as General)
        d_range = torch.arange(self.d, device=x.device)
        d_range_edges = d_range.repeat(self.d).view(-1,1) 
        d_range_nodes = d_range.repeat_interleave(self.d).view(-1,1)
        
        hyperedge_index = hyperedge_index.unsqueeze(1) 
        hyperedge_index_0 = self.d * hyperedge_index[0] + d_range_nodes
        hyperedge_index_0 = hyperedge_index_0.permute((1,0)).reshape(1,-1)
        hyperedge_index_1 = self.d * hyperedge_index[1] + d_range_edges
        hyperedge_index_1 = hyperedge_index_1.permute((1,0)).reshape(1,-1)
        h_general_sheaf_index = torch.concat((hyperedge_index_0, hyperedge_index_1), 0)

        # Block Normalization Logic
        if self.norm_type == 'block_norm':
            h_sheaf_view = h_sheaf.reshape(-1, self.d, self.d)
            row, col = hyperedge_index.squeeze(1)[0], hyperedge_index.squeeze(1)[1]
            num_nodes = row.max().item() + 1
            num_edges = col.max().item() + 1

            to_be_inv_nodes = torch.bmm(h_sheaf_view, h_sheaf_view.permute(0,2,1)) 
            to_be_inv_nodes = scatter_add(to_be_inv_nodes, row, dim=0, dim_size=num_nodes)

            to_be_inv_edges = torch.bmm(h_sheaf_view.permute(0,2,1), h_sheaf_view)
            to_be_inv_edges = scatter_add(to_be_inv_edges, col, dim=0, dim_size=num_edges)

            d_sqrt_inv_nodes = utils.batched_sym_matrix_pow(to_be_inv_nodes, -1.0) 
            d_sqrt_inv_edges = utils.batched_sym_matrix_pow(to_be_inv_edges, -1.0) 

            d_sqrt_inv_nodes_large = torch.index_select(d_sqrt_inv_nodes, dim=0, index=row)
            d_sqrt_inv_edges_large = torch.index_select(d_sqrt_inv_edges, dim=0, index=col)

            alpha_norm = torch.bmm(d_sqrt_inv_nodes_large, h_sheaf_view)
            alpha_norm = torch.bmm(alpha_norm, d_sqrt_inv_edges_large)
            h_sheaf = alpha_norm.clamp(min=-1, max=1)
            h_sheaf = h_sheaf.reshape(-1, self.d*self.d)

        h_general_sheaf_attributes = h_sheaf.reshape(-1)
        return h_general_sheaf_index, h_general_sheaf_attributes


# =============================================================================
# Section 4: HGCN sheaf Builders (Return Values Only)
# Used for sheafHyperGCN (which constructs Laplacian manually)
# =============================================================================

class HGCNsheafBuilderDiag(nn.Module):
    def __init__(self, args, hidden_dim):
        super(HGCNsheafBuilderDiag, self).__init__()
        self.args = args
        self.prediction_type = args.sheaf_pred_block
        self.sheaf_dropout = args.sheaf_dropout
        self.special_head = args.sheaf_special_head
        self.d = args.heads
        self.MLP_hidden = hidden_dim
        self.norm = args.AllSet_input_norm
        self.dropout = args.dropout

        in_ch_factor = 1 if self.prediction_type == 'MLP_var1' else 2
        
        self.sheaf_lin = MLP(in_channels=in_ch_factor*self.MLP_hidden + (args.MLP_hidden if in_ch_factor==1 else 0), 
                             hidden_channels=args.MLP_hidden, out_channels=self.d,
                             num_layers=1, dropout=0.0, Normalization='ln', InputNorm=self.norm)

        if self.prediction_type == 'MLP_var3':
            self.sheaf_lin2 = MLP(in_channels=self.MLP_hidden, hidden_channels=args.MLP_hidden,
                                  out_channels=self.MLP_hidden, num_layers=1, dropout=0.0,
                                  Normalization='ln', InputNorm=self.norm)
        if self.prediction_type == 'cp_decomp':
            self.cp_W = MLP(in_channels=self.MLP_hidden+1, hidden_channels=args.MLP_hidden,
                            out_channels=args.MLP_hidden, num_layers=1, dropout=0.0,
                            Normalization='ln', InputNorm=self.norm)
            self.cp_V = MLP(in_channels=args.MLP_hidden, hidden_channels=args.MLP_hidden,
                            out_channels=self.MLP_hidden, num_layers=1, dropout=0.0,
                            Normalization='ln', InputNorm=self.MLP_hidden)

    def reset_parameters(self):
        self.sheaf_lin.reset_parameters()
        if self.prediction_type == 'MLP_var3':
            self.sheaf_lin2.reset_parameters()
        elif self.prediction_type == 'cp_decomp':
            self.cp_W.reset_parameters()
            self.cp_V.reset_parameters()

    def forward(self, x, e, hyperedge_index):
        num_nodes = x.shape[0] // self.d
        num_edges = hyperedge_index[1].max().item() + 1
        x = x.view(num_nodes, self.d, x.shape[-1]).mean(1)
        e = e.view(num_edges, self.d, e.shape[-1]).mean(1)

        if self.prediction_type == 'MLP_var1':
            h_sheaf = predict_blocks(x, e, hyperedge_index, self.sheaf_lin, self.args)
        elif self.prediction_type == 'MLP_var2':
            h_sheaf = predict_blocks_var2(x, hyperedge_index, self.sheaf_lin, self.args)
        elif self.prediction_type == 'MLP_var3':
            h_sheaf = predict_blocks_var3(x, hyperedge_index, self.sheaf_lin, self.sheaf_lin2, self.args)
        elif self.prediction_type == 'cp_decomp':
            h_sheaf = predict_blocks_cp_decomp(x, hyperedge_index, self.cp_W, self.cp_V, self.sheaf_lin, self.args)

        if self.sheaf_dropout:
            h_sheaf = F.dropout(h_sheaf, p=self.dropout, training=self.training)
       
        return h_sheaf


class HGCNsheafBuilderGeneral(nn.Module):
    def __init__(self, args, hidden_dim):
        super(HGCNsheafBuilderGeneral, self).__init__()
        self.args = args
        self.prediction_type = args.sheaf_pred_block
        self.sheaf_dropout = args.sheaf_dropout
        self.d = args.heads
        self.MLP_hidden = hidden_dim
        self.norm = args.AllSet_input_norm
        self.dropout = args.dropout
        
        in_ch_factor = 1 if self.prediction_type == 'MLP_var1' else 2

        self.sheaf_lin = MLP(in_channels=in_ch_factor*self.MLP_hidden + (args.MLP_hidden if in_ch_factor==1 else 0), 
                             hidden_channels=args.MLP_hidden, out_channels=self.d*self.d,
                             num_layers=1, dropout=0.0, Normalization='ln', InputNorm=self.norm)

        if self.prediction_type == 'MLP_var3':
            self.sheaf_lin2 = MLP(in_channels=self.MLP_hidden, hidden_channels=args.MLP_hidden,
                                  out_channels=self.MLP_hidden, num_layers=1, dropout=0.0,
                                  Normalization='ln', InputNorm=self.norm)
        if self.prediction_type == 'cp_decomp':
            self.cp_W = MLP(in_channels=self.MLP_hidden+1, hidden_channels=args.MLP_hidden,
                            out_channels=args.MLP_hidden, num_layers=1, dropout=0.0,
                            Normalization='ln', InputNorm=self.norm)
            self.cp_V = MLP(in_channels=args.MLP_hidden, hidden_channels=args.MLP_hidden,
                            out_channels=self.MLP_hidden, num_layers=1, dropout=0.0,
                            Normalization='ln', InputNorm=self.MLP_hidden)

    def reset_parameters(self):
        self.sheaf_lin.reset_parameters()
        if self.prediction_type == 'MLP_var3':
            self.sheaf_lin2.reset_parameters()
        elif self.prediction_type == 'cp_decomp':
            self.cp_W.reset_parameters()
            self.cp_V.reset_parameters()

    def forward(self, x, e, hyperedge_index):
        num_nodes = x.shape[0] // self.d
        num_edges = hyperedge_index[1].max().item() + 1
        x = x.view(num_nodes, self.d, x.shape[-1]).mean(1)
        e = e.view(num_edges, self.d, e.shape[-1]).mean(1)

        if self.prediction_type == 'MLP_var1':
            h_sheaf = predict_blocks(x, e, hyperedge_index, self.sheaf_lin, self.args)
        elif self.prediction_type == 'MLP_var2':
            h_sheaf = predict_blocks_var2(x, hyperedge_index, self.sheaf_lin, self.args)
        elif self.prediction_type == 'MLP_var3':
            h_sheaf = predict_blocks_var3(x, hyperedge_index, self.sheaf_lin, self.sheaf_lin2, self.args)
        elif self.prediction_type == 'cp_decomp':
            h_sheaf = predict_blocks_cp_decomp(x, hyperedge_index, self.cp_W, self.cp_V, self.sheaf_lin, self.args)

        if self.sheaf_dropout:
            h_sheaf = F.dropout(h_sheaf, p=self.dropout, training=self.training)
        
        return h_sheaf


class HGCNsheafBuilderOrtho(nn.Module):
    def __init__(self, args, hidden_dim):
        super(HGCNsheafBuilderOrtho, self).__init__()
        self.args = args
        self.prediction_type = args.sheaf_pred_block
        self.sheaf_dropout = args.sheaf_dropout
        self.d = args.heads
        self.MLP_hidden = hidden_dim
        self.norm = args.AllSet_input_norm
        self.orth_transform = Orthogonal(d=self.d, orthogonal_map='householder')
        self.dropout = args.dropout

        in_ch_factor = 1 if self.prediction_type == 'MLP_var1' else 2
        
        self.sheaf_lin = MLP(in_channels=in_ch_factor*self.MLP_hidden + (args.MLP_hidden if in_ch_factor==1 else 0), 
                             hidden_channels=args.MLP_hidden, out_channels=self.d*(self.d-1)//2,
                             num_layers=1, dropout=0.0, Normalization='ln', InputNorm=self.norm)

        if self.prediction_type == 'MLP_var3':
            self.sheaf_lin2 = MLP(in_channels=self.MLP_hidden, hidden_channels=args.MLP_hidden,
                                  out_channels=self.MLP_hidden, num_layers=1, dropout=0.0,
                                  Normalization='ln', InputNorm=self.norm)
        if self.prediction_type == 'cp_decomp':
            self.cp_W = MLP(in_channels=self.MLP_hidden+1, hidden_channels=args.MLP_hidden,
                            out_channels=args.MLP_hidden, num_layers=1, dropout=0.0,
                            Normalization='ln', InputNorm=self.norm)
            self.cp_V = MLP(in_channels=args.MLP_hidden, hidden_channels=args.MLP_hidden,
                            out_channels=self.MLP_hidden, num_layers=1, dropout=0.0,
                            Normalization='ln', InputNorm=self.MLP_hidden)

    def reset_parameters(self):
        self.sheaf_lin.reset_parameters()
        if self.prediction_type == 'MLP_var3':
            self.sheaf_lin2.reset_parameters()
        elif self.prediction_type == 'cp_decomp':
            self.cp_W.reset_parameters()
            self.cp_V.reset_parameters()

    def forward(self, x, e, hyperedge_index):
        num_nodes = x.shape[0] // self.d
        num_edges = hyperedge_index[1].max().item() + 1
        x = x.view(num_nodes, self.d, x.shape[-1]).mean(1)
        e = e.view(num_edges, self.d, e.shape[-1]).mean(1)

        if self.prediction_type == 'MLP_var1':
            h_sheaf = predict_blocks(x, e, hyperedge_index, self.sheaf_lin, self.args)
        elif self.prediction_type == 'MLP_var2':
            h_sheaf = predict_blocks_var2(x, hyperedge_index, self.sheaf_lin, self.args)
        elif self.prediction_type == 'MLP_var3':
            h_sheaf = predict_blocks_var3(x, hyperedge_index, self.sheaf_lin, self.sheaf_lin2, self.args)
        elif self.prediction_type == 'cp_decomp':
            h_sheaf = predict_blocks_cp_decomp(x, hyperedge_index, self.cp_W, self.cp_V, self.sheaf_lin, self.args)

        h_sheaf = self.orth_transform(h_sheaf)

        if self.sheaf_dropout:
            h_sheaf = F.dropout(h_sheaf, p=self.dropout, training=self.training)

        return h_sheaf


class HGCNsheafBuilderLowRank(nn.Module):
    def __init__(self, args, hidden_dim):
        super(HGCNsheafBuilderLowRank, self).__init__()
        self.args = args
        self.prediction_type = args.sheaf_pred_block
        self.sheaf_dropout = args.sheaf_dropout
        self.d = args.heads
        self.MLP_hidden = hidden_dim
        self.norm = args.AllSet_input_norm
        self.rank = args.rank
        self.dropout = args.dropout
        
        in_ch_factor = 1 if self.prediction_type == 'MLP_var1' else 2
        out_dims = 2*self.d*self.rank + self.d

        self.sheaf_lin = MLP(in_channels=in_ch_factor*self.MLP_hidden + (args.MLP_hidden if in_ch_factor==1 else 0), 
                             hidden_channels=args.MLP_hidden, out_channels=out_dims,
                             num_layers=1, dropout=0.0, Normalization='ln', InputNorm=self.norm)

        if self.prediction_type == 'MLP_var3':
            self.sheaf_lin2 = MLP(in_channels=self.MLP_hidden, hidden_channels=args.MLP_hidden,
                                  out_channels=self.MLP_hidden, num_layers=1, dropout=0.0,
                                  Normalization='ln', InputNorm=self.norm)
        if self.prediction_type == 'cp_decomp':
            self.cp_W = MLP(in_channels=self.MLP_hidden+1, hidden_channels=args.MLP_hidden,
                            out_channels=args.MLP_hidden, num_layers=1, dropout=0.0,
                            Normalization='ln', InputNorm=self.norm)
            self.cp_V = MLP(in_channels=args.MLP_hidden, hidden_channels=args.MLP_hidden,
                            out_channels=self.MLP_hidden, num_layers=1, dropout=0.0,
                            Normalization='ln', InputNorm=self.MLP_hidden)

    def reset_parameters(self):
        self.sheaf_lin.reset_parameters()
        if self.prediction_type == 'MLP_var3':
            self.sheaf_lin2.reset_parameters()
        elif self.prediction_type == 'cp_decomp':
            self.cp_W.reset_parameters()
            self.cp_V.reset_parameters()

    def forward(self, x, e, hyperedge_index):
        num_nodes = x.shape[0] // self.d
        num_edges = hyperedge_index[1].max().item() + 1
        x = x.view(num_nodes, self.d, x.shape[-1]).mean(1)
        e = e.view(num_edges, self.d, e.shape[-1]).mean(1)

        if self.prediction_type == 'MLP_var1':
            h_sheaf = predict_blocks(x, e, hyperedge_index, self.sheaf_lin, self.args)
        elif self.prediction_type == 'MLP_var2':
            h_sheaf = predict_blocks_var2(x, hyperedge_index, self.sheaf_lin, self.args)
        elif self.prediction_type == 'MLP_var3':
            h_sheaf = predict_blocks_var3(x, hyperedge_index, self.sheaf_lin, self.sheaf_lin2, self.args)
        elif self.prediction_type == 'cp_decomp':
            h_sheaf = predict_blocks_cp_decomp(x, hyperedge_index, self.cp_W, self.cp_V, self.sheaf_lin, self.args)

        # Low-Rank reconstruction: AB^T + diag(C)
        split_idx_1 = self.d * self.rank
        split_idx_2 = 2 * self.d * self.rank
        
        part_A = h_sheaf[:, :split_idx_1].reshape(-1, self.d, self.rank)
        part_B = h_sheaf[:, split_idx_1:split_idx_2].reshape(-1, self.d, self.rank)
        part_C = h_sheaf[:, split_idx_2:].reshape(-1, self.d)

        h_sheaf = torch.bmm(part_A, part_B.transpose(2,1)) 
        diag = torch.diag_embed(part_C)
        h_sheaf = h_sheaf + diag
        
        h_sheaf = h_sheaf.reshape(h_sheaf.shape[0], self.d*self.d)

        if self.sheaf_dropout:
            h_sheaf = F.dropout(h_sheaf, p=self.dropout, training=self.training)

        return h_sheaf