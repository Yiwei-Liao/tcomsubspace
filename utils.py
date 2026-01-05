#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility Module for Hypergraph Neural Networks.
Contains:
1. HyperGraphConvolution Layer (Standard GCN-like for Hypergraphs)
2. Sparse Matrix Multiplication with Autograd
3. Laplacian Construction Helpers (Clique Expansion variants)
4. General Sparse Tensor & Matrix Utilities
"""

import math
import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


# =============================================================================
# Section 1: HyperGraph Convolution Layer
# =============================================================================

class HyperGraphConvolution(Module):
    """
    Simple GCN layer for Hypergraphs, adapted from https://arxiv.org/abs/1609.02907
    It can optionally re-approximate the Laplacian structure during forward pass.
    """
    def __init__(self, in_features, out_features, reapproximate=True, cuda=None):
        super(HyperGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.reapproximate = reapproximate

        self.W = Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        std = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, structure, H, m=True):
        """
        Args:
            structure: The graph structure (adjacency/laplacian).
            H: Node features.
            m: Mediator flag for Laplacian approximation.
        """
        HW = torch.mm(H, self.W)
        
        # Re-calculate Laplacian based on current features if requested
        if self.reapproximate:
            n = H.shape[0]
            X = HW.cpu().detach().numpy()    
            A = construct_laplacian(n, structure, X, m)
        else: 
            A = structure

        # Ensure A is a sparse tensor on the correct device
        if not isinstance(A, torch.Tensor):
            A = ssm2tst(A)
        A = A.to(H.device)

        # Sparse Matrix Multiplication: A * (H * W)
        AHW = SparseMM.apply(A, HW)     
        return AHW + self.bias

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class SparseMM(torch.autograd.Function):
    """
    Sparse x Dense matrix multiplication with autograd support.
    Reference: https://discuss.pytorch.org/t/does-pytorch-support-autograd-on-sparse-matrix/6156/7
    """
    @staticmethod
    def forward(ctx, M1, M2):
        ctx.save_for_backward(M1, M2)
        return torch.mm(M1, M2)

    @staticmethod
    def backward(ctx, g):
        M1, M2 = ctx.saved_tensors
        g1 = g2 = None

        if ctx.needs_input_grad[0]:
            g1 = torch.mm(g, M2.t())

        if ctx.needs_input_grad[1]:
            g2 = torch.mm(M1.t(), g)

        return g1, g2


# =============================================================================
# Section 2: Laplacian Construction (HyperGCN Specific)
# =============================================================================

def update(Se, Ie, mediator, weights, c):
    """
    Updates the weight on {Se,mediator} and {Ie,mediator}.
    This function is required by external files (e.g., hgcn_sheaf_laplacians.py).
    """    
    if (Se, mediator) not in weights:
        weights[(Se, mediator)] = 0
    weights[(Se, mediator)] += float(1/c)

    if (Ie, mediator) not in weights:
        weights[(Ie, mediator)] = 0
    weights[(Ie, mediator)] += float(1/c)

    if (mediator, Se) not in weights:
        weights[(mediator, Se)] = 0
    weights[(mediator, Se)] += float(1/c)

    if (mediator, Ie) not in weights:
        weights[(mediator, Ie)] = 0
    weights[(mediator, Ie)] += float(1/c)

    return weights

def construct_laplacian(V, E, X, m):
    """
    Approximates the Hypergraph Laplacian via Clique Expansion with/without mediators.
    This method projects node features to find 'Supremum' and 'Infimum' nodes 
    in a hyperedge to create edges.

    Args:
        V: number of vertices
        E: dictionary of hyperedges (key: hyperedge_id, value: list of nodes)
        X: features on the vertices
        m: Boolean, True enables mediators
    
    Returns:
        A: Sparse Adjacency Matrix (scipy/torch tensor converted)
    """
    edges = []
    weights = {}
    rv = np.random.rand(X.shape[1]) # Random vector for projection

    for k in E.keys():
        hyperedge = list(E[k])
        
        # Projection to find representative nodes
        p = np.dot(X[hyperedge], rv)   
        s, i = np.argmax(p), np.argmin(p)
        Se, Ie = hyperedge[s], hyperedge[i]

        # Normalization constant
        # Two stars with mediators vs Clique
        c = max(1, 2 * len(hyperedge) - 3)    

        if m:
            # 1. Connect Supremum (Se) and Infimum (Ie)
            edges.extend([[Se, Ie], [Ie, Se]])
            _add_weight(weights, Se, Ie, 1/c)
            _add_weight(weights, Ie, Se, 1/c)
            
            # 2. Connect Se and Ie with each mediator
            for mediator in hyperedge:
                if mediator != Se and mediator != Ie:
                    edges.extend([[Se, mediator], [Ie, mediator], [mediator, Se], [mediator, Ie]])
                    # Use the legacy update function internally if needed, or inline logic
                    # Keeping inline logic here for efficiency in this function, 
                    # but 'update' is exposed above for other files.
                    _add_weight(weights, Se, mediator, 1/c)
                    _add_weight(weights, Ie, mediator, 1/c)
                    _add_weight(weights, mediator, Se, 1/c)
                    _add_weight(weights, mediator, Ie, 1/c)
        else:
            # Simple Clique approximation (just Se-Ie connection here based on original logic)
            edges.extend([[Se, Ie], [Ie, Se]])
            w_val = 1.0 / len(hyperedge)
            _add_weight(weights, Se, Ie, w_val)
            _add_weight(weights, Ie, Se, w_val)

    return _build_adjacency(edges, weights, V)


def _add_weight(weights, u, v, val):
    """Helper to accumulate weights safely."""
    if (u, v) not in weights:
        weights[(u, v)] = 0
    weights[(u, v)] += float(val)


def _build_adjacency(edges, weights, n):
    """
    Constructs a normalized sparse adjacency matrix from edge list and weight dict.
    """
    # Create unique list of edges to align with weights
    edge_list = []
    weight_list = []
    
    for (u, v), w in weights.items():
        edge_list.append([u, v])
        weight_list.append(w)
    
    if not edge_list:
        # Handle case with no edges
        adj = sp.coo_matrix((n, n), dtype=np.float32)
    else:
        edge_arr = np.array(edge_list)
        weight_arr = np.array(weight_list)
        adj = sp.coo_matrix((weight_arr, (edge_arr[:, 0], edge_arr[:, 1])), shape=(n, n), dtype=np.float32)

    # Add self-loops
    adj = adj + sp.eye(n)

    # Symmetric Normalization: D^-1/2 A D^-1/2
    A = symnormalise(sp.csr_matrix(adj, dtype=np.float32))
    
    # Convert to Torch Sparse Tensor
    A = ssm2tst(A)
    return A


# =============================================================================
# Section 3: Sparse Matrix Utilities
# =============================================================================

def symnormalise(M):
    """
    Symmetrically normalise sparse matrix: D^{-1/2} M D^{-1/2}
    """
    d = np.array(M.sum(1))
    
    # Power -1/2 with safety checks
    with np.errstate(divide='ignore'):
        dhi = np.power(d, -0.5).flatten()
    dhi[np.isinf(dhi)] = 0.
    
    DHI = sp.diags(dhi)    # D^{-1/2}
    return (DHI.dot(M)).dot(DHI) 


def normalise(M):
    """
    Row-normalise sparse matrix: D^{-1} M
    """
    d = np.array(M.sum(1))
    
    with np.errstate(divide='ignore'):
        di = np.power(d, -1).flatten()
    di[np.isinf(di)] = 0.
    di = np.nan_to_num(di)
    
    DI = sp.diags(di)    # D^{-1}
    return DI.dot(M)


def ssm2tst(M):
    """
    Converts a Scipy Sparse Matrix (ssm) to a Torch Sparse Tensor (tst).
    """
    M = M.tocoo().astype(np.float32)
    
    indices = torch.from_numpy(np.vstack((M.row, M.col))).long()
    values = torch.from_numpy(M.data)
    shape = torch.Size(M.shape)
    
    return torch.sparse.FloatTensor(indices, values, shape)


def sparse_diagonal(diag, shape):
    """
    Creates a sparse diagonal matrix from a vector.
    """
    r, c = shape
    assert r == c, "Sparse diagonal matrix must be square."
    
    indexes = torch.arange(r, device=diag.device)
    indexes = torch.stack([indexes, indexes], dim=0)
    
    return torch.sparse.FloatTensor(indexes, diag, torch.Size(shape))


# =============================================================================
# Section 4: Math & Visualization Helpers
# =============================================================================

# Alias for compatibility with external calls (e.g. hgcn_sheaf_laplacians.py)
# Some files might call Laplacian() instead of construct_laplacian()
Laplacian = construct_laplacian
adjacency = _build_adjacency

def batched_sym_matrix_pow(matrices: torch.Tensor, p: float) -> torch.Tensor:
    """
    Computes power of a batch of symmetric matrices using SVD.
    M^p = V * S^p * V^T
    
    Args:
        matrices: Batch of matrices (B x N x N)
        p: Power exponent
    """
    # SVD is generally more robust for this purpose than eigh
    vecs, vals, _ = torch.linalg.svd(matrices)
    
    # Filter out numerical noise in singular values
    # Threshold based on max value and machine epsilon
    max_vals = vals.max(-1, True).values
    threshold = max_vals * vals.size(-1) * torch.finfo(vals.dtype).eps
    good = vals > threshold
    
    vals = vals.pow(p).where(good, torch.zeros((), device=matrices.device, dtype=matrices.dtype))
    
    # Reconstruct: U * Sigma * V^T (here U=V for symmetric positive semi-definite)
    matrix_power = (vecs * vals.unsqueeze(-2)) @ torch.transpose(vecs, -2, -1)
    return matrix_power


def generate_indices_general(indexes, d):
    """
    Expands indices for block-matrix operations.
    Maps index [i, j] -> block [d*i...d*i+d, d*j...d*j+d]
    """
    d_range = torch.arange(d, device=indexes.device)
    
    # [0, 1, ..., d, 0, 1, ..., d]
    d_range_edges = d_range.repeat(d).view(-1, 1) 
    
    # [0, 0, ..., 0, 1, 1, ..., 1]
    d_range_nodes = d_range.repeat_interleave(d).view(-1, 1)
    
    indexes = indexes.unsqueeze(1) 

    # Expand row indices
    large_indexes_0 = d * indexes[0] + d_range_nodes
    large_indexes_0 = large_indexes_0.permute((1, 0)).reshape(1, -1)
    
    # Expand col indices
    large_indexes_1 = d * indexes[1] + d_range_edges
    large_indexes_1 = large_indexes_1.permute((1, 0)).reshape(1, -1)
    
    large_indexes = torch.cat((large_indexes_0, large_indexes_1), 0)

    return large_indexes


def print_colored_ndarray(map_data, d, row_sep=""):
    """
    Visualizes a matrix with color-coded blocks for debugging.
    """
    def get_color_coded_background(color, i):
        return "\033[4{}m {:.4f} \033[0m".format(color + 1, i)

    map_data = np.round(map_data, 3)
    n, m = map_data.shape 
    n_blocks = n // d
    m_blocks = m // d
    
    color_range_row = np.arange(m_blocks)[np.newaxis, ...].repeat(d, axis=1)
    color_range_col = np.arange(n_blocks)[..., np.newaxis].repeat(d, axis=0)
    color_range = (color_range_row + color_range_col) % 7 # Modulo to keep colors in valid range

    back_map_modified = np.vectorize(get_color_coded_background)(color_range, map_data)
    n, m = back_map_modified.shape
    fmt_str = "\n".join([row_sep.join(["{}"]*m)]*n)
    print(fmt_str.format(*back_map_modified.ravel()))