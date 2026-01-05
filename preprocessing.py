#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Preprocessing Utility Module for Hypergraph Neural Networks.
Contains functions for expanding edges, constructing incidence/adjacency matrices,
normalizing weights, and splitting datasets.
"""

import torch
import numpy as np
from collections import Counter
from itertools import combinations
from torch_scatter import scatter_add, scatter
from torch_geometric.nn.conv.gcn_conv import gcn_norm


# =============================================================================
# Section 1: Edge & Index Processing
# =============================================================================

def expand_edge_index(data, edge_th=0):
    """
    Expands hyperedges by explicitly representing connections.
    Converts node-to-hyperedge relations into a sparse format suitable for
    message passing, creating distinct edge IDs for each node-hyperedge pair.
    
    Args:
        data: PyG data object containing edge_index.
        edge_th: Threshold for edge size. Hyperedges larger than this are ignored.
    """
    edge_index = data.edge_index
    num_nodes = data.n_x
    
    if hasattr(data, 'totedges'):
        num_edges = data.totedges
    else:
        num_edges = data.num_hyperedges

    expanded_n2he_index = []
    
    # Start new edge IDs after the last node ID
    cur_he_id = num_nodes
    # Mapping new_edge_id -> original_he_id (if needed for debugging)
    # new_edge_id_2_original_edge_id = {} 

    # Iterate through all hyperedges
    # Hyperedges are indexed from [num_nodes, num_nodes + num_edges)
    for he_idx in range(num_nodes, num_edges + num_nodes):
        # Find all nodes within the current hyperedge
        selected_he = edge_index[:, edge_index[1] == he_idx]
        size_of_he = selected_he.shape[1]

        # Filter hyperedges based on size threshold
        if edge_th > 0 and size_of_he > edge_th:
            continue

        # Case 1: Self-loop (Hyperedge with only 1 node)
        if size_of_he == 1:
            new_n2he = selected_he.clone()
            new_n2he[1] = cur_he_id
            expanded_n2he_index.append(new_n2he)
            cur_he_id += 1
            continue

        # Case 2: Standard Hyperedge
        # Repeat the node-edge pairs to create a clique-like expansion structure
        # We connect nodes to 'new' intermediate edge representations
        new_n2he = selected_he.repeat_interleave(size_of_he, dim=1)

        # Assign new unique edge IDs for this expansion
        new_edge_ids = torch.arange(cur_he_id, cur_he_id + size_of_he, device=edge_index.device).repeat(size_of_he)
        new_n2he[1] = new_edge_ids

        # Build mapping: node_id -> temp_edge_id
        # This helps identify self-connections which need to be removed
        tmp_node_id_2_he_id_dict = {}
        for idx in range(size_of_he):
            # new_edge_id_2_original_edge_id[cur_he_id] = he_idx
            cur_node_id = selected_he[0][idx].item()
            tmp_node_id_2_he_id_dict[cur_node_id] = cur_he_id + idx
        
        # Update cursor for the next hyperedge
        cur_he_id += size_of_he

        # Remove self-loops (node connecting to the specific edge ID representing itself)
        new_he_select_mask = torch.ones(new_n2he.shape[1], dtype=torch.bool, device=edge_index.device)
        for col_idx in range(new_n2he.shape[1]):
            tmp_node_id = new_n2he[0, col_idx].item()
            tmp_edge_id = new_n2he[1, col_idx].item()
            
            if tmp_node_id_2_he_id_dict[tmp_node_id] == tmp_edge_id:
                new_he_select_mask[col_idx] = False
        
        new_n2he = new_n2he[:, new_he_select_mask]
        expanded_n2he_index.append(new_n2he)

    new_edge_index = torch.cat(expanded_n2he_index, dim=1)
    
    # Sort by node index (row 0)
    new_order = new_edge_index[0].argsort()
    data.edge_index = new_edge_index[:, new_order]

    return data


def ExtractV2E(data):
    """
    Extracts the Node-to-Hyperedge (V2E) part of the edge_index.
    Assumes edge_index was concatenated as [V2E, E2V] or similar.
    It keeps edges where the source is a node (index < num_nodes).
    """
    edge_index = data.edge_index
    
    # Sort by source index
    _, sorted_idx = torch.sort(edge_index[0])
    edge_index = edge_index[:, sorted_idx].type(torch.LongTensor)

    num_nodes = data.n_x
    
    # Verify consistency
    max_idx = data.edge_index[0].max().item()
    if not ((data.n_x + data.num_hyperedges - 1) == max_idx):
        print(f'Warning: num_hyperedges check failed in ExtractV2E. Expected max idx {data.n_x + data.num_hyperedges - 1}, got {max_idx}')

    # Find split point where source index >= num_nodes (indicating E2V part)
    # If all source indices are nodes, this logic handles it gracefully or needs adjustment based on specific data format
    cidx_candidates = torch.where(edge_index[0] == num_nodes)[0]
    if len(cidx_candidates) > 0:
        cidx = cidx_candidates.min()
        data.edge_index = edge_index[:, :cidx].type(torch.LongTensor)
    else:
        # If no index == num_nodes, assuming all are V2E or indices are continuous
        pass 

    # print("inside ExtractV2E: ", data.edge_index.size())
    return data


def Add_Self_Loops(data):
    """
    Adds self-loops for nodes that do not already have one.
    A self-loop in a hypergraph is a hyperedge containing only that node.
    """
    edge_index = data.edge_index
    num_nodes = data.n_x
    num_hyperedges = data.num_hyperedges

    # Check consistency
    if not ((data.n_x + data.num_hyperedges - 1) == data.edge_index[1].max().item()):
        print('Warning: num_hyperedges check failed in Add_Self_Loops')

    # Identify nodes that are effectively self-loops (size 1 hyperedges)
    hyperedge_appear_fre = Counter(edge_index[1].numpy())
    skip_node_lst = []
    
    for edge, count in hyperedge_appear_fre.items():
        if count == 1:
            # Find the node associated with this single-node hyperedge
            idx = torch.where(edge_index[1] == edge)[0].item()
            skip_node = edge_index[0][idx]
            skip_node_lst.append(skip_node.item())

    # Create new self-loops for remaining nodes
    new_edge_idx = edge_index[1].max() + 1
    nodes_needing_loops = [i for i in range(num_nodes) if i not in skip_node_lst]
    
    if nodes_needing_loops:
        new_edges = torch.zeros((2, len(nodes_needing_loops)), dtype=edge_index.dtype, device=edge_index.device)
        new_edges[0] = torch.tensor(nodes_needing_loops, dtype=edge_index.dtype, device=edge_index.device)
        new_edges[1] = torch.arange(new_edge_idx, new_edge_idx + len(nodes_needing_loops), dtype=edge_index.dtype, device=edge_index.device)
        
        data.totedges = num_hyperedges + len(nodes_needing_loops)
        edge_index = torch.cat((edge_index, new_edges), dim=1)

    # Sort by node index
    _, sorted_idx = torch.sort(edge_index[0])
    data.edge_index = edge_index[:, sorted_idx].type(torch.LongTensor)
    
    return data


# =============================================================================
# Section 2: Matrix Construction
# =============================================================================

def get_HyperGCN_He_dict(data):
    """
    Constructs a dictionary mapping hyperedge IDs to the list of nodes they contain.
    Used for HyperGCN.
    """
    edge_index = np.array(data.edge_index)
    # Shift hyperedge indices to start from 0
    min_he_idx = edge_index[1, :].min()
    edge_index[1, :] = edge_index[1, :] - min_he_idx
    
    He_dict = {}
    for he in np.unique(edge_index[1, :]):
        nodes_in_he = list(edge_index[0, :][edge_index[1, :] == he])
        He_dict[he.item()] = nodes_in_he

    return He_dict


def ConstructH(data):
    """
    Constructs the Incidence Matrix H of size (num_nodes, num_hyperedges).
    H[v, e] = 1 if vertex v is in hyperedge e.
    """
    edge_index = np.array(data.edge_index)
    num_nodes = data.x.shape[0]
    num_hyperedges = np.max(edge_index[1]) - np.min(edge_index[1]) + 1
    
    H = np.zeros((num_nodes, num_hyperedges))
    
    # Mapping unique hyperedge IDs to 0...N-1
    unique_hes = np.unique(edge_index[1])
    for cur_idx, he in enumerate(unique_hes):
        nodes_in_he = edge_index[0][edge_index[1] == he]
        H[nodes_in_he, cur_idx] = 1.

    return H


def ConstructH_HNHN(data):
    """
    Constructs Incidence Matrix H for HNHN model.
    Similar to ConstructH but uses data.n_x and data.totedges for dimensions.
    """
    edge_index = np.array(data.edge_index)
    num_nodes = data.n_x
    num_hyperedges = int(data.totedges)
    
    H = np.zeros((num_nodes, num_hyperedges))
    
    unique_hes = np.unique(edge_index[1])
    for cur_idx, he in enumerate(unique_hes):
        nodes_in_he = edge_index[0][edge_index[1] == he]
        H[nodes_in_he, cur_idx] = 1.

    return H


def ConstructV2V(data):
    """
    Constructs a Vertex-to-Vertex (V2V) graph via clique expansion.
    Converts hyperedges into cliques (fully connected subgraphs).
    Counts edge weights based on how many hyperedges encompass a pair of nodes.
    """
    edge_index = np.array(data.edge_index)
    edge_weight_dict = {}

    for he in np.unique(edge_index[1, :]):
        nodes_in_he = np.sort(edge_index[0, :][edge_index[1, :] == he])
        if len(nodes_in_he) == 1:
            continue  # Skip self loops
        
        # Create cliques
        for comb in combinations(nodes_in_he, 2):
            if comb not in edge_weight_dict:
                edge_weight_dict[comb] = 1
            else:
                edge_weight_dict[comb] += 1

    # Convert dict to edge_index and norm (weights)
    num_edges = len(edge_weight_dict)
    new_edge_index = np.zeros((2, num_edges))
    new_norm = np.zeros((num_edges))
    
    for cur_idx, (edge, weight) in enumerate(edge_weight_dict.items()):
        new_edge_index[:, cur_idx] = edge
        new_norm[cur_idx] = weight

    data.edge_index = torch.tensor(new_edge_index).type(torch.LongTensor)
    data.norm = torch.tensor(new_norm).type(torch.FloatTensor)
    
    return data


# =============================================================================
# Section 3: Normalization & Graph Generation
# =============================================================================

def generate_G_from_H(data):
    """
    Generates the propagation matrix G for HGNN from incidence matrix H.
    G = D_v^(-1/2) * H * W * D_e^(-1) * H^T * D_v^(-1/2)
    Assumes data.edge_index is the dense incidence matrix H.
    """
    H = np.array(data.edge_index)
    n_edge = H.shape[1]
    
    # Weight of hyperedges (Uniform)
    W = np.ones(n_edge)
    
    # Degree matrices
    DV = np.sum(H * W, axis=1)
    DE = np.sum(H, axis=0)

    # Inverses
    invDE = np.diag(np.power(DE, -1))
    invDE[np.isinf(invDE)] = 0 # Handle division by zero
    
    DV2 = np.diag(np.power(DV, -0.5))
    DV2 = np.nan_to_num(DV2) # Handle isolated nodes

    W = np.diag(W)
    
    # G = DV2 * H * W * invDE * H.T * DV2
    HT = H.T
    G = DV2 @ H @ W @ invDE @ HT @ DV2
    
    data.edge_index = torch.Tensor(G)
    return data


def generate_G_for_HNHN(data, args):
    """
    Generates propagation matrices G_V2E and G_E2V for HNHN.
    """
    H = np.array(data.edge_index)
    alpha = args.HNHN_alpha
    beta = args.HNHN_beta

    DV = np.sum(H, axis=1)
    DE = np.sum(H, axis=0)

    # Avoid division by zero warnings
    with np.errstate(divide='ignore'):
        DE_pow_neg_beta = np.power(DE, -beta)
        DE_pow_neg_beta[np.isinf(DE_pow_neg_beta)] = 0
        
        DV_pow_beta = np.power(DV, beta)
        
        DV_pow_neg_alpha = np.power(DV, -alpha)
        DV_pow_neg_alpha[np.isinf(DV_pow_neg_alpha)] = 0
        
        DE_pow_alpha = np.power(DE, alpha)

    G_V2E = np.diag(DE_pow_neg_beta) @ H.T @ np.diag(DV_pow_beta)
    G_E2V = np.diag(DV_pow_neg_alpha) @ H @ np.diag(DE_pow_alpha)

    data.G_V2E = torch.Tensor(G_V2E)
    data.G_E2V = torch.Tensor(G_E2V)
    return data


def generate_norm_HNHN(H, data, args):
    """
    Calculates normalization terms for HNHN based on H.
    Populates data object with D_e_alpha, D_v_alpha_inv, etc.
    """
    alpha = args.HNHN_alpha
    beta = args.HNHN_beta
    H = np.array(H)

    # Standard degrees
    DV = np.sum(H, axis=1)
    DE = np.sum(H, axis=0)

    num_nodes = data.n_x
    num_hyperedges = int(data.totedges)

    # Calculate Alpha normalization terms
    D_e_alpha = DE ** alpha
    
    # Vectorized computation for D_v_alpha
    # Equivalent to sum(DE^alpha for edges connected to node v)
    # H is Node x Edge. H * DE^alpha broadcasts, then sum over edges (axis 1)
    D_v_alpha = np.sum(H * (DE ** alpha), axis=1)

    # Calculate Beta normalization terms
    D_v_beta = DV ** beta
    # Equivalent to sum(DV^beta for nodes in edge e)
    # H^T is Edge x Node. H^T * DV^beta broadcasts, then sum over nodes (axis 1)
    D_e_beta = np.sum(H.T * (DV ** beta), axis=1)

    # Inverses with safety check
    with np.errstate(divide='ignore'):
        D_v_alpha_inv = 1.0 / D_v_alpha
        D_v_alpha_inv[np.isinf(D_v_alpha_inv)] = 0

        D_e_beta_inv = 1.0 / D_e_beta
        D_e_beta_inv[np.isinf(D_e_beta_inv)] = 0

    data.D_e_alpha = torch.from_numpy(D_e_alpha).float()
    data.D_v_alpha_inv = torch.from_numpy(D_v_alpha_inv).float()
    data.D_v_beta = torch.from_numpy(D_v_beta).float()
    data.D_e_beta_inv = torch.from_numpy(D_e_beta_inv).float()

    return data


def norm_contruction(data, option='all_one', TYPE='V2E'):
    """
    Constructs normalization weights (data.norm) for the graph.
    """
    if TYPE == 'V2E':
        if option == 'all_one':
            data.norm = torch.ones_like(data.edge_index[0])

        elif option == 'deg_half_sym':
            edge_weight = torch.ones_like(data.edge_index[0])
            cidx = data.edge_index[1].min()
            
            Vdeg = scatter_add(edge_weight, data.edge_index[0], dim=0)
            HEdeg = scatter_add(edge_weight, data.edge_index[1] - cidx, dim=0)
            
            V_norm = Vdeg.pow(-0.5)
            V_norm[torch.isinf(V_norm)] = 0
            
            E_norm = HEdeg.pow(-0.5)
            E_norm[torch.isinf(E_norm)] = 0
            
            data.norm = V_norm[data.edge_index[0]] * E_norm[data.edge_index[1] - cidx]

    elif TYPE == 'V2V':
        data.edge_index, data.norm = gcn_norm(
            data.edge_index, data.norm, add_self_loops=True)
            
    return data


# =============================================================================
# Section 4: Dataset Splitting
# =============================================================================

def rand_train_test_idx(label, train_prop=.5, valid_prop=.25, ignore_negative=True, balance=False):
    """
    Randomly splits labels into train/valid/test sets.
    Adapted from https://github.com/CUAI/Non-Homophily-Benchmarks
    """
    if not balance:
        if ignore_negative:
            labeled_nodes = torch.where(label != -1)[0]
        else:
            labeled_nodes = torch.arange(label.shape[0])

        n = labeled_nodes.shape[0]
        train_num = int(n * train_prop)
        valid_num = int(n * valid_prop)

        perm = torch.randperm(n)

        train_indices = perm[:train_num]
        val_indices = perm[train_num:train_num + valid_num]
        test_indices = perm[train_num + valid_num:]

        if not ignore_negative:
            return train_indices, val_indices, test_indices

        split_idx = {
            'train': labeled_nodes[train_indices],
            'valid': labeled_nodes[val_indices],
            'test': labeled_nodes[test_indices]
        }
    else:
        # Balanced splitting: ensure equal number of samples per class in train
        indices = []
        for i in range(label.max() + 1):
            index = torch.where((label == i))[0]
            index = index[torch.randperm(index.size(0))]
            indices.append(index)

        percls_trn = int(train_prop / (label.max() + 1) * len(label))
        val_lb = int(valid_prop * len(label))
        
        train_idx = torch.cat([i[:percls_trn] for i in indices], dim=0)
        
        # Remaining indices for val/test
        rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]
        
        valid_idx = rest_index[:val_lb]
        test_idx = rest_index[val_lb:]
        
        split_idx = {
            'train': train_idx,
            'valid': valid_idx,
            'test': test_idx
        }
        
    return split_idx