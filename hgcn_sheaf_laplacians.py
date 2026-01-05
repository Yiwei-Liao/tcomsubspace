#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
sheaf Laplacian Construction Module.
Contains functions to construct the non-linear sheaf Laplacian for sheafHyperGCN.
"""

import torch
import numpy as np
import itertools
from torch_scatter import scatter_add

# =============================================================================
# Graph Reduction Logic
# =============================================================================

def reduce_graph(X_reduced, m, d, edge_index):
    """
    Constructs the graph structure for the sheaf Laplacian.
    
    Projects hyperedge features to 1D (via random projection in feature space),
    then finds the 'Supremum' (max) and 'Infimum' (min) nodes in the projected
    sheaf feature space to create directed edges.

    Args:
        X_reduced: (nnz, d, f) - Reduced sheaf features F_v<e(X_v)
        m: Boolean - Whether to use mediators (star expansion)
        d: Int - sheaf stalk dimension
        edge_index: (2, nnz) - Sparse incidence matrix indices [node_idx, he_idx]

    Returns:
        edges_idx: Indices in X_reduced for off-diagonal edges (source, dest)
        edges_idx_diag: Indices in X_reduced for diagonal calculation
        all_contained_hyperedges: Helper indices for scatter aggregation
        hgcn_edges: The constructed graph edges (node_u, node_v)
    """
    device = X_reduced.device
    
    # 1. Random Projection: Project feature dim (f) -> scalar
    # Result: nnz x d
    rv = torch.rand(X_reduced.shape[-1], device=device).unsqueeze(-1)
    X_hedge = X_reduced.reshape(-1, X_reduced.shape[-1]) # (nnz*d) x f
    p = X_hedge @ rv 
    p = p.view(-1, d) # nnz x d
    
    # Move to CPU for grouping logic (easier handling of variable-sized hyperedges)
    p_np = p.detach().cpu().numpy().T # d x nnz
    row, col = edge_index
    node_indices = row.detach().cpu().numpy()
    he_indices = col.detach().cpu().numpy()
    nnz_indices = np.arange(len(row))

    # ---------------------------------------------------------
    # Part A: Off-Diagonal Edges (Expansion)
    # ---------------------------------------------------------
    # Sort by hyperedge index to group nodes belonging to the same hyperedge
    sort_mask = np.argsort(he_indices)
    sorted_node_indices = node_indices[sort_mask]
    sorted_nnz_indices = nnz_indices[sort_mask]
    sorted_he_indices = he_indices[sort_mask]

    # Split into groups (hyperedges)
    _, split_indices = np.unique(sorted_he_indices, return_index=True)
    he_groups_nodes = np.split(sorted_node_indices, split_indices[1:])
    he_groups_nnz = np.split(sorted_nnz_indices, split_indices[1:])

    edges_list = []
    edges_idx_list = []

    for hyperedge_pos, hyperedge_nodes in zip(he_groups_nnz, he_groups_nodes):
        # hyperedge_pos: indices in the nnz vector
        # hyperedge_nodes: actual node IDs
        
        # Get projected features for this group: d x size_of_he
        p_current = p_np[:, hyperedge_pos]
        
        # Compute pairwise distance in d-dim space: d x k x k
        diff = p_current[:, :, None] - p_current[:, None, :]
        dists = np.linalg.norm(diff, axis=0, ord=2) # k x k
        
        # Find pair with max distance
        s_local, i_local = np.unravel_index(np.argmax(dists), dists.shape)
        
        Se_node, Ie_node = hyperedge_nodes[s_local], hyperedge_nodes[i_local]
        Se_nnz, Ie_nnz = hyperedge_pos[s_local], hyperedge_pos[i_local]
        
        # Add Edge: Supremum <-> Infimum
        edges_list.extend([[Se_node, Ie_node], [Ie_node, Se_node]])
        edges_idx_list.extend([[Se_nnz, Ie_nnz], [Ie_nnz, Se_nnz]])
        
        # Add Mediators if enabled
        if m:
            # Mediators are all nodes except Se and Ie
            mask = np.ones(len(hyperedge_nodes), dtype=bool)
            mask[s_local] = False
            mask[i_local] = False
            
            med_nodes = hyperedge_nodes[mask]
            med_nnz = hyperedge_pos[mask]
            
            for mn, mnnz in zip(med_nodes, med_nnz):
                # Connect Se/Ie to Mediator
                edges_list.extend([[Se_node, mn], [mn, Se_node], [Ie_node, mn], [mn, Ie_node]])
                edges_idx_list.extend([[Se_nnz, mnnz], [mnnz, Se_nnz], [Ie_nnz, mnnz], [mnnz, Ie_nnz]])

    # ---------------------------------------------------------
    # Part B: Diagonal Self-Loops
    # ---------------------------------------------------------
    # We need to aggregate \sum_e F_v<e^T F_v<e for each node v across all hyperedges it belongs to.
    
    # Sort by node index to group by node
    node_sort_mask = np.argsort(node_indices)
    sorted_nnz_by_node = nnz_indices[node_sort_mask]
    sorted_nodes_by_node = node_indices[node_sort_mask]
    
    unique_nodes, node_split_indices = np.unique(sorted_nodes_by_node, return_index=True)
    node_groups_nnz = np.split(sorted_nnz_by_node, node_split_indices[1:])
    
    diag_edges_list = []
    diag_idx_list = []
    all_contained_he_list = []
    
    # Iterate over each unique node
    for i, group_nnz in enumerate(node_groups_nnz):
        curr_node = unique_nodes[i]
        
        # Add ONE self-loop edge for the final graph per unique node
        diag_edges_list.append([curr_node, curr_node])
        
        # Add ALL instances of this node in hyperedges for aggregation
        # We need (idx, idx) pairs for diagonal multiplication F^T F
        diag_idx_list.extend(zip(group_nnz, group_nnz))
        
        # Helper index to scatter_add results to the correct node index 'i'
        all_contained_he_list.extend([i] * len(group_nnz))

    # Combine lists
    if diag_edges_list:
        edges_list.extend(diag_edges_list)

    # Convert to Tensors
    edges_idx = torch.tensor(np.array(edges_idx_list).T, device=device, dtype=torch.long)
    edges_idx_diag = torch.tensor(np.array(diag_idx_list).T, device=device, dtype=torch.long)
    all_contained_hyperedges = torch.tensor(np.array(all_contained_he_list), device=device, dtype=torch.long)
    hgcn_edges = torch.tensor(np.array(edges_list).T, device=device, dtype=torch.long)
    
    return edges_idx, edges_idx_diag, all_contained_hyperedges, hgcn_edges


# =============================================================================
# Laplacian Builders
# =============================================================================

def sheafLaplacianDiag(H, m, d, edge_index, sheaf, E=None):
    """
    Approximates the Laplacian with Diagonal sheaf.
    """
    F = sheaf
    num_nodes = H.shape[0] // d
    MLP_hidden = H.shape[-1]

    # 1. Apply Reduction: F_v<e(X_v)
    H_selected = H.view((num_nodes, d, -1))
    H_selected = torch.index_select(H_selected, dim=0, index=edge_index[0]) # Select nodes
    X_reduced = H_selected.permute(0, 2, 1) # nnz x f x d
    
    # Diagonal embedding: nnz x d -> nnz x d x d
    sheaf_mat = torch.diag_embed(sheaf, dim1=-2, dim2=-1) 
    sheaf_mat = sheaf_mat.unsqueeze(1).repeat(1, MLP_hidden, 1, 1).reshape(-1, d, d)

    # Matrix Mult: F @ X
    X_reduced = X_reduced.reshape(-1, d).unsqueeze(-1)
    X_reduced = torch.bmm(sheaf_mat, X_reduced)
    X_reduced = X_reduced.reshape(-1, MLP_hidden, d).permute(0, 2, 1) # nnz x d x f

    # 2. Build Graph Topology
    edges_idx, edges_idx_diag, all_contained_hyperedges, hgcn_edges = reduce_graph(X_reduced, m, d, edge_index) 

    # 3. Compute Laplacian Attributes
    # Off-diagonal: -F_v<e^T F_w<e
    F_source = torch.index_select(F, dim=0, index=edges_idx[0]) 
    F_dest = torch.index_select(F, dim=0, index=edges_idx[1])
    attributes = -F_source * F_dest 

    # Diagonal: sum_e F_v<e^T F_v<e
    F_source_diag = torch.index_select(F, dim=0, index=edges_idx_diag[0]) 
    F_dest_diag = torch.index_select(F, dim=0, index=edges_idx_diag[1])
    attributes_diag = F_source_diag * F_dest_diag
    
    # Aggregate self-loops
    attributes_diag = scatter_add(attributes_diag, all_contained_hyperedges, dim=0) 
    
    # Combine
    attributes = torch.concat([attributes, attributes_diag], axis=0)

    # 4. Expand indices to block matrix format
    d_range = torch.arange(d, device=H.device).view(1, -1, 1).repeat(2, 1, 1)
    hgcn_edges = hgcn_edges.unsqueeze(1) # 2 x 1 x K
    hgcn_edges = d * hgcn_edges + d_range 
    h_sheaf_index = hgcn_edges.permute(0, 2, 1).reshape(2, -1)
    h_sheaf_attributes = attributes.view(-1)

    return h_sheaf_index, h_sheaf_attributes


def sheafLaplacianGeneral(H, m, d, edge_index, sheaf, E=None):
    """
    Approximates the Laplacian with General (Full) sheaf.
    """
    F = sheaf # nnz x (d*d)
    num_nodes = H.shape[0] // d
    MLP_hidden = H.shape[-1]

    # 1. Apply Reduction
    H_selected = H.view((num_nodes, d, -1))
    H_selected = torch.index_select(H_selected, dim=0, index=edge_index[0])
    X_reduced = H_selected.permute(0, 2, 1).contiguous() # nnz x f x d
    
    # Reshape sheaf to matrix
    sheaf_mat = sheaf.view(sheaf.shape[0], d, d)
    sheaf_mat = sheaf_mat.unsqueeze(1).repeat(1, MLP_hidden, 1, 1).view(-1, d, d)
    
    X_reduced = X_reduced.view(-1, d).unsqueeze(-1)
    X_reduced = torch.bmm(sheaf_mat, X_reduced)
    X_reduced = X_reduced.reshape(-1, MLP_hidden, d).permute(0, 2, 1)

    # 2. Build Graph
    edges_idx, edges_idx_diag, all_contained_hyperedges, hgcn_edges = reduce_graph(X_reduced, m, d, edge_index)

    # 3. Compute Attributes (Matrix Multiplication for General Case)
    F = F.view(F.shape[0], d, d)
    F_source = torch.index_select(F, dim=0, index=edges_idx[0])
    F_dest = torch.index_select(F, dim=0, index=edges_idx[1])
    
    # -F^T @ F
    attributes = -1 * torch.bmm(F_source.transpose(1, 2), F_dest)

    # Diagonal
    F_source_diag = torch.index_select(F, dim=0, index=edges_idx_diag[0]) 
    F_dest_diag = torch.index_select(F, dim=0, index=edges_idx_diag[1])
    attributes_diag = torch.bmm(F_source_diag.transpose(1, 2), F_dest_diag)

    attributes_diag = scatter_add(attributes_diag, all_contained_hyperedges, dim=0)
    attributes = torch.concat([attributes, attributes_diag], axis=0)

    # 4. Expand Indices
    d_range = torch.arange(d, device=H.device)
    d_range_edges = d_range.repeat(d).view(-1, 1)
    d_range_nodes = d_range.repeat_interleave(d).view(-1, 1)
    
    hgcn_edges = hgcn_edges.unsqueeze(1) 
    idx_0 = d * hgcn_edges[0] + d_range_nodes
    idx_1 = d * hgcn_edges[1] + d_range_edges
    
    idx_0 = idx_0.permute((1, 0)).reshape(1, -1)
    idx_1 = idx_1.permute((1, 0)).reshape(1, -1)
    h_sheaf_index = torch.concat((idx_0, idx_1), 0)
    
    h_sheaf_attributes = attributes.view(-1)
    return h_sheaf_index, h_sheaf_attributes


def sheafLaplacianOrtho(H, m, d, edge_index, sheaf, E=None):
    """
    Approximates the Laplacian with Orthogonal sheaf.
    """
    F = sheaf
    num_nodes = H.shape[0] // d
    MLP_hidden = H.shape[-1]

    # 1. Apply Reduction
    H_selected = H.view((num_nodes, d, -1))
    H_selected = torch.index_select(H_selected, dim=0, index=edge_index[0])
    X_reduced = H_selected.permute(0, 2, 1).contiguous()
    
    sheaf_mat = sheaf.view(sheaf.shape[0], d, d)
    sheaf_mat = sheaf_mat.unsqueeze(1).repeat(1, MLP_hidden, 1, 1).view(-1, d, d)
    
    X_reduced = X_reduced.view(-1, d).unsqueeze(-1)
    X_reduced = torch.bmm(sheaf_mat, X_reduced)
    X_reduced = X_reduced.reshape(-1, MLP_hidden, d).permute(0, 2, 1)

    # 2. Build Graph
    edges_idx, edges_idx_diag, all_contained_hyperedges, hgcn_edges = reduce_graph(X_reduced, m, d, edge_index)

    # 3. Compute Attributes
    F = F.view(F.shape[0], d, d)
    F_source = torch.index_select(F, dim=0, index=edges_idx[0])
    F_dest = torch.index_select(F, dim=0, index=edges_idx[1])
    
    attributes = -1 * torch.bmm(F_source.transpose(1, 2), F_dest)

    # Diagonal: For Orthogonal matrices, F^T @ F = I
    # We sum Identity matrices for each hyperedge incidence
    attributes_diag = torch.eye(d).unsqueeze(0).repeat((edges_idx_diag.shape[1], 1, 1)).to(H.device)
    attributes_diag = scatter_add(attributes_diag, all_contained_hyperedges, dim=0)
    
    attributes = torch.concat([attributes, attributes_diag], axis=0)

    # 4. Expand Indices (Same as General)
    d_range = torch.arange(d, device=H.device)
    d_range_edges = d_range.repeat(d).view(-1, 1)
    d_range_nodes = d_range.repeat_interleave(d).view(-1, 1)
    
    hgcn_edges = hgcn_edges.unsqueeze(1) 
    idx_0 = d * hgcn_edges[0] + d_range_nodes
    idx_1 = d * hgcn_edges[1] + d_range_edges
    
    idx_0 = idx_0.permute((1, 0)).reshape(1, -1)
    idx_1 = idx_1.permute((1, 0)).reshape(1, -1)
    h_sheaf_index = torch.concat((idx_0, idx_1), 0)
    
    h_sheaf_attributes = attributes.view(-1)
    return h_sheaf_index, h_sheaf_attributes