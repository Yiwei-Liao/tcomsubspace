# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Orthogonal(nn.Module):
    """
    Parametrizes orthogonal matrices using various maps:
    - Matrix Exponential (matrix_exp)
    - Cayley Map (cayley)
    - Householder Reflections (householder)
    - Euler Angles (euler, restricted to d=2 or d=3)
    
    Reference: https://pytorch.org/docs/stable/_modules/torch/nn/utils/parametrizations.html#orthogonal
    """
    
    def __init__(self, d: int, orthogonal_map: str):
        super().__init__()
        valid_maps = ["matrix_exp", "cayley", "householder", "euler"]
        if orthogonal_map not in valid_maps:
            raise ValueError(f"orthogonal_map must be one of {valid_maps}, got '{orthogonal_map}'")
        
        self.d = d
        self.orthogonal_map = orthogonal_map

    @staticmethod
    def _get_2d_rotation(params: torch.Tensor) -> torch.Tensor:
        """Generates 2D rotation matrices from parameters."""
        # params: [batch_size, 1]
        assert params.size(-1) == 1
        angle = params * 2 * math.pi
        sin = torch.sin(angle)
        cos = torch.cos(angle)
        # Construct [[cos, -sin], [sin, cos]]
        return torch.cat([cos, -sin, sin, cos], dim=1).view(-1, 2, 2)

    @staticmethod
    def _get_3d_rotation(params: torch.Tensor) -> torch.Tensor:
        """Generates 3D rotation matrices from Euler angles."""
        # params: [batch_size, 3]
        assert params.size(-1) == 3

        alpha = params[:, 0].view(-1, 1) * 2 * math.pi
        beta  = params[:, 1].view(-1, 1) * 2 * math.pi
        gamma = params[:, 2].view(-1, 1) * 2 * math.pi

        sin_a, cos_a = torch.sin(alpha), torch.cos(alpha)
        sin_b, cos_b = torch.sin(beta),  torch.cos(beta)
        sin_g, cos_g = torch.sin(gamma), torch.cos(gamma)

        # R = R_z(alpha) * R_y(beta) * R_x(gamma) or similar convention
        row1 = torch.cat([cos_a*cos_b, cos_a*sin_b*sin_g - sin_a*cos_g, cos_a*sin_b*cos_g + sin_a*sin_g], dim=1)
        row2 = torch.cat([sin_a*cos_b, sin_a*sin_b*sin_g + cos_a*cos_g, sin_a*sin_b*cos_g - cos_a*sin_g], dim=1)
        row3 = torch.cat([-sin_b,      cos_b*sin_g,                     cos_b*cos_g],                     dim=1)

        return torch.cat([row1, row2, row3], dim=1).view(-1, 3, 3)

    def _params_to_matrix(self, params: torch.Tensor) -> torch.Tensor:
        """Constructs a lower triangular matrix (or similar) from flat parameters."""
        offset = -1 if self.orthogonal_map == 'householder' else 0
        
        # Create indices for the lower triangle
        tril_indices = torch.tril_indices(row=self.d, col=self.d, offset=offset, device=params.device)
        
        # Fill the matrix
        matrix = torch.zeros((params.size(0), self.d, self.d), dtype=params.dtype, device=params.device)
        matrix[:, tril_indices[0], tril_indices[1]] = params
        
        return matrix

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        # 1. Handle Euler Angles (Direct construction, no matrix filling needed)
        if self.orthogonal_map == 'euler':
            assert 2 <= self.d <= 3, "Euler map is only implemented for d=2 or d=3"
            if self.d == 2:
                return self._get_2d_rotation(params)
            else:
                return self._get_3d_rotation(params)

        # 2. Construct base matrix A from parameters for other methods
        A_raw = self._params_to_matrix(params)

        # 3. Apply Orthogonal Map
        if self.orthogonal_map == "matrix_exp":
            # Make A skew-symmetric: A - A^T
            # params was strictly lower tril, so A is strictly lower tril.
            # A - A.T creates a full skew-symmetric matrix.
            A_skew = A_raw - A_raw.transpose(-2, -1)
            Q = torch.matrix_exp(A_skew)

        elif self.orthogonal_map == "cayley":
            # Make A skew-symmetric
            A_skew = A_raw - A_raw.transpose(-2, -1)
            # Cayley retraction: (I + A/2)(I - A/2)^-1
            Id = torch.eye(self.d, dtype=A_skew.dtype, device=A_skew.device)
            Q = torch.linalg.solve(
                torch.add(Id, A_skew, alpha=-0.5), 
                torch.add(Id, A_skew, alpha=0.5)
            )

        elif self.orthogonal_map == 'householder':
            # Construct Householder vectors
            # A has 0 on diagonal and upper triangle due to offset=-1 in _params_to_matrix
            eye = torch.eye(self.d, device=params.device).unsqueeze(0)
            A_house = A_raw + eye # Add identity to get the Householder vectors
            
            # Use QR decomposition to orthogonalize based on these vectors
            # This replaces the missing 'torch_householder_orgqr' dependency
            # while maintaining the mathematical goal of generating orthogonal Q from input vectors.
            Q, _ = torch.linalg.qr(A_house) 
            
        else:
            raise ValueError(f"Unsupported transformation {self.orthogonal_map}")

        return Q