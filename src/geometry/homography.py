#!/usr/bin/env python3
"""
AE5510: Differentiable Homography and Perspective Warping Utilities
===================================================================
Module: adv_camouflage/geometry/homography.py

Implements pure-PyTorch differentiable perspective warping:
  1. Parameterizes theta = (tx, ty, log_s, phi, hx, hy)
  2. Constructs 3x3 forward and inverse homography matrices H(theta)
  3. Differentiable grid sampling via F.grid_sample without native DLL dependencies
  4. Returns warped RGB patch and warped alpha mask
"""

import math
from typing import Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_homography_matrix(
    theta: torch.Tensor,
    init_center_norm: Tuple[float, float],
    box_scale_norm: Tuple[float, float],
) -> torch.Tensor:
    """
    Constructs 3x3 forward homography matrix mapping canonical patch [-1, 1]^2
    to normalized image coordinates [-1, 1]^2.
    theta: (6,) tensor [tx, ty, log_s, phi, hx, hy]
    init_center_norm: (c_norm_x, c_norm_y) in [-1, 1]
    box_scale_norm: (bw_norm, bh_norm) in [0, 2]
    """
    tx, ty, log_s, phi, hx, hy = theta[0], theta[1], theta[2], theta[3], theta[4], theta[5]

    # Constrain shear to [-0.3, 0.3]
    hx_c = torch.clamp(hx, -0.30, 0.30)
    hy_c = torch.clamp(hy, -0.30, 0.30)

    # Scale calibrated to target object bounding dimensions
    s = torch.exp(torch.clamp(log_s, -4.0, 1.0))
    sx = max(float(box_scale_norm[0]), 1e-4) * s
    sy = max(float(box_scale_norm[1]), 1e-4) * s

    # Center translation
    cx = init_center_norm[0] + tx * (box_scale_norm[0] * 0.5)
    cy = init_center_norm[1] + ty * (box_scale_norm[1] * 0.5)

    cos_p = torch.cos(phi)
    sin_p = torch.sin(phi)

    # Scale Matrix S
    S = torch.zeros((3, 3), device=theta.device, dtype=theta.dtype)
    S[0, 0] = sx
    S[1, 1] = sy
    S[2, 2] = 1.0

    # Perspective Shear Matrix H_persp
    H_p = torch.eye(3, device=theta.device, dtype=theta.dtype)
    H_p[0, 1] = hx_c
    H_p[1, 0] = hy_c

    # Rotation Matrix R
    R = torch.eye(3, device=theta.device, dtype=theta.dtype)
    R[0, 0] = cos_p
    R[0, 1] = -sin_p
    R[1, 0] = sin_p
    R[1, 1] = cos_p

    # Translation Matrix T
    T = torch.eye(3, device=theta.device, dtype=theta.dtype)
    T[0, 2] = cx
    T[1, 2] = cy

    # Forward Homography H_fwd = T * R * H_p * S
    H_fwd = torch.matmul(T, torch.matmul(R, torch.matmul(H_p, S)))
    return H_fwd


def warp_patch_to_canvas(
    patch_rgb: torch.Tensor,
    patch_alpha: torch.Tensor,
    H_fwd: torch.Tensor,
    out_shape: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Warps canonical patch and alpha mask onto the full canvas.
    patch_rgb: (1, 3, h_p, w_p)
    patch_alpha: (1, 1, h_p, w_p)
    H_fwd: (3, 3) forward homography matrix
    out_shape: (H, W) canvas shape
    Returns:
      warped_rgb: (1, 3, H, W)
      warped_alpha: (1, 1, H, W)
    """
    H, W = out_shape
    device = patch_rgb.device
    dtype = patch_rgb.dtype

    # Invert forward homography to sample output image grid from canonical patch coordinates
    try:
        H_inv = torch.linalg.inv(H_fwd)
    except Exception:
        try:
            H_inv = torch.linalg.inv(H_fwd + 1e-5 * torch.eye(3, device=device, dtype=dtype))
        except Exception:
            H_inv = torch.linalg.pinv(H_fwd)

    # Generate normalized meshgrid for output canvas [-1, 1]
    y_coords = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
    x_coords = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")

    # Reshape to homogeneous coordinates (3, H*W)
    ones = torch.ones_like(grid_x)
    xy1 = torch.stack([grid_x, grid_y, ones], dim=0).view(3, -1)  # (3, N)

    # Multiply by inverse homography: P_patch = H_inv * P_canvas
    src_xyw = torch.matmul(H_inv, xy1)  # (3, N)
    src_x = src_xyw[0] / (src_xyw[2] + 1e-8)
    src_y = src_xyw[1] / (src_xyw[2] + 1e-8)

    # Reshape back to (1, H, W, 2) grid required by F.grid_sample
    grid = torch.stack([src_x, src_y], dim=-1).view(1, H, W, 2)

    warped_rgb = F.grid_sample(
        patch_rgb, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    warped_alpha = F.grid_sample(
        patch_alpha, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )

    return warped_rgb, warped_alpha
