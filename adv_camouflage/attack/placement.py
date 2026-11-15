#!/usr/bin/env python3
"""
AE5510: Differentiable Placement Module
=======================================
Module: adv_camouflage/attack/placement.py

Manages differentiable placement parameterization:
  theta = [tx, ty, log_s, phi, hx, hy]
  1. tx, ty: center offset normalized by target bounding dimension
  2. log_s: log-scale of patch
  3. phi: rotation angle in radians
  4. hx, hy: perspective shear components
  5. Multi-start initialization from DistanceAnalyzer top-K clearance centers
"""

from typing import Tuple, List, Dict, Optional
import math
import torch
import torch.nn as nn
import numpy as np

from adv_camouflage.geometry.homography import build_homography_matrix


class PatchPlacement(nn.Module):
    def __init__(
        self,
        init_center: Tuple[float, float],
        base_radius: float,
        canvas_hw: Tuple[int, int],
        target_box_wh: Tuple[float, float],
        initial_scale_fraction: float = 0.50,
    ):
        """
        init_center: (cx, cy) in absolute pixel coordinates
        base_radius: clearance radius R_max in pixels
        canvas_hw: (H, W) image dimensions
        target_box_wh: (w_box, h_box) in pixels
        initial_scale_fraction: fraction of clearance radius to initialize scale
        """
        super().__init__()
        H, W = canvas_hw
        self.H = H
        self.W = W
        self.canvas_hw = canvas_hw

        # Normalized coordinates in [-1, 1]
        cx_norm = (init_center[0] / W) * 2.0 - 1.0
        cy_norm = (init_center[1] / H) * 2.0 - 1.0
        self.init_center_norm = (float(cx_norm), float(cy_norm))

        # Box dimensions normalized to canvas [-1, 1] range (where full canvas width is 2.0)
        bw_norm = (target_box_wh[0] / W) * 2.0
        bh_norm = (target_box_wh[1] / H) * 2.0
        self.box_scale_norm = (float(bw_norm), float(bh_norm))

        # Initial scale relative to target box dimensions
        s0 = float(np.clip(initial_scale_fraction, 0.10, 0.80))
        log_s0 = math.log(s0)

        # Trainable parameters: [tx, ty, log_s, phi, hx, hy]
        init_theta = torch.tensor([0.0, 0.0, log_s0, 0.0, 0.0, 0.0], dtype=torch.float32)
        self.theta = nn.Parameter(init_theta)

    def get_homography(self) -> torch.Tensor:
        """Returns 3x3 forward homography matrix."""
        return build_homography_matrix(self.theta, self.init_center_norm, self.box_scale_norm)

    def get_patch_corners_canvas(self) -> torch.Tensor:
        """
        Computes the 4 corners of the patch in canvas pixel coordinates [0, W] x [0, H].
        Returns: (4, 2) tensor of corner coordinates [(x0, y0), (x1, y1), (x2, y2), (x3, y3)]
        """
        H_fwd = self.get_homography()  # (3, 3)

        # Canonical corner coordinates in [-1, 1]
        canonical_corners = torch.tensor([
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ], dtype=self.theta.dtype, device=self.theta.device).T  # (3, 4)

        # Project via forward homography
        proj = torch.matmul(H_fwd, canonical_corners)  # (3, 4)
        proj_norm_x = proj[0] / (proj[2] + 1e-8)  # in [-1, 1]
        proj_norm_y = proj[1] / (proj[2] + 1e-8)  # in [-1, 1]

        # Convert to canvas pixel coordinates
        canvas_x = (proj_norm_x + 1.0) * 0.5 * self.W
        canvas_y = (proj_norm_y + 1.0) * 0.5 * self.H

        return torch.stack([canvas_x, canvas_y], dim=1)  # (4, 2)


def create_multi_start_placements(
    dist_analysis: Dict,
    target_box_wh: Tuple[float, float],
    canvas_hw: Tuple[int, int],
    num_starts: int = 5,
) -> List[PatchPlacement]:
    """
    Creates candidate PatchPlacement modules initialized at the top-K distance transform clearance centers.
    """
    centers = dist_analysis.get("top_k_centers", [])
    r_max = dist_analysis.get("R_max", 10.0)

    if not centers:
        centers = [dist_analysis.get("centroid", (canvas_hw[1] // 2, canvas_hw[0] // 2))]

    placements = []
    for i, center in enumerate(centers[:num_starts]):
        placement = PatchPlacement(
            init_center=center,
            base_radius=r_max,
            canvas_hw=canvas_hw,
            target_box_wh=target_box_wh,
            initial_scale_fraction=0.45,
        )
        placements.append(placement)

    return placements
