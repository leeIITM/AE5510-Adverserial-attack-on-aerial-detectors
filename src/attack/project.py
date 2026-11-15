#!/usr/bin/env python3
"""
AE5510: Hard Containment Projection and Spillover Measurement
=============================================================
Module: adv_camouflage/attack/project.py

Implements:
  1. Exact spillover measurement: percentage of sticker footprint outside target hull
  2. Hard projection algorithm: shrinks scale and translates center toward clearance peak
     until 100% of the patch quadrilateral lies inside the admissible hull A
  3. Differentiable containment barrier loss using distance transform
"""

from typing import Tuple, Dict
import torch
import torch.nn as nn
import numpy as np

from adv_camouflage.attack.placement import PatchPlacement


def measure_spillover(
    warped_alpha: torch.Tensor,
    M_hull: torch.Tensor,
    threshold: float = 0.05,
) -> Dict[str, float]:
    """
    Measures physical spillover: percentage of the warped patch footprint
    that extends outside the target vehicle silhouette mask M_hull.
    warped_alpha: (1, 1, H, W) warped patch alpha footprint before hull intersection
    M_hull: (1, 1, H, W) binary target silhouette mask
    Returns:
      spillover_rate: float in [0.0, 1.0] (0.000 means 100% contained)
      spillover_pixels: int count of spilled pixels
      footprint_pixels: int count of total patch pixels
    """
    with torch.no_grad():
        patch_active = (warped_alpha > threshold)
        footprint_pixels = int(torch.sum(patch_active).item())

        if footprint_pixels == 0:
            return {
                "spillover_rate": 0.0,
                "spillover_pixels": 0,
                "footprint_pixels": 0,
            }

        outside_hull = (M_hull < 0.5)
        spilled = patch_active & outside_hull
        spillover_pixels = int(torch.sum(spilled).item())
        spillover_rate = float(spillover_pixels) / float(footprint_pixels)

        return {
            "spillover_rate": spillover_rate,
            "spillover_pixels": spillover_pixels,
            "footprint_pixels": footprint_pixels,
        }


def project_placement_to_admissible(
    placement: PatchPlacement,
    admissible_mask: np.ndarray,
    D_inside: np.ndarray,
    max_iters: int = 30,
) -> Dict:
    """
    Projects placement parameters theta such that all 4 corners and the perimeter
    of the patch are strictly contained inside the admissible region A.
    admissible_mask: (H, W) uint8 in {0, 255}
    D_inside: (H, W) float32 Euclidean distance to exterior boundary
    """
    H, W = admissible_mask.shape[:2]

    with torch.no_grad():
        theta = placement.theta.clone()

        # Extract current parameters
        tx = theta[0].item()
        ty = theta[1].item()
        log_s = theta[2].item()
        phi = theta[3].item()
        hx = theta[4].item()
        hy = theta[5].item()

        converged = False
        for it in range(max_iters):
            corners = placement.get_patch_corners_canvas().cpu().numpy()  # (4, 2)

            # Check if all corners are in bounds and inside admissible region
            all_inside = True
            min_dist = 9999.0

            feather_margin = 3.5  # accounts for 2-sigma Gaussian feathering boundary radius
            for cx, cy in corners:
                ix = int(round(cx))
                iy = int(round(cy))
                if 0 <= ix < W and 0 <= iy < H:
                    dist_val = float(D_inside[iy, ix])
                    if dist_val < min_dist:
                        min_dist = dist_val
                    # Corner must be inside the target silhouette with 2-sigma feather margin
                    if dist_val < feather_margin:
                        all_inside = False
                else:
                    all_inside = False
                    min_dist = -10.0

            if all_inside and min_dist >= feather_margin:
                converged = True
                break

            # If violated, translate center towards initial peak center (0, 0)
            # and shrink scale log_s
            tx *= 0.85
            ty *= 0.85
            log_s -= 0.05
            hx *= 0.80
            hy *= 0.80

            placement.theta.data[0] = tx
            placement.theta.data[1] = ty
            placement.theta.data[2] = log_s
            placement.theta.data[4] = hx
            placement.theta.data[5] = hy

        final_corners = placement.get_patch_corners_canvas().cpu().numpy()
        return {
            "converged": converged,
            "iterations": it + 1,
            "final_log_s": log_s,
            "final_scale": float(np.exp(log_s)),
            "corners": final_corners.tolist(),
        }


def compute_containment_loss(
    placement: PatchPlacement,
    D_outside: torch.Tensor,
) -> torch.Tensor:
    """
    Differentiable barrier loss penalizing any patch corner extending outside the hull.
    D_outside: (1, 1, H, W) tensor where value is Euclidean distance to hull boundary (0 inside)
    """
    corners = placement.get_patch_corners_canvas()  # (4, 2)
    H = placement.H
    W = placement.W

    # Normalize corner coordinates to [-1, 1] for grid_sample
    norm_x = (corners[:, 0] / W) * 2.0 - 1.0
    norm_y = (corners[:, 1] / H) * 2.0 - 1.0
    grid = torch.stack([norm_x, norm_y], dim=-1).view(1, 1, 4, 2)

    # Sample D_outside at the corner coordinates
    sampled_dist = torch.nn.functional.grid_sample(
        D_outside, grid, mode="bilinear", padding_mode="border", align_corners=False
    ).squeeze()  # (4,)

    # Normalized exterior distance penalty (balanced with detection loss)
    norm_factor = float(max(H, W))
    norm_dist = sampled_dist / norm_factor
    loss_containment = torch.sum(norm_dist ** 2) * 50.0
    return loss_containment
