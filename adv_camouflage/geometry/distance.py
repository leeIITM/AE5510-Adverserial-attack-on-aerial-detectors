#!/usr/bin/env python3
"""
AE5510: Distance Transform and Admissible Region Computation
============================================================
Module: adv_camouflage/geometry/distance.py

Computes:
  1. Euclidean distance transform D inside silhouette mask M
  2. Inscribed radius peak clearance R_max = max(D) and maximum location
  3. Admissible region A = { (x, y) in M : D(x, y) >= delta }, delta = max(2.5 px, 0.10 * R_max)
  4. Top K local maxima of D inside M for multi-start optimization initialization
  5. Outside distance transform D_outside for differentiable containment barrier loss
"""

from typing import Dict, List, Tuple
import cv2
import numpy as np
from scipy.ndimage import maximum_filter


class DistanceAnalyzer:
    def __init__(self, delta_min_px: float = 2.5, delta_fraction: float = 0.10):
        self.delta_min_px = delta_min_px
        self.delta_fraction = delta_fraction

    def analyze_silhouette(self, mask: np.ndarray, top_k: int = 5) -> Dict:
        """
        Computes distance transforms and admissible region.
        mask: uint8 array of shape (H, W) in {0, 255}
        """
        H, W = mask.shape[:2]
        bin_mask = (mask > 127).astype(np.uint8)

        if np.sum(bin_mask) == 0:
            # Empty mask fallback
            return {
                "D_inside": np.zeros((H, W), dtype=np.float32),
                "D_outside": np.zeros((H, W), dtype=np.float32),
                "R_max": 0.0,
                "delta": self.delta_min_px,
                "centroid": (W // 2, H // 2),
                "top_k_centers": [(W // 2, H // 2)],
                "admissible_mask": np.zeros((H, W), dtype=np.uint8),
            }

        # 1. Euclidean Distance Transform inside M (distance to nearest background pixel)
        D_inside = cv2.distanceTransform(bin_mask, cv2.DIST_L2, 5)

        # Inscribed peak clearance
        min_val, max_r, min_loc, max_loc = cv2.minMaxLoc(D_inside)
        R_max = float(max_r)
        peak_cx, peak_cy = max_loc

        # 2. Admissible boundary threshold delta
        delta = max(self.delta_min_px, self.delta_fraction * R_max)

        # 3. Admissible Region Mask A = { (x, y) in M : D(x, y) >= delta }
        admissible_mask = (D_inside >= delta).astype(np.uint8) * 255
        if np.sum(admissible_mask == 255) < 10:
            # If delta is too strict for micro-objects, relax to non-zero distance
            admissible_mask = (D_inside >= max(1.0, 0.5 * delta)).astype(np.uint8) * 255

        # 4. Outside Distance Transform (distance to nearest inside vehicle pixel)
        inv_mask = cv2.bitwise_not(bin_mask * 255)
        D_outside = cv2.distanceTransform(inv_mask, cv2.DIST_L2, 5)

        # 5. Extract top K local maxima of D inside M
        local_max = maximum_filter(D_inside, size=7) == D_inside
        peaks_mask = local_max & (D_inside >= delta)
        peak_ys, peak_xs = np.where(peaks_mask)

        candidates = []
        for py, px in zip(peak_ys, peak_xs):
            candidates.append((float(D_inside[py, px]), int(px), int(py)))

        # Sort descending by distance clearance
        candidates.sort(key=lambda item: item[0], reverse=True)

        top_k_centers = []
        min_dist_between_peaks = max(4.0, R_max * 0.25)

        for score, px, py in candidates:
            # Check spatial separation from already selected peaks
            is_separated = True
            for _, ex, ey in top_k_centers:
                dist = np.hypot(px - ex, py - ey)
                if dist < min_dist_between_peaks:
                    is_separated = False
                    break
            if is_separated:
                top_k_centers.append((score, px, py))
                if len(top_k_centers) >= top_k:
                    break

        if not top_k_centers:
            top_k_centers = [(R_max, peak_cx, peak_cy)]

        top_k_pts = [(px, py) for _, px, py in top_k_centers]

        return {
            "D_inside": D_inside,
            "D_outside": D_outside,
            "R_max": R_max,
            "delta": delta,
            "centroid": (peak_cx, peak_cy),
            "top_k_centers": top_k_pts,
            "admissible_mask": admissible_mask,
        }
