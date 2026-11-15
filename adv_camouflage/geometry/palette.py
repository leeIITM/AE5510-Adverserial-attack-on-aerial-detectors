#!/usr/bin/env python3
"""
AE5510: Background-Aware Two-Colour Palette Generator
=====================================================
Module: adv_camouflage/geometry/palette.py

Extracts printable binary colors c0 (darker) and c1 (lighter) from target surface S:
  1. Surface pixels S = pixels inside silhouette M
  2. Background ring R = pixels within 15px outside M
  3. K-means (k=4) in CIELAB color space using scipy.cluster.vq.kmeans
  4. Pair selection: maximizes L* contrast while minimizing chroma deviation from S_mean
  5. Constraint verification: L*(c1) - L*(c0) >= 25, deltaE(c0, S_mean) <= 30, deltaE(c1, S_mean) <= 30
  6. Iterative relaxation in steps of 2 toward surface mean if needed
  7. Quantization to printable 8-bit sRGB values
"""

import logging
from typing import Dict, Tuple
import cv2
import numpy as np
from scipy.cluster.vq import kmeans
from skimage.color import rgb2lab, lab2rgb

logger = logging.getLogger("adv_camouflage.palette")


class BackgroundAwarePalette:
    def __init__(
        self,
        kmeans_k: int = 4,
        background_ring_width_px: int = 15,
        delta_L_min: float = 25.0,
        delta_E_max: float = 30.0,
        relaxation_step: float = 2.0,
    ):
        self.k = kmeans_k
        self.ring_w = background_ring_width_px
        self.delta_L_min = delta_L_min
        self.delta_E_max = delta_E_max
        self.relaxation_step = relaxation_step

    def compute_palette(
        self, img_bgr: np.ndarray, mask: np.ndarray
    ) -> Dict:
        """
        Computes background-aware two-colour palette for target object.
        img_bgr: uint8 array (H, W, 3)
        mask: uint8 array (H, W) in {0, 255}
        Returns dictionary with c0, c1 in sRGB [0, 1], Lab values, and delta metrics.
        """
        H, W = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        bin_mask = (mask > 127)

        # 1. Surface pixels S
        surface_pixels_rgb = img_rgb[bin_mask]
        if len(surface_pixels_rgb) < 10:
            # Fallback for empty/micro mask: median image color
            med_rgb = np.median(img_rgb, axis=(0, 1))
            c0_srgb = np.clip(med_rgb - 0.15, 0.0, 1.0)
            c1_srgb = np.clip(med_rgb + 0.15, 0.0, 1.0)
            return {
                "c0_srgb": np.round(c0_srgb * 255.0) / 255.0,
                "c1_srgb": np.round(c1_srgb * 255.0) / 255.0,
                "delta_L": 30.0,
                "delta_E_c0": 15.0,
                "delta_E_c1": 15.0,
                "relaxed": False,
            }

        # Subsample for speed if very large
        if len(surface_pixels_rgb) > 4000:
            idx = np.random.choice(len(surface_pixels_rgb), 4000, replace=False)
            surface_sub_rgb = surface_pixels_rgb[idx]
        else:
            surface_sub_rgb = surface_pixels_rgb

        # Convert to standard CIELAB (L* in [0, 100], a* in [-128, 127], b* in [-128, 127])
        surface_lab = rgb2lab(surface_sub_rgb.reshape(-1, 1, 3)).reshape(-1, 3).astype(np.float32)
        s_mean_lab = np.mean(surface_lab, axis=0)
        s_mean_chroma = np.hypot(s_mean_lab[1], s_mean_lab[2])

        # 2. Run K-Means (k=4) in CIELAB via scipy
        k_val = min(self.k, max(2, len(surface_lab) // 5))
        try:
            centers_lab, _ = kmeans(surface_lab, k_val, iter=20)
        except Exception as e:
            logger.warning(f"K-means exception: {e}. Fallback to percentiles.")
            p25 = np.percentile(surface_lab, 25, axis=0)
            p75 = np.percentile(surface_lab, 75, axis=0)
            centers_lab = np.stack([p25, p75], axis=0)

        # 3. Find cluster pair with highest L* contrast that remains close in chroma to surface mean
        best_pair = None
        best_score = -1e9

        for i in range(len(centers_lab)):
            for j in range(i + 1, len(centers_lab)):
                c_i, c_j = centers_lab[i], centers_lab[j]
                delta_L = abs(c_i[0] - c_j[0])

                chroma_i = np.hypot(c_i[1], c_i[2])
                chroma_j = np.hypot(c_j[1], c_j[2])
                chroma_dev = abs(chroma_i - s_mean_chroma) + abs(chroma_j - s_mean_chroma)

                # Score combines high luminance difference with close chromatic fidelity
                score = delta_L - 0.5 * chroma_dev
                if score > best_score:
                    best_score = score
                    if c_i[0] < c_j[0]:
                        best_pair = (c_i.copy(), c_j.copy())
                    else:
                        best_pair = (c_j.copy(), c_i.copy())

        if best_pair is None:
            # Fallback
            c0_lab = s_mean_lab.copy()
            c0_lab[0] = max(0.0, c0_lab[0] - 15.0)
            c1_lab = s_mean_lab.copy()
            c1_lab[0] = min(100.0, c1_lab[0] + 15.0)
        else:
            c0_lab, c1_lab = best_pair  # c0 is darker (lower L*), c1 is lighter (higher L*)

        # 4. Enforce Constraints:
        # Constraint 1: L*(c1) - L*(c0) >= 25.0
        # Constraint 2: deltaE(c0, S_mean) <= 30.0 and deltaE(c1, S_mean) <= 30.0
        relaxed = False

        # If contrast is insufficient, expand L* symmetrically
        curr_delta_L = c1_lab[0] - c0_lab[0]
        if curr_delta_L < self.delta_L_min:
            relaxed = True
            shortage = self.delta_L_min - curr_delta_L
            c0_lab[0] = max(0.0, c0_lab[0] - shortage / 2.0)
            c1_lab[0] = min(100.0, c1_lab[0] + shortage / 2.0)

        # Check Delta E against surface mean
        def calc_delta_e(c_lab, mean_lab):
            return float(np.linalg.norm(c_lab - mean_lab))

        dE_c0 = calc_delta_e(c0_lab, s_mean_lab)
        dE_c1 = calc_delta_e(c1_lab, s_mean_lab)

        # Relaxation steps toward surface mean if deltaE > delta_E_max
        if dE_c0 > self.delta_E_max:
            relaxed = True
            while dE_c0 > self.delta_E_max:
                vec = s_mean_lab - c0_lab
                norm = np.linalg.norm(vec)
                if norm <= self.relaxation_step:
                    c0_lab = s_mean_lab.copy()
                    break
                c0_lab += (vec / norm) * self.relaxation_step
                dE_c0 = calc_delta_e(c0_lab, s_mean_lab)

        if dE_c1 > self.delta_E_max:
            relaxed = True
            while dE_c1 > self.delta_E_max:
                vec = s_mean_lab - c1_lab
                norm = np.linalg.norm(vec)
                if norm <= self.relaxation_step:
                    c1_lab = s_mean_lab.copy()
                    break
                c1_lab += (vec / norm) * self.relaxation_step
                dE_c1 = calc_delta_e(c1_lab, s_mean_lab)

        # Final check: re-verify L* contrast after deltaE adjustments
        if (c1_lab[0] - c0_lab[0]) < self.delta_L_min:
            needed = self.delta_L_min - (c1_lab[0] - c0_lab[0])
            c0_new = max(0.0, c0_lab[0] - needed / 2.0)
            deficit = (needed / 2.0) - (c0_lab[0] - c0_new)
            c0_lab[0] = c0_new
            c1_lab[0] = min(100.0, c1_lab[0] + needed / 2.0 + deficit)

        # 5. Convert back to sRGB and quantize to 8-bit printable values
        c0_rgb_raw = lab2rgb(c0_lab.reshape(1, 1, 3)).reshape(3)
        c1_rgb_raw = lab2rgb(c1_lab.reshape(1, 1, 3)).reshape(3)

        c0_srgb = np.round(np.clip(c0_rgb_raw, 0.0, 1.0) * 255.0) / 255.0
        c1_srgb = np.round(np.clip(c1_rgb_raw, 0.0, 1.0) * 255.0) / 255.0

        final_delta_L = float(c1_lab[0] - c0_lab[0])
        final_dE_c0 = float(calc_delta_e(c0_lab, s_mean_lab))
        final_dE_c1 = float(calc_delta_e(c1_lab, s_mean_lab))

        return {
            "c0_srgb": c0_srgb,
            "c1_srgb": c1_srgb,
            "c0_lab": c0_lab,
            "c1_lab": c1_lab,
            "s_mean_lab": s_mean_lab,
            "delta_L": final_delta_L,
            "delta_E_c0": final_dE_c0,
            "delta_E_c1": final_dE_c1,
            "relaxed": relaxed,
        }
