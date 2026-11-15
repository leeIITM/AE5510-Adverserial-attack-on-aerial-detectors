#!/usr/bin/env python3
"""
AE5510: Joint Adversarial Camouflage Optimizer & Constrained Size Search
========================================================================
Module: adv_camouflage/attack/optimize.py

Implements:
  1. Joint optimization of placement theta and binary pattern logits z in one loop
  2. Multi-start initialization from DistanceAnalyzer peak clearance centers
  3. Straight-Through Estimator (STE) temperature annealing
  4. Physical EOT sampling during training
  5. Hard projection to admissible region at convergence
  6. Constrained size search: finds minimal patch area satisfying conf < tau (0.25)
  7. Modular ablation support (fixed placement, B/W palette, no shading, no EOT)
"""

from typing import Dict, List, Tuple, Optional, Any
import os
import csv
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from adv_camouflage.attack.pattern import BinaryPatternGenerator
from adv_camouflage.attack.placement import PatchPlacement, create_multi_start_placements
from adv_camouflage.attack.render import PhysicalRadianceRenderer
from adv_camouflage.attack.eot import EOTSampler
from adv_camouflage.attack.losses import CompositeCamouflageLoss
from adv_camouflage.attack.project import project_placement_to_admissible, measure_spillover
from adv_camouflage.geometry.segmentation import TargetSegmenter
from adv_camouflage.geometry.distance import DistanceAnalyzer
from adv_camouflage.geometry.palette import BackgroundAwarePalette
from adv_camouflage.geometry.homography import build_homography_matrix
from adv_camouflage.detector.yolo_wrapper import YOLODetectorWrapper


class CamouflageOptimizer:
    def __init__(
        self,
        detector: YOLODetectorWrapper,
        renderer: Optional[PhysicalRadianceRenderer] = None,
        patch_res: int = 64,
        lr_pattern: float = 0.12,
        lr_placement: float = 0.02,
        device: Optional[torch.device] = None,
    ):
        self.detector = detector
        self.device = device or detector.device
        self.renderer = renderer or PhysicalRadianceRenderer().to(self.device)
        self.patch_res = patch_res
        self.lr_pattern = lr_pattern
        self.lr_placement = lr_placement
        self.loss_fn = CompositeCamouflageLoss(weight_area=0.20)
        self.eot = EOTSampler()

    def optimize_target(
        self,
        image_rgb: np.ndarray,
        target_info: Dict[str, Any],
        total_steps: int = 80,
        enable_eot: bool = True,
        enable_shading: bool = True,
        optimize_placement: bool = True,
        use_bw_palette: bool = False,
        target_scale_fraction: Optional[float] = None,
        log_csv_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Runs joint optimization for a single target object in an image.
        image_rgb: (H, W, 3) uint8 numpy array
        target_info: dict with pixel_box, cls_id, class_name, etc.
        """
        image_rgb = np.ascontiguousarray(image_rgb)
        H, W = image_rgb.shape[:2]
        pixel_box = target_info["pixel_box"]
        cls_id = target_info["cls_id"]
        box_area = target_info["area_px"]
        bw = pixel_box[2] - pixel_box[0]
        bh = pixel_box[3] - pixel_box[1]

        # 1. Target Silhouette Segmentation
        segmenter = TargetSegmenter()
        img_bgr = image_rgb[:, :, ::-1].copy()
        mask_u8, seg_method = segmenter.extract_silhouette(img_bgr, pixel_box)
        M_hull_np = (mask_u8 > 128).astype(np.float32)
        M_hull_tensor = torch.from_numpy(M_hull_np).float().unsqueeze(0).unsqueeze(0).to(self.device)

        # 2. Distance Transform and Clearance
        dist_analyzer = DistanceAnalyzer()
        dist_res = dist_analyzer.analyze_silhouette(mask_u8)
        R_max = float(dist_res["R_max"])
        best_center = dist_res["top_k_centers"][0]

        # Distance outside tensor for barrier loss
        D_outside_tensor = torch.from_numpy(dist_res["D_outside"]).float().unsqueeze(0).unsqueeze(0).to(self.device)

        # 3. Two-Colour Palette
        if use_bw_palette:
            c0 = torch.tensor([0.0, 0.0, 0.0], device=self.device, dtype=torch.float32)
            c1 = torch.tensor([1.0, 1.0, 1.0], device=self.device, dtype=torch.float32)
            palette_info = {"c0_srgb": [0, 0, 0], "c1_srgb": [255, 255, 255], "delta_e": 100.0}
        else:
            palette_generator = BackgroundAwarePalette()
            palette_info = palette_generator.compute_palette(img_bgr, mask_u8)
            c0 = torch.tensor(palette_info["c0_srgb"], device=self.device, dtype=torch.float32)
            c1 = torch.tensor(palette_info["c1_srgb"], device=self.device, dtype=torch.float32)

        # 4. Modules: Pattern Generator and Placement
        init_scale = target_scale_fraction if target_scale_fraction is not None else 0.45
        placement = PatchPlacement(
            init_center=best_center,
            base_radius=R_max,
            canvas_hw=(H, W),
            target_box_wh=(float(bw), float(bh)),
            initial_scale_fraction=init_scale,
        ).to(self.device)

        pattern = BinaryPatternGenerator(
            height=self.patch_res,
            width=self.patch_res,
        ).to(self.device)

        # Clean image tensor
        I_clean = torch.from_numpy(image_rgb).float().permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0

        # Baseline Clean Detection Score
        with torch.no_grad():
            _, clean_metrics = self.detector.compute_detection_loss(
                x=I_clean,
                target_pixel_box=pixel_box,
                target_class_id=cls_id,
            )
            clean_conf = clean_metrics["max_conf"]

        # Setup Optimizer
        params = [{"params": pattern.parameters(), "lr": self.lr_pattern}]
        if optimize_placement:
            params.append({"params": placement.parameters(), "lr": self.lr_placement})

        optimizer = optim.Adam(params)

        # CSV Logging setup
        csv_file = None
        csv_writer = None
        if log_csv_path:
            os.makedirs(os.path.dirname(log_csv_path), exist_ok=True)
            csv_file = open(log_csv_path, "w", newline="")
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([
                "step", "total_loss", "loss_det", "max_conf", "loss_in", "loss_area",
                "temperature", "scale"
            ])

        # 5. Joint Optimization Loop
        best_adv_image = None
        min_loss = float("inf")

        for step in range(total_steps):
            optimizer.zero_grad()

            # EOT Perturbations
            if enable_eot:
                theta_used = self.eot.sample_placement_jitter(placement.theta)
                gamma_used = self.eot.sample_gamma()
            else:
                theta_used = placement.theta
                gamma_used = 2.0

            # Homography & Pattern
            H_fwd = build_homography_matrix(theta_used, placement.init_center_norm, placement.box_scale_norm)
            patch_rgb, _ = pattern(c0, c1, step=step, total_steps=total_steps)

            # Render
            I_adv, alpha, _ = self.renderer.render(
                I_clean=I_clean,
                patch_rgb=patch_rgb,
                H_fwd=H_fwd,
                M_hull=M_hull_tensor,
                R_max=R_max if enable_shading else 0.0,
                gamma=gamma_used,
            )

            # Optional radiometric EOT
            if enable_eot:
                I_input = self.eot.apply_radiometric_eot(I_adv, active_mask=alpha)
            else:
                I_input = I_adv

            # Target detection suppression loss
            loss_det, det_metrics = self.detector.compute_detection_loss(
                x=I_input,
                target_pixel_box=pixel_box,
                target_class_id=cls_id,
            )

            tau = pattern.get_temperature(step, total_steps)
            total_loss, breakdown = self.loss_fn(
                loss_det=loss_det,
                placement=placement,
                D_outside=D_outside_tensor,
                patch_logits=pattern.logits,
                target_box_area_px=float(box_area),
                tau=tau,
            )

            total_loss.backward()
            optimizer.step()

            # Log metrics
            max_c = det_metrics["max_conf"]
            curr_scale = float(torch.exp(placement.theta[2]).item())
            if csv_writer:
                csv_writer.writerow([
                    step, total_loss.item(), loss_det.item(), max_c,
                    breakdown["loss_in"], breakdown["loss_area"], tau, curr_scale
                ])

        if csv_file:
            csv_file.close()

        # 6. Post-Optimization: Hard Projection & Strict Binarization
        project_res = project_placement_to_admissible(
            placement=placement,
            admissible_mask=dist_res["admissible_mask"],
            D_inside=dist_res["D_inside"],
        )

        with torch.no_grad():
            # Strict binary pattern (STE hard forward)
            final_patch_rgb, _ = pattern(c0, c1, force_hard=True)
            final_H_fwd = placement.get_homography()

            # Render final adversarial image (camera sensor gamma = 2.0)
            I_adv_final, alpha_final, warped_patch_srgb = self.renderer.render(
                I_clean=I_clean,
                patch_rgb=final_patch_rgb,
                H_fwd=final_H_fwd,
                M_hull=M_hull_tensor,
                R_max=R_max,
                gamma=2.0,
            )

            # Final detection metrics
            _, final_metrics = self.detector.compute_detection_loss(
                x=I_adv_final,
                target_pixel_box=pixel_box,
                target_class_id=cls_id,
            )
            final_conf = final_metrics["max_conf"]

            # Measure spillover
            spill_info = measure_spillover(warped_alpha=alpha_final, M_hull=M_hull_tensor)

            # Measure patch area relative to object hull
            patch_pixels = int(torch.sum(alpha_final > 0.05).item())
            hull_pixels = max(1, int(torch.sum(M_hull_tensor > 0.5).item()))
            patch_area_pct = (patch_pixels / hull_pixels) * 100.0

            # Verify background bit-identical
            outside = (alpha_final == 0.0).repeat(1, 3, 1, 1)
            bg_diff = float(torch.max(torch.abs(I_adv_final[outside] - I_clean[outside])).item())

            # Export discrete design image
            design_u8 = pattern.get_discrete_design_image(c0, c1)

            # Final adversarial numpy image
            adv_img_np = (I_adv_final[0].permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)

        # Successful camouflage if final confidence suppressed below threshold 0.25
        tau_threshold = 0.25
        is_attack_success = (final_conf < tau_threshold)

        return {
            "clean_conf": float(clean_conf),
            "adv_conf": float(final_conf),
            "conf_reduction": float(clean_conf - final_conf),
            "attack_success": bool(is_attack_success),
            "patch_area_pct": float(patch_area_pct),
            "spillover_rate": float(spill_info["spillover_rate"]),
            "spillover_pixels": int(spill_info["spillover_pixels"]),
            "bg_max_diff": float(bg_diff),
            "palette": palette_info,
            "project_converged": bool(project_res["converged"]),
            "adv_image_rgb": adv_img_np,
            "design_image_rgb": design_u8,
            "final_alpha": alpha_final.squeeze().cpu().numpy(),
            "target_class": target_info["class_name"],
            "target_box": pixel_box,
        }

    def constrained_size_search(
        self,
        image_rgb: np.ndarray,
        target_info: Dict[str, Any],
        tau_thresh: float = 0.25,
        scales: Tuple[float, ...] = (0.20, 0.30, 0.40, 0.50, 0.60, 0.75),
        steps_per_scale: int = 50,
    ) -> Dict[str, Any]:
        """
        Searches over patch scales to find the minimal patch area fraction that satisfies conf < tau.
        """
        best_result = None
        min_successful_scale = None

        for scale in sorted(scales):
            res = self.optimize_target(
                image_rgb=image_rgb,
                target_info=target_info,
                total_steps=steps_per_scale,
                target_scale_fraction=scale,
            )
            if res["adv_conf"] < tau_thresh:
                best_result = res
                min_successful_scale = scale
                break
            else:
                best_result = res

        return {
            "min_successful_scale": min_successful_scale,
            "optimal_result": best_result,
        }
