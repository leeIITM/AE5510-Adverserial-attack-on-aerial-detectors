#!/usr/bin/env python3
"""
AE5510: Adversarial Camouflage Pipeline Runner
==============================================
Runs the proposed binary adversarial camouflage pipeline on the official test images:
  1. Measures baseline clean detections on fine-tuned YOLO11s.
  2. Extracts target silhouette (YOLO11n-seg + GrabCut) and Euclidean distance clearance.
  3. Computes background-adapted CIELAB two-tone palette {c0, c1}.
  4. Optimizes discrete binary decal logits (Straight-Through Estimator) and 6-DoF homography pose.
  5. Enforces hard containment projection (0.000% spillover guarantee).
  6. Applies photometric sensor radiance rendering and evaluates detection suppression.
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Dynamically resolve repository root
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import cv2
import torch
import yaml
from ultralytics import YOLO

from adv_camouflage.detector.yolo_wrapper import YOLODetectorWrapper
from adv_camouflage.attack.optimize import CamouflageOptimizer

CLASS_NAMES = {0: "Aircraft", 1: "Building", 2: "Military Truck", 3: "Tank"}

def run_attack(image_path: Path, label_path: Path, output_dir: Path, device: torch.device):
    weights_path = REPO_ROOT / "weights" / "yolo11s_best.pt"
    assert weights_path.exists(), f"Weights not found: {weights_path}"

    detector = YOLODetectorWrapper(str(weights_path), device=device)
    optimizer = CamouflageOptimizer(detector=detector, patch_res=64, lr_pattern=0.12, lr_placement=0.02, device=device)

    img = cv2.imread(str(image_path))
    assert img is not None, f"Could not read image: {image_path}"
    H, W, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    print(f"\nProcessing scene: {image_path.name} ({W}x{H})")
    
    targets = []
    if label_path.exists():
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_id = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:])
                    bx1 = int((xc - w / 2) * W)
                    by1 = int((yc - h / 2) * H)
                    bx2 = int((xc + w / 2) * W)
                    by2 = int((yc + h / 2) * H)
                    pixel_box = (max(0, bx1), max(0, by1), min(W, bx2), min(H, by2))
                    bw = pixel_box[2] - pixel_box[0]
                    bh = pixel_box[3] - pixel_box[1]
                    targets.append({
                        "cls_id": cls_id,
                        "class_name": CLASS_NAMES.get(cls_id, f"Class_{cls_id}"),
                        "pixel_box": pixel_box,
                        "area_px": float(bw * bh)
                    })

    print(f"Found {len(targets)} ground-truth tactical targets in scene.")
    output_dir.mkdir(parents=True, exist_ok=True)

    adv_img_rgb = img_rgb.copy()
    for idx, t_info in enumerate(targets):
        print(f"  --> Optimizing target {idx+1}/{len(targets)} ({t_info['class_name']})")
        res = optimizer.optimize_target(
            image_rgb=adv_img_rgb,
            target_info=t_info,
            total_steps=80,
            enable_eot=True,
            enable_shading=True,
            optimize_placement=True
        )
        adv_img_rgb = res["adv_image_rgb"]
        print(f"      Clean conf: {res['clean_conf']:.3f} --> Adv conf: {res['final_conf']:.3f} | Spillover: {res['spillover_pct']:.3f}%")

    out_file = output_dir / f"adv_{image_path.name}"
    cv2.imwrite(str(out_file), cv2.cvtColor(adv_img_rgb, cv2.COLOR_RGB2BGR))
    print(f"Saved adversarial image: {out_file}")

def main():
    parser = argparse.ArgumentParser(description="AE5510 Adversarial Camouflage Runner")
    parser.add_argument("--image", type=str, default=None, help="Path to single test image")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Running on device: {device}")

    test_img_dir = REPO_ROOT / "dataset" / "Testing Data" / "images"
    test_lbl_dir = REPO_ROOT / "dataset" / "Testing Data" / "labels"
    out_dir = REPO_ROOT / "outputs" / "adversarial_images"

    if args.image:
        img_p = Path(args.image)
        lbl_p = test_lbl_dir / (img_p.stem + ".txt")
        run_attack(img_p, lbl_p, out_dir, device)
    else:
        test_images = sorted(list(test_img_dir.glob("*.jpg")))
        if not test_images:
            print(f"No test images found in {test_img_dir}")
            return
        sample_img = test_images[0]
        sample_lbl = test_lbl_dir / (sample_img.stem + ".txt")
        print(f"Running sample attack on {sample_img.name}...")
        run_attack(sample_img, sample_lbl, out_dir, device)

if __name__ == "__main__":
    main()
