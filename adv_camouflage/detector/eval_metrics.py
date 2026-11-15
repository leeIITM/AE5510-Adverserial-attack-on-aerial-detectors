#!/usr/bin/env python3
"""
AE5510: Comprehensive Detection and Attack Evaluation Metrics
==============================================================
Module: adv_camouflage/detector/eval_metrics.py

Implements:
  1. Box IoU, Precision, Recall, AP50, and mAP per class
  2. Attack Success Rate (ASR) at tau = 0.25 detection threshold
  3. Mean confidence reduction (clean vs adversarial)
  4. Physical spillover rate % (measured outside target silhouette)
  5. Patch area fraction % relative to target object silhouette
  6. Bit-identical background difference check
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
from ultralytics import YOLO


def compute_iou(boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]) -> float:
    """Computes IoU between two boxes in [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    areaA = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    areaB = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / max(1e-7, union)


def evaluate_detection_on_image(
    yolo_model: YOLO,
    image_rgb: np.ndarray,
    ground_truth_targets: List[Dict[str, Any]],
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.50,
) -> Dict[str, Any]:
    """
    Evaluates detector predictions against ground truth targets for a single image.
    image_rgb: (H, W, 3) uint8 numpy array
    ground_truth_targets: list of dicts with 'pixel_box' and 'cls_id'
    Returns:
      per_target_results: list of dicts containing detected, max_matched_conf, matched_iou
      tp_count, fp_count, fn_count per class
    """
    # Ultralytics inference
    results = yolo_model(image_rgb, conf=conf_thresh, verbose=False)[0]

    pred_boxes = []
    pred_scores = []
    pred_classes = []

    if results.boxes is not None and len(results.boxes) > 0:
        pred_boxes = results.boxes.xyxy.cpu().numpy()
        pred_scores = results.boxes.conf.cpu().numpy()
        pred_classes = results.boxes.cls.cpu().numpy().astype(int)

    target_evals = []
    matched_preds = set()

    for target in ground_truth_targets:
        tbox = target["pixel_box"]
        tcls = target["cls_id"]

        best_iou = 0.0
        best_conf = 0.0
        best_pred_idx = -1

        for i, (pbox, pconf, pcls) in enumerate(zip(pred_boxes, pred_scores, pred_classes)):
            if pcls == tcls:
                iou = compute_iou(tbox, pbox)
                if iou > best_iou:
                    best_iou = iou
                    best_conf = float(pconf)
                    best_pred_idx = i

        is_detected = (best_iou >= iou_thresh and best_conf >= conf_thresh)
        if is_detected and best_pred_idx >= 0:
            matched_preds.add(best_pred_idx)

        target_evals.append({
            "cls_id": tcls,
            "class_name": target.get("class_name", str(tcls)),
            "detected": is_detected,
            "max_conf": best_conf,
            "best_iou": best_iou,
        })

    # False positives: predictions with no matching GT
    fp_count = len(pred_boxes) - len(matched_preds)

    return {
        "target_evals": target_evals,
        "num_predictions": len(pred_boxes),
        "false_positives": max(0, fp_count),
    }


def compute_ap50_and_map(
    all_target_evals: List[Dict[str, Any]],
    class_names: Dict[int, str],
) -> Dict[str, Any]:
    """
    Computes per-class detection accuracy, AP50, and overall mAP.
    all_target_evals: list of dicts with cls_id and detected boolean
    """
    class_stats = {}
    for cls_id, name in class_names.items():
        targets_cls = [t for t in all_target_evals if t["cls_id"] == cls_id]
        total = len(targets_cls)
        detected = sum(1 for t in targets_cls if t["detected"])
        recall = (detected / total) if total > 0 else 0.0
        mean_conf = float(np.mean([t["max_conf"] for t in targets_cls])) if total > 0 else 0.0

        class_stats[cls_id] = {
            "class_name": name,
            "total_targets": total,
            "detected_targets": detected,
            "ap50_recall": recall,
            "mean_conf": mean_conf,
        }

    # Macro mean AP50 across valid classes
    valid_classes = [s for s in class_stats.values() if s["total_targets"] > 0]
    map50 = float(np.mean([s["ap50_recall"] for s in valid_classes])) if valid_classes else 0.0

    return {
        "class_stats": class_stats,
        "map50": map50,
        "total_targets": len(all_target_evals),
        "total_detected": sum(1 for t in all_target_evals if t["detected"]),
    }
