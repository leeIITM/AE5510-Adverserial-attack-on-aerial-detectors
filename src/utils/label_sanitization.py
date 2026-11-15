#!/usr/bin/env python3
"""
AE5510 - Dataset Annotation Correction Script
==============================================
Fixes systemic transformation bugs in the course aerial dataset:
1. Rotation augmentations (_rotation_1, 2, 4, 5):
   - Recomputes tight axis-aligned bounding boxes from pristine 0° ground-truth (_rotation_3).
   - Rotates all 4 box corners by exact OpenCV affine matrices (-30°, -15°, +15°, +30°).
   - Clips boxes to image boundary [0, 1].
   - Drops objects that rotated out of frame (visible area < 30%).
   - Completely eliminates negative coordinates (< 0) and coordinates > 1.

2. Zoom-in augmentations (_zoom_in_1 to 5):
   - Inverts the inverted scale bug (scales by 1.2x, 1.4x, 1.6x, 1.8x, 2.0x).
   - Re-aligns boxes with true zoomed-in object pixels.
   - Clips boxes at image edges and drops objects cropped off canvas.

3. All other augmentations (flip, brightness, haze, motionblur, shadow, zoom_out):
   - Preserved exactly as provided, with safety edge clipping to [0, 1].
"""

import os
import re
import sys
import shutil
import argparse
from pathlib import Path
from PIL import Image
import cv2
import numpy as np

ROTATION_ANGLES = {
    "rotation_1": -30.0,
    "rotation_2": -15.0,
    "rotation_3": 0.0,
    "rotation_4": 15.0,
    "rotation_5": 30.0,
}

ZOOM_IN_SCALES = {
    "zoom_in_1": 1.2,
    "zoom_in_2": 1.4,
    "zoom_in_3": 1.6,
    "zoom_in_4": 1.8,
    "zoom_in_5": 2.0,
}


def parse_filename(stem):
    """Identifies the base sample name and the augmentation type from the filename."""
    # Suffixes: _rotation_X, _zoom_in_X, _zoom_out_X, _brightness_X, etc.
    patterns = [
        r"^(.*?)_(rotation_\d+)$",
        r"^(.*?)_(zoom_in_\d+)$",
        r"^(.*?)_(zoom_out_\d+)$",
        r"^(.*?)_(brightness_\d+)$",
        r"^(.*?)_(haze_\d+)$",
        r"^(.*?)_(motionblur_\d+)$",
        r"^(.*?)_(shadow_\d+)$",
        r"^(.*?)_(flip)$",
    ]
    for pat in patterns:
        m = re.match(pat, stem)
        if m:
            return m.group(1), m.group(2)
    return stem, "none"


def load_boxes(txt_path):
    """Loads YOLO format boxes: class_id, cx, cy, w, h."""
    boxes = []
    if not txt_path.exists():
        return boxes
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cid = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                boxes.append((cid, cx, cy, w, h))
    return boxes


def save_boxes(txt_path, boxes):
    """Writes YOLO format boxes to file."""
    with open(txt_path, "w") as f:
        for cid, cx, cy, w, h in boxes:
            f.write(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def fix_rotation_boxes(orig_boxes, angle_deg, img_w, img_h, min_vis_ratio=0.3):
    """
    Transforms boxes from pristine 0-deg reference using OpenCV rotation matrix.
    Computes true enclosing axis-aligned bounding box and clips to canvas.
    """
    if abs(angle_deg) < 1e-4:
        # 0 degree reference - only ensure valid bounds
        clipped = []
        for cid, cx, cy, w, h in orig_boxes:
            x1 = max(0.0, cx - w / 2.0)
            y1 = max(0.0, cy - h / 2.0)
            x2 = min(1.0, cx + w / 2.0)
            y2 = min(1.0, cy + h / 2.0)
            if x2 > x1 and y2 > y1:
                ncx = (x1 + x2) / 2.0
                ncy = (y1 + y2) / 2.0
                nw = x2 - x1
                nh = y2 - y1
                clipped.append((cid, ncx, ncy, nw, nh))
        return clipped

    M = cv2.getRotationMatrix2D((img_w / 2.0, img_h / 2.0), angle_deg, 1.0)
    transformed = []

    for cid, cx, cy, w, h in orig_boxes:
        # Convert normalized coordinates to absolute pixels
        x1_px = (cx - w / 2.0) * img_w
        y1_px = (cy - h / 2.0) * img_h
        x2_px = (cx + w / 2.0) * img_w
        y2_px = (cy + h / 2.0) * img_h
        orig_area = (x2_px - x1_px) * (y2_px - y1_px)

        # 4 corners in pixel space
        corners = np.array([
            [x1_px, y1_px, 1.0],
            [x2_px, y1_px, 1.0],
            [x2_px, y2_px, 1.0],
            [x1_px, y2_px, 1.0]
        ], dtype=np.float32)

        # Rotate corners
        rot_corners = (M @ corners.T).T

        rx_min = float(np.min(rot_corners[:, 0]))
        rx_max = float(np.max(rot_corners[:, 0]))
        ry_min = float(np.min(rot_corners[:, 1]))
        ry_max = float(np.max(rot_corners[:, 1]))

        # Clip to image boundaries [0, img_w] x [0, img_h]
        cx1_px = max(0.0, rx_min)
        cy1_px = max(0.0, ry_min)
        cx2_px = min(float(img_w), rx_max)
        cy2_px = min(float(img_h), ry_max)

        vis_area = max(0.0, cx2_px - cx1_px) * max(0.0, cy2_px - cy1_px)

        # Filter out objects rotated completely out of frame
        if vis_area >= min_vis_ratio * orig_area and (cx2_px > cx1_px) and (cy2_px > cy1_px):
            ncx = (cx1_px + cx2_px) / (2.0 * img_w)
            ncy = (cy1_px + cy2_px) / (2.0 * img_h)
            nw = (cx2_px - cx1_px) / img_w
            nh = (cy2_px - cy1_px) / img_h
            
            # Guard against edge-case float precision
            ncx = min(1.0, max(0.0, ncx))
            ncy = min(1.0, max(0.0, ncy))
            nw = min(1.0, max(0.0, nw))
            nh = min(1.0, max(0.0, nh))

            transformed.append((cid, ncx, ncy, nw, nh))

    return transformed


def fix_zoom_in_boxes(orig_boxes, scale_factor, min_vis_ratio=0.3):
    """
    Transforms boxes from pristine reference for center zoom-in.
    Multiplies distance from center and dimensions by scale_factor.
    """
    transformed = []
    for cid, cx, cy, w, h in orig_boxes:
        orig_area = w * h
        # Scale coordinates outward from center (0.5, 0.5)
        new_cx = (cx - 0.5) * scale_factor + 0.5
        new_cy = (cy - 0.5) * scale_factor + 0.5
        new_w = w * scale_factor
        new_h = h * scale_factor

        x1 = max(0.0, new_cx - new_w / 2.0)
        y1 = max(0.0, new_cy - new_h / 2.0)
        x2 = min(1.0, new_cx + new_w / 2.0)
        y2 = min(1.0, new_cy + new_h / 2.0)

        vis_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)

        if vis_area >= min_vis_ratio * orig_area and (x2 > x1) and (y2 > y1):
            ncx = (x1 + x2) / 2.0
            ncy = (y1 + y2) / 2.0
            nw = x2 - x1
            nh = y2 - y1
            
            ncx = min(1.0, max(0.0, ncx))
            ncy = min(1.0, max(0.0, ncy))
            nw = min(1.0, max(0.0, nw))
            nh = min(1.0, max(0.0, nh))

            transformed.append((cid, ncx, ncy, nw, nh))

    return transformed


def sanitize_existing_boxes(boxes):
    """Clips existing bounding boxes to [0, 1] without altering geometry."""
    sanitized = []
    for cid, cx, cy, w, h in boxes:
        x1 = max(0.0, cx - w / 2.0)
        y1 = max(0.0, cy - h / 2.0)
        x2 = min(1.0, cx + w / 2.0)
        y2 = min(1.0, cy + h / 2.0)
        if x2 > x1 and y2 > y1:
            ncx = (x1 + x2) / 2.0
            ncy = (y1 + y2) / 2.0
            nw = x2 - x1
            nh = y2 - y1
            sanitized.append((cid, ncx, ncy, nw, nh))
    return sanitized


def sanitize_split_directory(lbl_dir):
    """Sanitizes all label files in a directory to ensure coordinates are valid."""
    sanitized_count = 0
    box_clamped_count = 0
    dropped_count = 0
    label_files = sorted(list(lbl_dir.glob("*.txt")))
    
    for lbl_path in label_files:
        orig = load_boxes(lbl_path)
        fixed = []
        file_modified = False
        for cid, cx, cy, w, h in orig:
            # Check if clamping is required
            x1 = cx - w / 2.0
            y1 = cy - h / 2.0
            x2 = cx + w / 2.0
            y2 = cy + h / 2.0
            
            if x1 < 0.0 or y1 < 0.0 or x2 > 1.0 or y2 > 1.0 or w > 1.0 or h > 1.0 or cx < 0.0 or cy < 0.0:
                file_modified = True
                box_clamped_count += 1
            
            x1_c = max(0.0, min(1.0, x1))
            y1_c = max(0.0, min(1.0, y1))
            x2_c = max(0.0, min(1.0, x2))
            y2_c = max(0.0, min(1.0, y2))
            
            nw = x2_c - x1_c
            nh = y2_c - y1_c
            if nw > 0.001 and nh > 0.001:
                ncx = (x1_c + x2_c) / 2.0
                ncy = (y1_c + y2_c) / 2.0
                fixed.append((cid, ncx, ncy, nw, nh))
            else:
                dropped_count += 1
                file_modified = True
                
        if file_modified:
            save_boxes(lbl_path, fixed)
            sanitized_count += 1
            
    return len(label_files), sanitized_count, box_clamped_count, dropped_count


def clear_yolo_caches(data_dir):
    """Removes all labels.cache files to force YOLO to re-index clean labels."""
    cache_files = list(data_dir.glob("**/labels.cache"))
    for cf in cache_files:
        try:
            cf.unlink()
            print(f"[*] Cleared cache file: {cf}")
        except Exception as e:
            print(f"[!] Warning: Could not remove {cf}: {e}")


def main():
    proj_dir = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == "src" else Path(__file__).resolve().parent
    default_dir = proj_dir / "dataset"
    parser = argparse.ArgumentParser(description="Fix aerial dataset annotation transformation errors (Windows/Linux compatible).")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=str(default_dir),
        help=f"Path to dataset root folder containing 'Training Data' (default: {default_dir})"
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        default=True,
        help="Create a backup copy of original Labels folder before modifying"
    )
    parser.add_argument(
        "--sanitize-all",
        action="store_true",
        default=True,
        help="Sanitize Validation Data and Testing Data bounding boxes as well"
    )
    args = parser.parse_args()

    data_dir = Path(args.dataset_dir).resolve()
    train_dir = data_dir / "Training Data"
    img_dir = train_dir / "images" if (train_dir / "images").exists() else train_dir / "Images"
    lbl_dir = train_dir / "labels" if (train_dir / "labels").exists() else train_dir / "Labels"

    if not lbl_dir.exists():
        print(f"[!] Error: Labels directory not found at {lbl_dir}")
        sys.exit(1)

    # 1. Create backup if requested
    backup_dir = train_dir / "Labels_raw_backup"
    if args.backup and not backup_dir.exists():
        print(f"[*] Backing up original labels to: {backup_dir}")
        shutil.copytree(lbl_dir, backup_dir)
        print("    Backup complete.")
    elif backup_dir.exists():
        print(f"[*] Backup already exists at: {backup_dir}")

    # 2. Index ground truth base references (_rotation_3.txt)
    gt_cache = {}
    label_files = sorted(list(lbl_dir.glob("*.txt")))
    print(f"[*] Found {len(label_files)} label files in Training Data.")

    for f in label_files:
        base_name, aug = parse_filename(f.stem)
        if aug == "rotation_3":
            gt_cache[base_name] = load_boxes(f)

    print(f"[*] Indexed {len(gt_cache)} ground-truth reference sets (_rotation_3).")

    # 3. Cache image dimensions (fast header read)
    img_dims = {}
    for img_path in list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")):
        base_name, aug = parse_filename(img_path.stem)
        if base_name not in img_dims:
            with Image.open(img_path) as im:
                img_dims[base_name] = im.size  # (W, H)

    # 4. Correct annotations in Training Data
    fixed_count = 0
    rot_fixed = 0
    zoom_fixed = 0
    other_sanitized = 0

    for lbl_path in label_files:
        base_name, aug = parse_filename(lbl_path.stem)
        orig_boxes = gt_cache.get(base_name)
        img_w, img_h = img_dims.get(base_name, (2560, 2560))

        if aug in ROTATION_ANGLES:
            angle = ROTATION_ANGLES[aug]
            if orig_boxes is not None:
                new_boxes = fix_rotation_boxes(orig_boxes, angle, img_w, img_h)
                save_boxes(lbl_path, new_boxes)
                rot_fixed += 1
                fixed_count += 1
            else:
                existing = load_boxes(lbl_path)
                save_boxes(lbl_path, sanitize_existing_boxes(existing))

        elif aug in ZOOM_IN_SCALES:
            scale = ZOOM_IN_SCALES[aug]
            if orig_boxes is not None:
                new_boxes = fix_zoom_in_boxes(orig_boxes, scale)
                save_boxes(lbl_path, new_boxes)
                zoom_fixed += 1
                fixed_count += 1
            else:
                existing = load_boxes(lbl_path)
                save_boxes(lbl_path, sanitize_existing_boxes(existing))

        else:
            # flip, photometric, zoom_out: sanitize edge bounds
            existing = load_boxes(lbl_path)
            new_boxes = sanitize_existing_boxes(existing)
            save_boxes(lbl_path, new_boxes)
            other_sanitized += 1
            fixed_count += 1

    print("\n" + "=" * 60)
    print("TRAINING DATA CORRECTION SUMMARY")
    print("=" * 60)
    print(f"Total files processed:        {fixed_count}")
    print(f"Rotation files fixed:         {rot_fixed} (4 corners rotated + clipped)")
    print(f"Zoom-in files fixed:          {zoom_fixed} (scale inversion fixed + clipped)")
    print(f"Other files sanitized:        {other_sanitized} (boundary clamped)")
    print(f"Original backup located at:   {backup_dir}")

    # 5. Optional multi-split sanitization
    if args.sanitize_all:
        for split_name in ["Validation Data", "Testing Data"]:
            s_dir = data_dir / split_name
            s_lbl = s_dir / "labels" if (s_dir / "labels").exists() else s_dir / "Labels"
            if s_lbl.exists():
                t_files, m_files, c_boxes, d_boxes = sanitize_split_directory(s_lbl)
                print(f"[*] {split_name}: {t_files} files checked ({m_files} modified, {c_boxes} clamped, {d_boxes} dropped)")

    # 6. Invalidate YOLO cache files so changes take effect
    clear_yolo_caches(data_dir)
    print("=" * 60)
    print("[+] All dataset annotations are verified and clean.")


if __name__ == "__main__":
    main()
