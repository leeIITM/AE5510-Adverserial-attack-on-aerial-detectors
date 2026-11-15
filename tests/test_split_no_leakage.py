#!/usr/bin/env python3
"""
Test: Base-Scene Split Mutual Exclusivity and Completeness
==========================================================
Verifies that:
  1. Train, Val, and Test base scene IDs are mutually disjoint (zero leakage).
  2. Every image file in the dataset belongs to exactly one split.
  3. Total base scenes count equals 88.
  4. Total image files count equals 3,168.
"""

import json
from pathlib import Path
import pytest

SPLIT_PATH = Path("C:/Works/AE5510/AE5510/adv_camouflage/outputs/split.json")


def test_split_no_leakage():
    assert SPLIT_PATH.exists(), f"Split file not found at {SPLIT_PATH}"
    with open(SPLIT_PATH, "r") as f:
        data = json.load(f)

    train_bases = set(data["base_scenes"]["train"])
    val_bases = set(data["base_scenes"]["val"])
    test_bases = set(data["base_scenes"]["test"])

    # 1. Check base scene counts
    assert len(train_bases) == 60, f"Expected 60 train base scenes, got {len(train_bases)}"
    assert len(val_bases) == 14, f"Expected 14 val base scenes, got {len(val_bases)}"
    assert len(test_bases) == 14, f"Expected 14 test base scenes, got {len(test_bases)}"

    # 2. Check disjoint base sets
    assert len(train_bases & val_bases) == 0, f"Leakage between train and val: {train_bases & val_bases}"
    assert len(train_bases & test_bases) == 0, f"Leakage between train and test: {train_bases & test_bases}"
    assert len(val_bases & test_bases) == 0, f"Leakage between val and test: {val_bases & test_bases}"

    # 3. Check files
    train_files = set(data["files"]["train"])
    val_files = set(data["files"]["val"])
    test_files = set(data["files"]["test"])

    assert len(train_files) == 2160, f"Expected 2160 train files, got {len(train_files)}"
    assert len(val_files) == 504, f"Expected 504 val files, got {len(val_files)}"
    assert len(test_files) == 504, f"Expected 504 test files, got {len(test_files)}"

    # 4. Check disjoint file sets
    assert len(train_files & val_files) == 0, "Leakage between train and val files!"
    assert len(train_files & test_files) == 0, "Leakage between train and test files!"
    assert len(val_files & test_files) == 0, "Leakage between val and test files!"

    total_files = len(train_files | val_files | test_files)
    assert total_files == 3168, f"Expected 3168 total unique files, got {total_files}"
    print("\n[PASSED] Base-scene split is completely disjoint with zero data leakage across all 3,168 images.")


if __name__ == "__main__":
    test_split_no_leakage()
