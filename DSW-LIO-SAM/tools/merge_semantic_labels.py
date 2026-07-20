#!/usr/bin/env python3
"""
Merge offline Cylinder3D labels:
  rooster_bin_001/*.bin + rooster_label_001/*.label -> rooster_semantic_001/*.bin
"""

import os

import numpy as np


# SemanticKITTI raw label id -> dsw_lio_sam SemanticLabel id.
SEMANTICKITTI_TO_LABEL = {
    0: 0,    # unlabeled     -> UNKNOWN
    10: 1,   # car           -> CAR
    11: 5,   # bicycle       -> BICYCLE
    15: 6,   # motorcycle    -> MOTORCYCLE
    18: 2,   # truck         -> TRUCK
    20: 2,   # other-vehicle -> TRUCK
    30: 4,   # person        -> PERSON
    31: 5,   # bicyclist     -> BICYCLE
    32: 6,   # motorcyclist  -> MOTORCYCLE
    40: 8,   # road          -> ROAD
    44: 8,   # parking       -> ROAD
    48: 9,   # sidewalk      -> SIDEWALK
    49: 10,  # other-ground  -> TERRAIN
    50: 7,   # building      -> BUILDING
    51: 13,  # fence         -> FENCE
    70: 11,  # vegetation    -> TREE
    71: 11,  # trunk         -> TREE
    72: 10,  # terrain       -> TERRAIN
    80: 12,  # pole          -> POLE
    81: 12,  # traffic-sign  -> POLE
}

_LUT_SIZE = 82
_LABEL_LUT = np.zeros(_LUT_SIZE, dtype=np.uint32)
for _k, _v in SEMANTICKITTI_TO_LABEL.items():
    _LABEL_LUT[_k] = _v

BASE_DIR = "/mnt/e/Code_reproduction/DSW-LIO-SAM/datasets"

INPUT_BIN = os.path.join(BASE_DIR, "rooster_bin_001")
INPUT_LABEL = os.path.join(BASE_DIR, "rooster_label_001")
OUTPUT_DIR = os.path.join(BASE_DIR, "rooster_semantic_001")

os.makedirs(OUTPUT_DIR, exist_ok=True)

bin_files = sorted([f for f in os.listdir(INPUT_BIN) if f.endswith(".bin")])
print(f"找到 {len(bin_files)} 帧")

for fname in bin_files:
    stem = fname.replace(".bin", "")
    bin_path = os.path.join(INPUT_BIN, fname)
    label_path = os.path.join(INPUT_LABEL, stem + ".label")

    if not os.path.exists(label_path):
        print(f"  跳过 {stem}：无对应 label")
        continue

    xyzi = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    labels = np.fromfile(label_path, dtype=np.uint32)
    semantic_ids = labels & 0xFFFF
    labels_mapped = _LABEL_LUT[np.clip(semantic_ids, 0, _LUT_SIZE - 1)]
    n_pts = xyzi.shape[0]

    if n_pts != len(labels):
        print(f"  点数不一致：{stem} bin={n_pts}, label={len(labels)}")
        continue

    combined = np.zeros(
        n_pts,
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("intensity", "f4"),
            ("label", "u4"),
        ],
    )
    combined["x"] = xyzi[:, 0]
    combined["y"] = xyzi[:, 1]
    combined["z"] = xyzi[:, 2]
    combined["intensity"] = xyzi[:, 3]
    combined["label"] = labels_mapped

    out_path = os.path.join(OUTPUT_DIR, stem + ".bin")
    combined.tofile(out_path)
    print(
        f"  {stem}: {n_pts} 点, "
        f"raw=[{semantic_ids.min()}~{semantic_ids.max()}] "
        f"-> mapped=[{labels_mapped.min()}~{labels_mapped.max()}]"
    )

print(f"\n完成 -> {OUTPUT_DIR}")
