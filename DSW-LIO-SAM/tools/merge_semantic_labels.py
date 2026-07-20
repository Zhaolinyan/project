#!/usr/bin/env python3
"""Merge Cylinder3D .label predictions with XYZI .bin frames for semantic replay."""

import argparse
import os
import shutil
from pathlib import Path

import numpy as np


SEMANTICKITTI_TO_DSW = {
    0: 0,     # unlabeled -> UNKNOWN
    10: 1,    # car -> CAR
    11: 5,    # bicycle -> BICYCLE
    13: 3,    # bus -> BUS
    15: 6,    # motorcycle -> MOTORCYCLE
    16: 3,    # on-rails -> BUS
    18: 2,    # truck -> TRUCK
    20: 2,    # other-vehicle -> TRUCK
    30: 4,    # person -> PERSON
    31: 5,    # bicyclist -> BICYCLE
    32: 6,    # motorcyclist -> MOTORCYCLE
    40: 8,    # road -> ROAD
    44: 8,    # parking -> ROAD
    48: 9,    # sidewalk -> SIDEWALK
    49: 10,   # other-ground -> TERRAIN
    50: 7,    # building -> BUILDING
    51: 13,   # fence -> FENCE
    52: 14,   # other-structure -> WALL
    60: 9,    # lane-marking -> SIDEWALK
    70: 11,   # vegetation -> TREE
    71: 11,   # trunk -> TREE
    72: 10,   # terrain -> TERRAIN
    80: 12,   # pole -> POLE
    81: 12,   # traffic-sign -> POLE
    99: 0,    # other-object -> UNKNOWN
    252: 1,   # moving-car -> CAR
    253: 5,   # moving-bicyclist -> BICYCLE
    254: 4,   # moving-person -> PERSON
    255: 6,   # moving-motorcyclist -> MOTORCYCLE
    256: 2,   # moving-on-rails -> TRUCK
    257: 3,   # moving-bus -> BUS
    258: 2,   # moving-truck -> TRUCK
    259: 2,   # moving-other-vehicle -> TRUCK
}

SEMANTIC_DTYPE = np.dtype([
    ("x", "<f4"),
    ("y", "<f4"),
    ("z", "<f4"),
    ("intensity", "<f4"),
    ("label", "<u4"),
])


def build_lut():
    lut = np.zeros(max(SEMANTICKITTI_TO_DSW) + 1, dtype=np.uint32)
    for src, dst in SEMANTICKITTI_TO_DSW.items():
        lut[src] = dst
    return lut


def map_labels(raw_labels, lut):
    semantic_ids = raw_labels & 0xFFFF
    mapped = np.zeros(len(semantic_ids), dtype=np.uint32)
    valid = semantic_ids < len(lut)
    mapped[valid] = lut[semantic_ids[valid]]
    return semantic_ids, mapped


def copy_if_present(src_dir, dst_dir, name):
    src = src_dir / name
    if src.is_file():
        shutil.copy2(src, dst_dir / name)
        return True
    return False


def merge_frames(input_bin, input_label, output_dir):
    input_bin = Path(input_bin)
    input_label = Path(input_label)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lut = build_lut()
    bin_files = sorted(input_bin.glob("*.bin"))
    merged = 0
    skipped = 0

    for bin_path in bin_files:
        label_path = input_label / f"{bin_path.stem}.label"
        if not label_path.is_file():
            print(f"skip {bin_path.stem}: missing label")
            skipped += 1
            continue

        xyzi = np.fromfile(bin_path, dtype="<f4")
        if xyzi.size % 4 != 0:
            print(f"skip {bin_path.stem}: invalid XYZI length")
            skipped += 1
            continue
        xyzi = xyzi.reshape(-1, 4)

        raw_labels = np.fromfile(label_path, dtype="<u4")
        if len(raw_labels) != len(xyzi):
            print(f"skip {bin_path.stem}: point count mismatch bin={len(xyzi)} label={len(raw_labels)}")
            skipped += 1
            continue

        semantic_ids, labels = map_labels(raw_labels, lut)
        out = np.zeros(len(xyzi), dtype=SEMANTIC_DTYPE)
        out["x"] = xyzi[:, 0]
        out["y"] = xyzi[:, 1]
        out["z"] = xyzi[:, 2]
        out["intensity"] = xyzi[:, 3]
        out["label"] = labels
        out.tofile(output_dir / bin_path.name)
        merged += 1
        print(
            f"merged {bin_path.name}: points={len(xyzi)} "
            f"raw=[{int(semantic_ids.min())},{int(semantic_ids.max())}] "
            f"mapped=[{int(labels.min())},{int(labels.max())}]"
        )

    copied_manifest = copy_if_present(input_bin, output_dir, "frames.csv")
    copied_metadata = copy_if_present(input_label, output_dir, "inference_metadata.json")
    if not copied_metadata:
        copied_metadata = copy_if_present(input_bin, output_dir, "inference_metadata.json")

    if not copied_manifest:
        print("warning: frames.csv was not found; semantic_replay_node requires it")
    if not copied_metadata:
        print("warning: inference_metadata.json was not found; semantic_replay_node requires it")

    return merged, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-bin", required=True, help="Directory containing source XYZI .bin frames")
    parser.add_argument("--input-label", required=True, help="Directory containing Cylinder3D .label predictions")
    parser.add_argument("--output-dir", required=True, help="Directory for semantic replay .bin frames")
    args = parser.parse_args()

    merged, skipped = merge_frames(args.input_bin, args.input_label, args.output_dir)
    print(f"done: merged={merged}, skipped={skipped}, output={os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
