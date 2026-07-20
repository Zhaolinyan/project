import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "semantic_replay_utils", ROOT / "semantic_bridge" / "semantic_replay_utils.py")
REPLAY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPLAY)


RAW_DTYPE = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "<f4")
])


def make_cloud():
    points = np.zeros(4, dtype=RAW_DTYPE)
    points[0] = (1.0, 2.0, 3.0, 10.0)
    points[1] = (0.0, 0.0, 0.0, 20.0)
    points[2] = (4.0, 5.0, 6.0, 30.0)
    points[3] = (np.nan, 0.0, 0.0, 40.0)

    fields = [
        SimpleNamespace(name="x", offset=0, datatype=REPLAY.FLOAT32),
        SimpleNamespace(name="y", offset=4, datatype=REPLAY.FLOAT32),
        SimpleNamespace(name="z", offset=8, datatype=REPLAY.FLOAT32),
        SimpleNamespace(name="intensity", offset=12, datatype=REPLAY.FLOAT32),
    ]
    return SimpleNamespace(
        height=1,
        width=len(points),
        point_step=RAW_DTYPE.itemsize,
        fields=fields,
        is_bigendian=False,
        data=points.tobytes(),
    )


def make_semantic_frame():
    frame = np.zeros(2, dtype=REPLAY.SEMANTIC_DTYPE)
    frame["x"] = [1.0, 4.0]
    frame["y"] = [2.0, 5.0]
    frame["z"] = [3.0, 6.0]
    frame["label"] = [8, 11]
    return frame


def test_align_semantic_frame_restores_original_order():
    aligned = REPLAY.align_semantic_frame(make_cloud(), make_semantic_frame())
    assert len(aligned) == 4
    assert aligned["label"].tolist() == [8, 0, 11, 0]
    assert aligned["intensity"].tolist() == [10.0, 20.0, 30.0, 40.0]


def test_align_semantic_frame_rejects_count_mismatch():
    with pytest.raises(ValueError, match="point count mismatch"):
        REPLAY.align_semantic_frame(make_cloud(), make_semantic_frame()[:1])


def test_align_semantic_frame_rejects_low_coordinate_coverage():
    frame = make_semantic_frame()
    frame["x"][1] = 99.0

    with pytest.raises(ValueError, match="coverage 0.500 below required 0.750"):
        REPLAY.align_semantic_frame(
            make_cloud(),
            frame,
            allow_coordinate_lookup=True,
            minimum_coverage=0.75,
        )


def test_find_frame_for_stamp_uses_nearest_with_tolerance():
    frames = {
        100: {"path": "a.bin"},
        200: {"path": "b.bin"},
    }
    sorted_stamps = sorted(frames)

    assert REPLAY.find_frame_for_stamp(frames, sorted_stamps, 100, 0) == (100, frames[100])
    assert REPLAY.find_frame_for_stamp(frames, sorted_stamps, 198, 5) == (200, frames[200])
    assert REPLAY.find_frame_for_stamp(frames, sorted_stamps, 198, 1) == (None, None)


def test_load_manifest_uses_timestamp_keys(tmp_path):
    frame_path = tmp_path / "000000.bin"
    make_semantic_frame().tofile(frame_path)
    with (tmp_path / "frames.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["filename", "stamp_ns", "point_count", "original_point_count"],
        )
        writer.writeheader()
        writer.writerow({
            "filename": frame_path.name,
            "stamp_ns": 123,
            "point_count": 2,
            "original_point_count": 4,
        })

    frames = REPLAY.load_manifest(str(tmp_path))
    assert list(frames) == [123]
    assert frames[123]["point_count"] == 2
    assert frames[123]["original_point_count"] == 4


def test_validate_inference_metadata_rejects_old_incompatible_labels(tmp_path):
    (tmp_path / 'inference_metadata.json').write_text(json.dumps({
        'model_kind': 'generic_pretrained_fallback',
        'preprocessing_profile': 'newer_college_finetuned',
    }), encoding='utf-8')

    with pytest.raises(ValueError, match='checkpoint-incompatible'):
        REPLAY.validate_inference_metadata(str(tmp_path))


def test_validate_inference_metadata_accepts_matching_profile(tmp_path):
    metadata = {
        'model_kind': 'generic_pretrained_fallback',
        'preprocessing_profile': 'semantic_kitti_pretrained',
    }
    (tmp_path / 'inference_metadata.json').write_text(
        json.dumps(metadata), encoding='utf-8')

    assert REPLAY.validate_inference_metadata(str(tmp_path)) == metadata
