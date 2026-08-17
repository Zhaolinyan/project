import importlib.util
import struct
import zipfile
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "evaluate_newer_college", ROOT / "tools" / "evaluate_newer_college.py")
EVALUATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATION)


CSV_TEXT = """#sec,nsec,x,y,z,qx,qy,qz,qw
100,0,0,0,0,0,0,0,1
100,100000000,0.1,0,0,0,0,0,1
100,200000000,0.2,0,0,0,0,0,1
"""


def write_ascii_pcd(path, rows):
    lines = [
        "VERSION 0.7",
        "FIELDS x y z intensity roll pitch yaw time",
        "SIZE 4 4 4 4 4 4 4 8",
        "TYPE F F F F F F F F",
        "COUNT 1 1 1 1 1 1 1 1",
        f"WIDTH {len(rows)}",
        "HEIGHT 1",
        f"POINTS {len(rows)}",
        "DATA ascii",
    ]
    lines.extend(" ".join(str(value) for value in row) for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def test_reads_ground_truth_csv_and_official_zip(tmp_path):
    csv_path = tmp_path / "registered_poses.csv"
    csv_path.write_text(CSV_TEXT, encoding="utf-8")
    direct = EVALUATION.read_newer_college_ground_truth(csv_path)

    zip_path = tmp_path / "short_experiment.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr(
            "01_short_experiment/ground_truth/registered_poses.csv", CSV_TEXT)
    archived = EVALUATION.read_newer_college_ground_truth(zip_path)

    np.testing.assert_allclose(direct.timestamps, [100.0, 100.1, 100.2])
    np.testing.assert_allclose(archived.positions, direct.positions)


def test_reads_binary_lio_sam_pcd(tmp_path):
    path = tmp_path / "transformations.pcd"
    header = "\n".join((
        "VERSION 0.7",
        "FIELDS x y z intensity roll pitch yaw time",
        "SIZE 4 4 4 4 4 4 4 8",
        "TYPE F F F F F F F F",
        "COUNT 1 1 1 1 1 1 1 1",
        "WIDTH 2", "HEIGHT 1", "POINTS 2", "DATA binary", "",
    )).encode("ascii")
    records = b"".join((
        struct.pack("<7fd", 1, 2, 3, 0, 0.1, 0.2, 0.3, 100.0),
        struct.pack("<7fd", 4, 5, 6, 1, 0.4, 0.5, 0.6, 101.0),
    ))
    path.write_bytes(header + records)

    trajectory = EVALUATION.read_lio_sam_trajectory(path)

    np.testing.assert_allclose(trajectory.timestamps, [100.0, 101.0])
    np.testing.assert_allclose(trajectory.positions, [[1, 2, 3], [4, 5, 6]])


def test_rejects_ascii_pcd_with_rounded_epoch_timestamps(tmp_path):
    path = tmp_path / "transformations.pcd"
    write_ascii_pcd(path, [
        (0, 0, 0, 0, 0, 0, 0, 1.5943692e9),
        (1, 0, 0, 1, 0, 0, 0, 1.5943692e9),
    ])

    with pytest.raises(ValueError, match="savePCDFileBinary"):
        EVALUATION.read_lio_sam_trajectory(path)


def test_evaluation_removes_global_rigid_transform(tmp_path):
    timestamp_ns = 100_000_000_000 + np.arange(21, dtype=np.int64) * 100_000_000
    timestamps = timestamp_ns.astype(np.float64) * 1e-9
    positions = np.column_stack((
        np.arange(len(timestamps), dtype=float) * 0.2,
        np.sin(np.arange(len(timestamps)) * 0.2),
        np.cos(np.arange(len(timestamps)) * 0.1) * 0.1,
    ))
    csv_path = tmp_path / "registered_poses.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        stream.write("#sec,nsec,x,y,z,qx,qy,qz,qw\n")
        for nanoseconds, position in zip(timestamp_ns, positions):
            sec, nsec = divmod(int(nanoseconds), 1_000_000_000)
            stream.write(
                f"{sec},{nsec},{position[0]},{position[1]},{position[2]},0,0,0,1\n")

    angle = np.pi / 3.0
    rotation = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    offset = np.array([5.0, -2.0, 0.7])
    estimated_positions = (rotation.T @ (positions - offset).T).T
    estimated_yaw = np.full(len(timestamps), -angle)
    rows = [
        (*position, index, 0.0, 0.0, yaw, timestamp)
        for index, (position, yaw, timestamp) in enumerate(
            zip(estimated_positions, estimated_yaw, timestamps))
    ]
    pcd_path = tmp_path / "transformations.pcd"
    write_ascii_pcd(pcd_path, rows)

    result = EVALUATION.evaluate_newer_college(
        csv_path, pcd_path, rpe_delta=1.0, rpe_tolerance=0.25)

    assert result["association"]["matched_poses"] == len(timestamps)
    assert result["ate_translation_m"]["rmse"] < 1e-10
    assert result["rpe"]["translation_m"]["rmse"] < 1e-10
    assert result["rpe"]["rotation_deg"]["rmse"] < 1e-6


def test_association_rejects_large_ground_truth_gap():
    identity = np.tile([0.0, 0.0, 0.0, 1.0], (3, 1))
    ground_truth = EVALUATION.PoseTrajectory(
        np.array([0.0, 0.1, 1.0]), np.zeros((3, 3)), identity)
    estimated = EVALUATION.PoseTrajectory(
        np.array([0.05, 0.5, 1.0]), np.zeros((3, 3)), identity)

    with pytest.raises(ValueError, match="only 2 estimate poses"):
        EVALUATION.associate_ground_truth(ground_truth, estimated, max_time_gap=0.2)
