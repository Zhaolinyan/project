#!/usr/bin/env python3
"""
Evaluate a LIO-SAM trajectory against Newer College ground truth.

The estimated trajectory is read from LIO-SAM's ``transformations.pcd``.
Ground truth can be either ``registered_poses.csv`` or an official Newer
College zip archive containing that file.
"""

import argparse
import csv
import io
import json
import math
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PoseTrajectory:
    timestamps: np.ndarray
    positions: np.ndarray
    quaternions: np.ndarray  # xyzw

    def __post_init__(self):
        count = len(self.timestamps)
        if self.positions.shape != (count, 3):
            raise ValueError("positions must have shape (N, 3)")
        if self.quaternions.shape != (count, 4):
            raise ValueError("quaternions must have shape (N, 4)")
        if count and not np.all(np.isfinite(
                np.column_stack((self.timestamps, self.positions, self.quaternions)))):
            raise ValueError("trajectory contains non-finite values")
        if count > 1 and np.any(np.diff(self.timestamps) <= 0.0):
            raise ValueError("trajectory timestamps must be strictly increasing")

    @property
    def rotations(self):
        return quaternion_to_rotation_matrix(self.quaternions)


def normalize_quaternions(quaternions):
    values = np.asarray(quaternions, dtype=np.float64)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError("trajectory contains a zero-length quaternion")
    return values / norms


def quaternion_to_rotation_matrix(quaternions):
    q = normalize_quaternions(quaternions)
    x, y, z, w = np.moveaxis(q, -1, 0)
    matrices = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    matrices[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    matrices[..., 0, 1] = 2.0 * (x * y - z * w)
    matrices[..., 0, 2] = 2.0 * (x * z + y * w)
    matrices[..., 1, 0] = 2.0 * (x * y + z * w)
    matrices[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    matrices[..., 1, 2] = 2.0 * (y * z - x * w)
    matrices[..., 2, 0] = 2.0 * (x * z - y * w)
    matrices[..., 2, 1] = 2.0 * (y * z + x * w)
    matrices[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return matrices


def rotation_matrix_to_quaternion(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    trace = np.trace(matrix)
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quat = np.array([
            (matrix[2, 1] - matrix[1, 2]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
            0.25 * scale,
        ])
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quat = np.array([
                0.25 * scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
            ])
        elif index == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quat = np.array([
                (matrix[0, 1] + matrix[1, 0]) / scale,
                0.25 * scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
            ])
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quat = np.array([
                (matrix[0, 2] + matrix[2, 0]) / scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
                0.25 * scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ])
    return normalize_quaternions(quat)


def quaternion_multiply(left, right):
    left, right = np.broadcast_arrays(
        np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64))
    lx, ly, lz, lw = np.moveaxis(left, -1, 0)
    rx, ry, rz, rw = np.moveaxis(right, -1, 0)
    return np.stack((
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    ), axis=-1)


def euler_rpy_to_quaternion(roll, pitch, yaw):
    roll = np.asarray(roll, dtype=np.float64) * 0.5
    pitch = np.asarray(pitch, dtype=np.float64) * 0.5
    yaw = np.asarray(yaw, dtype=np.float64) * 0.5
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return normalize_quaternions(np.stack((
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ), axis=-1))


def slerp(left, right, fraction):
    left = normalize_quaternions(left)
    right = normalize_quaternions(right)
    dot = float(np.dot(left, right))
    if dot < 0.0:
        right = -right
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        return normalize_quaternions(left + fraction * (right - left))
    angle = math.acos(dot)
    sin_angle = math.sin(angle)
    return (math.sin((1.0 - fraction) * angle) / sin_angle * left
            + math.sin(fraction * angle) / sin_angle * right)


def _open_ground_truth(path):
    path = Path(path)
    if path.suffix.lower() != ".zip":
        return path.open("r", encoding="utf-8-sig", newline=""), None

    archive = zipfile.ZipFile(path)
    members = [
        name for name in archive.namelist()
        if name.replace("\\", "/").endswith("ground_truth/registered_poses.csv")
    ]
    if len(members) != 1:
        archive.close()
        raise ValueError(
            f"expected exactly one ground_truth/registered_poses.csv in {path}, "
            f"found {len(members)}")
    stream = io.TextIOWrapper(archive.open(members[0]), encoding="utf-8-sig", newline="")
    return stream, archive


def read_newer_college_ground_truth(path):
    """Read registered_poses.csv (or its official zip) as a trajectory."""
    stream, archive = _open_ground_truth(path)
    try:
        reader = csv.reader(stream)
        try:
            header = [name.strip().lstrip("#") for name in next(reader)]
        except StopIteration as exc:
            raise ValueError(f"ground-truth file is empty: {path}") from exc
        required = ("sec", "nsec", "x", "y", "z", "qx", "qy", "qz", "qw")
        missing = [name for name in required if name not in header]
        if missing:
            raise ValueError(f"ground truth is missing columns: {', '.join(missing)}")
        indices = {name: header.index(name) for name in required}
        rows = []
        for line_number, row in enumerate(reader, start=2):
            if not row or not any(value.strip() for value in row):
                continue
            try:
                sec = int(row[indices["sec"]])
                nsec = int(row[indices["nsec"]])
                values = [float(row[indices[name]]) for name in required[2:]]
            except (IndexError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid ground-truth row {line_number} in {path}") from exc
            if not 0 <= nsec < 1_000_000_000:
                raise ValueError(f"invalid nsec at row {line_number}: {nsec}")
            rows.append((sec + nsec * 1e-9, *values))
    finally:
        stream.close()
        if archive is not None:
            archive.close()

    if not rows:
        raise ValueError(f"ground truth contains no poses: {path}")
    values = np.asarray(rows, dtype=np.float64)
    return PoseTrajectory(
        values[:, 0], values[:, 1:4], normalize_quaternions(values[:, 4:8]))


def _pcd_scalar_type(type_code, size):
    types = {
        ("F", 4): "<f4", ("F", 8): "<f8",
        ("I", 1): "<i1", ("I", 2): "<i2", ("I", 4): "<i4", ("I", 8): "<i8",
        ("U", 1): "<u1", ("U", 2): "<u2", ("U", 4): "<u4", ("U", 8): "<u8",
    }
    try:
        return np.dtype(types[(type_code.upper(), size)])
    except KeyError as exc:
        raise ValueError(f"unsupported PCD field type {type_code}{size}") from exc


def read_pcd_fields(path, required_fields):
    """Read selected scalar fields from an ASCII or binary PCD file."""
    path = Path(path)
    header = {}
    with path.open("rb") as stream:
        while True:
            raw_line = stream.readline()
            if not raw_line:
                raise ValueError(f"PCD header has no DATA line: {path}")
            line = raw_line.decode("ascii", errors="strict").strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            header[parts[0].upper()] = parts[1:]
            if parts[0].upper() == "DATA":
                data = stream.read()
                break

    try:
        fields = header["FIELDS"]
        sizes = [int(value) for value in header["SIZE"]]
        types = header["TYPE"]
    except KeyError as exc:
        raise ValueError(f"PCD header is missing {exc.args[0]}: {path}") from exc
    counts = [int(value) for value in header.get("COUNT", ["1"] * len(fields))]
    if not (len(fields) == len(sizes) == len(types) == len(counts)):
        raise ValueError(f"inconsistent PCD field metadata: {path}")
    missing = [name for name in required_fields if name not in fields]
    if missing:
        raise ValueError(f"PCD is missing fields: {', '.join(missing)}")
    for name in required_fields:
        if counts[fields.index(name)] != 1:
            raise ValueError(f"required PCD field {name} is not scalar")

    if "POINTS" in header:
        point_count = int(header["POINTS"][0])
    else:
        point_count = int(header["WIDTH"][0]) * int(header.get("HEIGHT", ["1"])[0])
    data_format = header["DATA"][0].lower()

    if data_format == "ascii":
        column_count = sum(counts)
        values = np.fromstring(data.decode("ascii"), sep=" ", dtype=np.float64)
        if values.size != point_count * column_count:
            raise ValueError(
                f"PCD contains {values.size} values, expected {point_count * column_count}")
        values = values.reshape(point_count, column_count)
        offsets = np.cumsum([0] + counts[:-1])
        return {
            name: values[:, offsets[fields.index(name)]].copy()
            for name in required_fields
        }
    if data_format == "binary":
        formats = []
        for type_code, size, count in zip(types, sizes, counts):
            scalar = _pcd_scalar_type(type_code, size)
            formats.append(scalar if count == 1 else (scalar, (count,)))
        dtype = np.dtype({"names": fields, "formats": formats})
        expected_size = point_count * dtype.itemsize
        if len(data) < expected_size:
            raise ValueError(f"binary PCD data is truncated: {path}")
        records = np.frombuffer(data[:expected_size], dtype=dtype, count=point_count)
        return {
            name: np.asarray(records[name], dtype=np.float64).copy()
            for name in required_fields
        }
    raise ValueError(f"unsupported PCD DATA format: {data_format}")


def read_lio_sam_trajectory(path):
    """Read LIO-SAM transformations.pcd as a timestamped pose trajectory."""
    names = ("x", "y", "z", "roll", "pitch", "yaw", "time")
    fields = read_pcd_fields(path, names)
    positions = np.column_stack([fields[name] for name in ("x", "y", "z")])
    quaternions = euler_rpy_to_quaternion(
        fields["roll"], fields["pitch"], fields["yaw"])
    if len(fields["time"]) > 1 and np.any(np.diff(fields["time"]) <= 0.0):
        raise ValueError(
            "estimated PCD timestamps are not strictly increasing; an ASCII PCD "
            "may have rounded epoch timestamps. Re-run or re-save transformations.pcd "
            "with savePCDFileBinary")
    return PoseTrajectory(fields["time"], positions, quaternions)


def associate_ground_truth(ground_truth, estimated, max_time_gap=0.2):
    """Interpolate ground truth at estimate timestamps within valid GT gaps."""
    if max_time_gap <= 0.0:
        raise ValueError("max_time_gap must be positive")
    times, positions, quaternions, estimate_indices = [], [], [], []
    gt_times = ground_truth.timestamps
    for estimate_index, timestamp in enumerate(estimated.timestamps):
        right = int(np.searchsorted(gt_times, timestamp, side="left"))
        if right < len(gt_times) and abs(gt_times[right] - timestamp) <= 1e-9:
            times.append(timestamp)
            positions.append(ground_truth.positions[right])
            quaternions.append(ground_truth.quaternions[right])
            estimate_indices.append(estimate_index)
            continue
        if right == 0 or right == len(gt_times):
            continue
        left = right - 1
        interval = gt_times[right] - gt_times[left]
        if interval > max_time_gap + 1e-12:
            continue
        fraction = (timestamp - gt_times[left]) / interval
        times.append(timestamp)
        positions.append(
            (1.0 - fraction) * ground_truth.positions[left]
            + fraction * ground_truth.positions[right])
        quaternions.append(slerp(
            ground_truth.quaternions[left], ground_truth.quaternions[right], fraction))
        estimate_indices.append(estimate_index)

    if len(times) < 3:
        raise ValueError(
            f"only {len(times)} estimate poses overlap valid ground-truth intervals; "
            "at least 3 are required")
    matched_ground_truth = PoseTrajectory(
        np.asarray(times), np.asarray(positions), np.asarray(quaternions))
    indices = np.asarray(estimate_indices, dtype=np.int64)
    matched_estimated = PoseTrajectory(
        estimated.timestamps[indices], estimated.positions[indices],
        estimated.quaternions[indices])
    return matched_ground_truth, matched_estimated


def align_se3(reference, estimated):
    """Rigidly align estimated positions to reference positions (no scaling)."""
    source_center = np.mean(estimated.positions, axis=0)
    target_center = np.mean(reference.positions, axis=0)
    source = estimated.positions - source_center
    target = reference.positions - target_center
    u, _, vt = np.linalg.svd(source.T @ target)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ correction @ u.T
    translation = target_center - rotation @ source_center
    positions = (rotation @ estimated.positions.T).T + translation
    alignment_quaternion = rotation_matrix_to_quaternion(rotation)
    quaternions = normalize_quaternions(
        quaternion_multiply(alignment_quaternion, estimated.quaternions))
    return PoseTrajectory(estimated.timestamps, positions, quaternions), rotation, translation


def error_statistics(errors):
    errors = np.asarray(errors, dtype=np.float64)
    if not errors.size:
        raise ValueError("cannot compute statistics for an empty error set")
    return {
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "std": float(np.std(errors)),
        "min": float(np.min(errors)),
        "max": float(np.max(errors)),
    }


def calculate_ate(reference, estimated):
    """Calculate translational absolute trajectory error in metres."""
    if len(reference.timestamps) != len(estimated.timestamps):
        raise ValueError("ATE trajectories must have the same number of poses")
    errors = np.linalg.norm(reference.positions - estimated.positions, axis=1)
    return error_statistics(errors)


def _distance_pairs(reference_positions, delta, tolerance):
    cumulative = np.concatenate((
        [0.0], np.cumsum(np.linalg.norm(np.diff(reference_positions, axis=0), axis=1))))
    pairs = []
    for start in range(len(cumulative) - 1):
        target = cumulative[start] + delta
        insertion = int(np.searchsorted(cumulative, target, side="left"))
        candidates = [
            index for index in (insertion - 1, insertion)
            if start < index < len(cumulative)
        ]
        if not candidates:
            continue
        end = min(candidates, key=lambda index: abs(cumulative[index] - target))
        if abs((cumulative[end] - cumulative[start]) - delta) <= tolerance:
            pairs.append((start, end))
    return pairs


def calculate_rpe(reference, estimated, delta=1.0, tolerance=0.2):
    """Calculate SE(3) relative pose error for reference-distance pairs."""
    if delta <= 0.0 or tolerance < 0.0:
        raise ValueError("RPE delta must be positive and tolerance non-negative")
    if len(reference.timestamps) != len(estimated.timestamps):
        raise ValueError("RPE trajectories must have the same number of poses")
    pairs = _distance_pairs(reference.positions, delta, tolerance)
    if not pairs:
        raise ValueError(
            f"no RPE pose pairs found for delta={delta:g} m and "
            f"tolerance={tolerance:g} m")

    reference_rotations = reference.rotations
    estimated_rotations = estimated.rotations
    translation_errors, rotation_errors = [], []
    for start, end in pairs:
        reference_relative_rotation = (
            reference_rotations[start].T @ reference_rotations[end])
        estimated_relative_rotation = (
            estimated_rotations[start].T @ estimated_rotations[end])
        reference_relative_translation = reference_rotations[start].T @ (
            reference.positions[end] - reference.positions[start])
        estimated_relative_translation = estimated_rotations[start].T @ (
            estimated.positions[end] - estimated.positions[start])
        translation_errors.append(np.linalg.norm(
            reference_relative_rotation.T @ (
                estimated_relative_translation - reference_relative_translation)))
        error_rotation = reference_relative_rotation.T @ estimated_relative_rotation
        cosine = np.clip((np.trace(error_rotation) - 1.0) * 0.5, -1.0, 1.0)
        rotation_errors.append(math.degrees(math.acos(cosine)))

    return {
        "delta_m": float(delta),
        "tolerance_m": float(tolerance),
        "pairs": len(pairs),
        "translation_m": error_statistics(translation_errors),
        "rotation_deg": error_statistics(rotation_errors),
    }


def evaluate_newer_college(
        ground_truth_path, estimated_path, max_time_gap=0.2,
        rpe_delta=1.0, rpe_tolerance=0.2):
    """Compute aligned translational ATE and distance-based SE(3) RPE."""
    ground_truth = read_newer_college_ground_truth(ground_truth_path)
    estimated = read_lio_sam_trajectory(estimated_path)
    reference, matched_estimated = associate_ground_truth(
        ground_truth, estimated, max_time_gap=max_time_gap)
    aligned, rotation, translation = align_se3(reference, matched_estimated)
    return {
        "ground_truth": str(Path(ground_truth_path)),
        "estimated": str(Path(estimated_path)),
        "association": {
            "ground_truth_poses": len(ground_truth.timestamps),
            "estimated_poses": len(estimated.timestamps),
            "matched_poses": len(reference.timestamps),
            "dropped_estimated_poses": len(estimated.timestamps) - len(reference.timestamps),
            "start_time": float(reference.timestamps[0]),
            "end_time": float(reference.timestamps[-1]),
            "max_time_gap_s": float(max_time_gap),
        },
        "alignment": {
            "type": "se3",
            "scale": 1.0,
            "rotation": rotation.tolist(),
            "translation_m": translation.tolist(),
        },
        "ate_translation_m": calculate_ate(reference, aligned),
        "rpe": calculate_rpe(
            reference, aligned, delta=rpe_delta, tolerance=rpe_tolerance),
    }


def format_report(result):
    association = result["association"]
    ate = result["ate_translation_m"]
    rpe = result["rpe"]
    translation = rpe["translation_m"]
    rotation = rpe["rotation_deg"]
    return "\n".join((
        "Newer College trajectory evaluation",
        f"  matched poses: {association['matched_poses']} / "
        f"{association['estimated_poses']} estimated",
        f"  time span: {association['start_time']:.6f} - "
        f"{association['end_time']:.6f} s",
        "  alignment: SE(3), scale fixed at 1",
        f"  ATE translation RMSE: {ate['rmse']:.6f} m",
        f"    mean / median / std / min / max: {ate['mean']:.6f} / "
        f"{ate['median']:.6f} / {ate['std']:.6f} / "
        f"{ate['min']:.6f} / {ate['max']:.6f} m",
        f"  RPE: {rpe['pairs']} pairs at {rpe['delta_m']:g} +/- "
        f"{rpe['tolerance_m']:g} m",
        f"    translation RMSE: {translation['rmse']:.6f} m",
        f"    rotation RMSE: {rotation['rmse']:.6f} deg",
    ))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "ground_truth",
        help="registered_poses.csv or official Newer College zip archive")
    parser.add_argument("estimated", help="LIO-SAM transformations.pcd")
    parser.add_argument(
        "--max-time-gap", type=float, default=0.2, metavar="SECONDS",
        help="maximum ground-truth interval allowed for interpolation (default: 0.2)")
    parser.add_argument(
        "--rpe-delta", type=float, default=1.0, metavar="METRES",
        help="reference path distance between RPE pose pairs (default: 1.0)")
    parser.add_argument(
        "--rpe-tolerance", type=float, default=0.2, metavar="METRES",
        help="allowed RPE pair distance error (default: 0.2)")
    parser.add_argument(
        "--output-json", type=Path,
        help="also write the full machine-readable result to this file")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        result = evaluate_newer_college(
            args.ground_truth, args.estimated,
            max_time_gap=args.max_time_gap,
            rpe_delta=args.rpe_delta,
            rpe_tolerance=args.rpe_tolerance)
        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(
                json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(format_report(result))
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
