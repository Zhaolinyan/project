#!/usr/bin/env python3
"""Compare the four DSW-LIO-SAM experiments with Newer College ground truth."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from evaluate_newer_college import (
    align_se3,
    associate_ground_truth,
    calculate_ate,
    calculate_rpe,
    read_lio_sam_trajectory,
    read_newer_college_ground_truth,
)


DEFAULT_RESULTS_ROOT = Path.home() / "dsw_lio_sam_results"
DEFAULT_TRAJECTORIES = {
    "original": "original_lio_sam_rooster_001/transformations.pcd",
    "baseline": "baseline_no_semantic_rooster_001/transformations.pcd",
    "pseudo": "pseudo_semantic_rooster_001/transformations.pcd",
    "true": "cylinder3d_true_semantic_rooster_001/transformations.pcd",
}


def evaluate_experiment(ground_truth, estimated_path, max_time_gap=0.2,
                        rpe_delta=1.0, rpe_tolerance=0.2):
    estimated = read_lio_sam_trajectory(estimated_path)
    reference, matched = associate_ground_truth(
        ground_truth, estimated, max_time_gap=max_time_gap)
    aligned, rotation, translation = align_se3(reference, matched)

    estimated_displacement = float(np.linalg.norm(
        matched.positions[-1] - matched.positions[0]))
    ground_truth_displacement = float(np.linalg.norm(
        reference.positions[-1] - reference.positions[0]))
    start_error = float(np.linalg.norm(
        aligned.positions[0] - reference.positions[0]))
    end_error = float(np.linalg.norm(
        aligned.positions[-1] - reference.positions[-1]))
    relative_displacement_error = float(np.linalg.norm(
        (aligned.positions[-1] - aligned.positions[0])
        - (reference.positions[-1] - reference.positions[0])))

    return {
        "estimated": str(Path(estimated_path)),
        "association": {
            "estimated_poses": len(estimated.timestamps),
            "matched_poses": len(reference.timestamps),
            "dropped_estimated_poses": (
                len(estimated.timestamps) - len(reference.timestamps)),
            "start_time": float(reference.timestamps[0]),
            "end_time": float(reference.timestamps[-1]),
        },
        "alignment": {
            "type": "se3",
            "scale": 1.0,
            "rotation": rotation.tolist(),
            "translation_m": translation.tolist(),
        },
        "endpoint": {
            "estimated_start_to_end_m": estimated_displacement,
            "ground_truth_start_to_end_m": ground_truth_displacement,
            "start_to_end_distance_error_m": abs(
                estimated_displacement - ground_truth_displacement),
            "aligned_start_error_m": start_error,
            "aligned_end_error_m": end_error,
            "relative_displacement_error_m": relative_displacement_error,
        },
        "ate_translation_m": calculate_ate(reference, aligned),
        "rpe": calculate_rpe(
            reference, aligned, delta=rpe_delta, tolerance=rpe_tolerance),
    }


def flatten_result(name, result):
    association = result["association"]
    endpoint = result["endpoint"]
    ate = result["ate_translation_m"]
    rpe = result["rpe"]
    return {
        "experiment": name,
        "matched_poses": association["matched_poses"],
        "estimated_poses": association["estimated_poses"],
        "estimated_start_to_end_m": endpoint["estimated_start_to_end_m"],
        "ground_truth_start_to_end_m": endpoint["ground_truth_start_to_end_m"],
        "start_to_end_distance_error_m": endpoint[
            "start_to_end_distance_error_m"],
        "aligned_start_error_m": endpoint["aligned_start_error_m"],
        "aligned_end_error_m": endpoint["aligned_end_error_m"],
        "relative_displacement_error_m": endpoint[
            "relative_displacement_error_m"],
        "ate_rmse_m": ate["rmse"],
        "ate_max_m": ate["max"],
        "rpe_pairs": rpe["pairs"],
        "rpe_translation_rmse_m": rpe["translation_m"]["rmse"],
        "rpe_rotation_rmse_deg": rpe["rotation_deg"]["rmse"],
    }


def format_table(rows):
    headers = (
        "experiment", "matched", "est_end(m)", "gt_end(m)", "end_diff(m)",
        "ATE_RMSE(m)", "RPE_t(m)", "RPE_r(deg)")
    values = [headers]
    for row in rows:
        values.append((
            row["experiment"],
            f'{row["matched_poses"]}/{row["estimated_poses"]}',
            f'{row["estimated_start_to_end_m"]:.3f}',
            f'{row["ground_truth_start_to_end_m"]:.3f}',
            f'{row["start_to_end_distance_error_m"]:.3f}',
            f'{row["ate_rmse_m"]:.4f}',
            f'{row["rpe_translation_rmse_m"]:.4f}',
            f'{row["rpe_rotation_rmse_deg"]:.3f}',
        ))
    widths = [max(len(str(row[i])) for row in values) for i in range(len(headers))]
    return "\n".join(
        "  ".join(str(value).ljust(widths[i]) for i, value in enumerate(row))
        for row in values)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "ground_truth",
        help="registered_poses.csv or official Newer College zip archive")
    parser.add_argument(
        "--results-root", type=Path, default=DEFAULT_RESULTS_ROOT,
        help=f"experiment results root (default: {DEFAULT_RESULTS_ROOT})")
    parser.add_argument("--max-time-gap", type=float, default=0.2)
    parser.add_argument("--rpe-delta", type=float, default=1.0)
    parser.add_argument("--rpe-tolerance", type=float, default=0.2)
    parser.add_argument(
        "--output-json", type=Path,
        help="write detailed results as JSON")
    parser.add_argument(
        "--output-csv", type=Path,
        help="write the comparison table as UTF-8 CSV")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        ground_truth = read_newer_college_ground_truth(args.ground_truth)
        detailed = {}
        rows = []
        for name, relative_path in DEFAULT_TRAJECTORIES.items():
            path = args.results_root / relative_path
            result = evaluate_experiment(
                ground_truth, path,
                max_time_gap=args.max_time_gap,
                rpe_delta=args.rpe_delta,
                rpe_tolerance=args.rpe_tolerance)
            detailed[name] = result
            rows.append(flatten_result(name, result))

        print(format_table(rows))
        print("\nEndpoint details after SE(3) alignment:")
        for row in rows:
            print(
                f'  {row["experiment"]}: start={row["aligned_start_error_m"]:.3f} m, '
                f'end={row["aligned_end_error_m"]:.3f} m, '
                f'relative displacement error='
                f'{row["relative_displacement_error_m"]:.3f} m')

        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(
                json.dumps({
                    "ground_truth": str(Path(args.ground_truth)),
                    "rpe_delta_m": args.rpe_delta,
                    "rpe_tolerance_m": args.rpe_tolerance,
                    "experiments": detailed,
                }, indent=2) + "\n", encoding="utf-8")
        if args.output_csv:
            args.output_csv.parent.mkdir(parents=True, exist_ok=True)
            with args.output_csv.open("w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
