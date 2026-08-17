#!/usr/bin/env python3
"""Report continuity metrics for an Odometry topic recorded in a ROS 2 bag."""

import argparse
from pathlib import Path

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def stamp_seconds(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def analyze(bag_path, topic_name):
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_path), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    topic_types = {
        topic.name: get_message(topic.type)
        for topic in reader.get_all_topics_and_types()
    }
    if topic_name not in topic_types:
        raise RuntimeError(f"Topic {topic_name!r} is not present in {bag_path}")

    stamps = []
    positions = []
    quaternions = []
    quaternion_norms = []
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic != topic_name:
            continue
        message = deserialize_message(data, topic_types[topic])
        stamps.append(stamp_seconds(message.header.stamp))
        positions.append([
            message.pose.pose.position.x,
            message.pose.pose.position.y,
            message.pose.pose.position.z,
        ])
        orientation = message.pose.pose.orientation
        quaternion = [
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w,
        ]
        quaternions.append(quaternion)
        quaternion_norms.append(np.linalg.norm(quaternion))

    if len(stamps) < 2:
        raise RuntimeError(f"Need at least two messages on {topic_name!r}")

    stamps = np.asarray(stamps)
    positions = np.asarray(positions)
    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    dts = np.diff(stamps)
    valid_dt = dts > 0.0
    speeds = steps[valid_dt] / dts[valid_dt]
    displacement = np.linalg.norm(positions[-1] - positions[0])

    print(f"topic: {topic_name}")
    print(f"messages: {len(stamps)}")
    print(f"header duration [s]: {stamps[-1] - stamps[0]:.6f}")
    print(f"path length [m]: {steps.sum():.6f}")
    print(f"start-to-end displacement [m]: {displacement:.6f}")
    print(f"position min xyz [m]: {positions.min(axis=0)}")
    print(f"position max xyz [m]: {positions.max(axis=0)}")
    print(f"median/max step [m]: {np.median(steps):.6f} / {steps.max():.6f}")
    print(f"median/max speed [m/s]: {np.median(speeds):.6f} / {speeds.max():.6f}")
    print(
        "quaternion norm min/max: "
        f"{np.min(quaternion_norms):.9f} / {np.max(quaternion_norms):.9f}"
    )
    return stamps, positions, np.asarray(quaternions)


def evaluate_ground_truth(stamps, positions, quaternions, ground_truth_path):
    from evaluate_newer_college import (
        PoseTrajectory,
        align_se3,
        associate_ground_truth,
        calculate_ate,
        calculate_rpe,
        read_newer_college_ground_truth,
    )

    estimated = PoseTrajectory(stamps, positions, quaternions)
    ground_truth = read_newer_college_ground_truth(ground_truth_path)
    reference, matched = associate_ground_truth(ground_truth, estimated)
    aligned, _, _ = align_se3(reference, matched)
    ate = calculate_ate(reference, aligned)
    rpe = calculate_rpe(reference, aligned)
    print(f"matched ground-truth poses: {len(reference.timestamps)}")
    print(f"ATE RMSE/median/max [m]: {ate['rmse']:.6f} / {ate['median']:.6f} / {ate['max']:.6f}")
    print(
        "RPE translation RMSE [m] / rotation RMSE [deg] / pairs: "
        f"{rpe['translation_m']['rmse']:.6f} / "
        f"{rpe['rotation_deg']['rmse']:.6f} / {rpe['pairs']}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bag", type=Path)
    parser.add_argument(
        "--topic",
        default="/dsw_lio_sam/mapping/odometry",
    )
    parser.add_argument("--ground-truth", type=Path)
    args = parser.parse_args()
    stamps, positions, quaternions = analyze(args.bag, args.topic)
    if args.ground_truth:
        evaluate_ground_truth(
            stamps, positions, quaternions, args.ground_truth)


if __name__ == "__main__":
    main()
