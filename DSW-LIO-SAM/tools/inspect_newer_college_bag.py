#!/usr/bin/env python3
"""Inspect the IMU and Ouster timing fields in a Newer College ROS 2 bag."""

import argparse
from pathlib import Path

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def stamp_seconds(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def describe(values):
    array = np.asarray(values, dtype=np.float64)
    return (
        f"min={array.min():.9g}, median={np.median(array):.9g}, "
        f"mean={array.mean():.9g}, max={array.max():.9g}"
    )


def strided_field(message, name):
    field = next(field for field in message.fields if field.name == name)
    dtype_by_datatype = {
        1: "i1",
        2: "u1",
        3: "<i2",
        4: "<u2",
        5: "<i4",
        6: "<u4",
        7: "<f4",
        8: "<f8",
    }
    dtype = np.dtype(dtype_by_datatype[field.datatype])
    if message.is_bigendian:
        dtype = dtype.newbyteorder(">")
    return np.ndarray(
        shape=(message.width * message.height,),
        dtype=dtype,
        buffer=message.data,
        offset=field.offset,
        strides=(message.point_step,),
    )


def inspect_bag(bag_path, imu_limit, cloud_limit):
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_path), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    topic_types = {
        topic.name: get_message(topic.type)
        for topic in reader.get_all_topics_and_types()
    }

    imu_acc = []
    imu_gyro = []
    imu_qnorm = []
    imu_stamps = []
    imu_stamp_offsets = []
    accel_covariance = None
    orientation_covariance = None
    cloud_summaries = []

    while reader.has_next():
        topic, data, bag_stamp = reader.read_next()
        if topic == "/os1_cloud_node/imu" and len(imu_acc) < imu_limit:
            message = deserialize_message(data, topic_types[topic])
            acceleration = np.array([
                message.linear_acceleration.x,
                message.linear_acceleration.y,
                message.linear_acceleration.z,
            ])
            angular_velocity = np.array([
                message.angular_velocity.x,
                message.angular_velocity.y,
                message.angular_velocity.z,
            ])
            orientation = np.array([
                message.orientation.w,
                message.orientation.x,
                message.orientation.y,
                message.orientation.z,
            ])
            imu_acc.append(acceleration)
            imu_gyro.append(angular_velocity)
            imu_qnorm.append(np.linalg.norm(orientation))
            imu_stamps.append(stamp_seconds(message.header.stamp))
            imu_stamp_offsets.append(stamp_seconds(message.header.stamp) - bag_stamp * 1e-9)
            accel_covariance = message.linear_acceleration_covariance
            orientation_covariance = message.orientation_covariance
        elif topic == "/os1_cloud_node/points" and len(cloud_summaries) < cloud_limit:
            message = deserialize_message(data, topic_types[topic])
            point_times = strided_field(message, "t")
            rings = strided_field(message, "ring")
            cloud_summaries.append({
                "bag_stamp": bag_stamp * 1e-9,
                "header_stamp": stamp_seconds(message.header.stamp),
                "height": message.height,
                "width": message.width,
                "point_step": message.point_step,
                "t_min": int(point_times.min()),
                "t_max": int(point_times.max()),
                "ring_min": int(rings.min()),
                "ring_max": int(rings.max()),
            })

        if len(imu_acc) >= imu_limit and len(cloud_summaries) >= cloud_limit:
            break

    if not imu_acc or not cloud_summaries:
        raise RuntimeError("Expected Newer College IMU and point-cloud topics were not found")

    imu_acc = np.asarray(imu_acc)
    imu_gyro = np.asarray(imu_gyro)
    imu_dts = np.diff(imu_stamps)
    print(f"IMU samples: {len(imu_acc)}")
    print(f"IMU dt [s]: {describe(imu_dts)}")
    print(f"IMU header-bag offset [s]: {describe(imu_stamp_offsets)}")
    print(f"acceleration xyz mean [m/s^2]: {imu_acc.mean(axis=0)}")
    print(f"acceleration norm [m/s^2]: {describe(np.linalg.norm(imu_acc, axis=1))}")
    print(f"angular velocity xyz mean [rad/s]: {imu_gyro.mean(axis=0)}")
    print(f"angular velocity norm [rad/s]: {describe(np.linalg.norm(imu_gyro, axis=1))}")
    print(f"orientation quaternion norm: {describe(imu_qnorm)}")
    print(f"linear acceleration covariance: {list(accel_covariance)}")
    print(f"orientation covariance: {list(orientation_covariance)}")
    print("Point clouds:")
    for index, summary in enumerate(cloud_summaries):
        scan_duration = (summary["t_max"] - summary["t_min"]) * 1e-9
        stamp_offset = summary["header_stamp"] - summary["bag_stamp"]
        print(
            f"  {index}: {summary['height']}x{summary['width']}, "
            f"point_step={summary['point_step']}, rings="
            f"{summary['ring_min']}..{summary['ring_max']}, t="
            f"{summary['t_min']}..{summary['t_max']} ns "
            f"({scan_duration:.6f} s), header-bag={stamp_offset:+.9f} s"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bag", type=Path)
    parser.add_argument("--imu-limit", type=int, default=500)
    parser.add_argument("--cloud-limit", type=int, default=10)
    args = parser.parse_args()
    inspect_bag(args.bag, args.imu_limit, args.cloud_limit)


if __name__ == "__main__":
    main()
