#!/usr/bin/env python3
"""Replay precomputed semantic frames on /semantic_cloud."""

import os

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField

from semantic_replay_utils import (
    SEMANTIC_DTYPE,
    align_semantic_frame,
    find_frame_for_stamp,
    load_manifest,
    stamp_to_ns,
    validate_inference_metadata,
)


class SemanticReplayNode(Node):
    def __init__(self):
        super().__init__("semantic_replay_node")

        self.declare_parameter("semantic_dir", "")
        self.declare_parameter("input_topic", "/os1_cloud_node/points")
        self.declare_parameter("minimum_match_coverage", 0.95)
        self.declare_parameter("stamp_tolerance_sec", 0.001)

        semantic_dir = self.get_parameter("semantic_dir").get_parameter_value().string_value
        self.input_topic = self.get_parameter("input_topic").get_parameter_value().string_value
        self.minimum_match_coverage = (
            self.get_parameter("minimum_match_coverage").get_parameter_value().double_value
        )
        stamp_tolerance_sec = (
            self.get_parameter("stamp_tolerance_sec").get_parameter_value().double_value
        )
        self.stamp_tolerance_ns = int(stamp_tolerance_sec * 1_000_000_000)

        if not semantic_dir:
            self.get_logger().error(
                "Missing semantic_dir parameter; use --ros-args -p semantic_dir:=/path/to/semantic"
            )
            raise SystemExit(1)
        if not os.path.isdir(semantic_dir):
            self.get_logger().error(f"semantic_dir does not exist: {semantic_dir}")
            raise SystemExit(1)

        try:
            metadata = validate_inference_metadata(semantic_dir)
            self.frame_index = load_manifest(semantic_dir)
        except (FileNotFoundError, ValueError) as exc:
            self.get_logger().error(f"Invalid semantic directory: {exc}")
            raise SystemExit(1)

        self.semantic_dir = semantic_dir
        self.sorted_stamps = sorted(self.frame_index)
        self.total_frames = len(self.sorted_stamps)
        self.published_count = 0
        self.received_count = 0
        self.logged_first_input = False

        model_kind = metadata.get("model_kind", "unknown")
        profile = metadata.get("preprocessing_profile", "unknown")
        self.get_logger().info(
            f"Indexed {self.total_frames} semantic frames, model={model_kind}, "
            f"profile={profile}, input_topic={self.input_topic}, "
            f"stamp_tolerance={stamp_tolerance_sec:.3f}s, "
            f"minimum_match_coverage={self.minimum_match_coverage:.2f}"
        )

        input_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.sub = self.create_subscription(PointCloud2, self.input_topic, self.callback, input_qos)
        self.pub = self.create_publisher(PointCloud2, "/semantic_cloud", 10)

        self.last_msg_time = self.get_clock().now()
        self.shutdown_timer = self.create_timer(5.0, self.check_shutdown)

    def callback(self, msg: PointCloud2):
        self.last_msg_time = self.get_clock().now()
        self.received_count += 1
        if not self.logged_first_input:
            self.get_logger().info(f"Received first lidar cloud on {self.input_topic}")
            self.logged_first_input = True

        stamp_ns = stamp_to_ns(msg.header.stamp)
        matched_stamp, entry = find_frame_for_stamp(
            self.frame_index, self.sorted_stamps, stamp_ns, self.stamp_tolerance_ns
        )
        if entry is None:
            self.get_logger().warn(
                f"No semantic frame for lidar stamp={stamp_ns}, "
                f"tolerance_ns={self.stamp_tolerance_ns}"
            )
            return

        semantic_path = entry["path"]
        semantic_frame = np.fromfile(semantic_path, dtype=SEMANTIC_DTYPE)
        if len(semantic_frame) != entry["point_count"]:
            self.get_logger().error(
                f"Semantic frame size differs from manifest: "
                f"{os.path.basename(semantic_path)} actual={len(semantic_frame)} "
                f"expected={entry['point_count']}"
            )
            return

        try:
            aligned = align_semantic_frame(
                msg,
                semantic_frame,
                allow_coordinate_lookup=True,
                minimum_coverage=self.minimum_match_coverage,
            )
        except ValueError as exc:
            self.get_logger().error(
                f"Semantic frame {os.path.basename(semantic_path)} does not match lidar cloud: {exc}"
            )
            return

        n = aligned.shape[0]
        out = PointCloud2()
        out.header = msg.header
        out.height, out.width = 1, n
        out.is_bigendian, out.is_dense = False, True
        out.point_step, out.row_step = 20, 20 * n
        out.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name="label", offset=16, datatype=PointField.UINT32, count=1),
        ]
        out.data = aligned.tobytes()
        self.pub.publish(out)
        self.published_count += 1

        if self.published_count % 100 == 0 or self.published_count == self.total_frames:
            diff_ms = abs(matched_stamp - stamp_ns) / 1_000_000.0
            self.get_logger().info(
                f"Published {self.published_count}/{self.total_frames}, "
                f"stamp_diff={diff_ms:.3f}ms"
            )

    def check_shutdown(self):
        if self.received_count == 0 or self.published_count < self.total_frames:
            return
        elapsed = (self.get_clock().now() - self.last_msg_time).nanoseconds * 1e-9
        if elapsed > 5.0:
            self.get_logger().info(
                f"Replay complete: published {self.published_count}/{self.total_frames} frames"
            )
            raise SystemExit(0)


def main():
    rclpy.init()
    rclpy.spin(SemanticReplayNode())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
