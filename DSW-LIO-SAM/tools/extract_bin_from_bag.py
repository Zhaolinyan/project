#!/usr/bin/env python3
"""
Extract PointCloud2 messages from a ROS2 bag and save Cylinder3D XYZI .bin files.

The important detail is PointCloud2 field layout: fields are interleaved per point,
so each field must be read with point_step stride. Reading from field.offset as one
contiguous float array corrupts x/y/z/intensity.
"""

import os
import platform
import sys

import numpy as np


_IS_WINDOWS = platform.system() == "Windows"

if _IS_WINDOWS:
    BAG_DIR = r"E:\Code_reproduction\DSW-LIO-SAM\datasets\rooster_2020-07-10-09-19-26_2-001_ros2"
    OUTPUT_DIR = r"E:\Code_reproduction\DSW-LIO-SAM\datasets\nc_bin_test"
else:
    BAG_DIR = "/mnt/e/Code_reproduction/DSW-LIO-SAM/datasets/rooster_2020-07-10-09-19-26_2-001_ros2"
    OUTPUT_DIR = "/mnt/e/Code_reproduction/DSW-LIO-SAM/datasets/rooster_bin_001"

if len(sys.argv) > 1:
    BAG_DIR = sys.argv[1]
if len(sys.argv) > 2:
    OUTPUT_DIR = sys.argv[2]

MAX_FRAMES = 0
TOPIC_NAME = "/os1_cloud_node/points"

# sensor_msgs/msg/PointField datatype ids
POINT_FIELD_UINT8 = 2
POINT_FIELD_UINT16 = 4
POINT_FIELD_FLOAT32 = 7


def _clean_name(name):
    return name.strip("\x00").strip()


def _point_rows(msg):
    num_points = msg.width * msg.height
    point_step = msg.point_step
    data = msg.data
    if not isinstance(data, (bytes, bytearray, memoryview)):
        data = bytes(data)
    raw = np.frombuffer(data, dtype=np.uint8, count=num_points * point_step)
    return raw.reshape(num_points, point_step)


def _field_by_name(msg, names):
    names = set(names)
    for field in msg.fields:
        if _clean_name(field.name) in names:
            return field
    return None


def _read_strided_field(rows, field, dtype, byte_width):
    values = rows[:, field.offset:field.offset + byte_width].copy()
    return values.view(dtype).reshape(-1)


def _read_float32_field(rows, field):
    if field.datatype != POINT_FIELD_FLOAT32:
        raise ValueError(f"field {field.name!r} is not FLOAT32")
    return _read_strided_field(rows, field, "<f4", 4).astype(np.float32, copy=False)


def _read_intensity(rows, field, num_points):
    if field is None:
        return np.zeros(num_points, dtype=np.float32)
    if field.datatype == POINT_FIELD_FLOAT32:
        return _read_strided_field(rows, field, "<f4", 4).astype(np.float32, copy=False)
    if field.datatype == POINT_FIELD_UINT16:
        return _read_strided_field(rows, field, "<u2", 2).astype(np.float32)
    if field.datatype == POINT_FIELD_UINT8:
        return _read_strided_field(rows, field, np.uint8, 1).astype(np.float32)
    raise ValueError(f"unsupported intensity datatype {field.datatype}")


def parse_pointcloud_xyzi(msg):
    num_points = msg.width * msg.height
    if num_points == 0:
        return None, "empty PointCloud2"

    x_field = _field_by_name(msg, ["x"])
    y_field = _field_by_name(msg, ["y"])
    z_field = _field_by_name(msg, ["z"])
    i_field = _field_by_name(msg, ["intensity", "signal", "reflectivity"])
    if x_field is None or y_field is None or z_field is None:
        names = [_clean_name(f.name) for f in msg.fields]
        return None, f"missing xyz fields; available={names}"

    rows = _point_rows(msg)
    xyz_full = np.zeros((num_points, 3), dtype=np.float32)
    xyz_full[:, 0] = _read_float32_field(rows, x_field)
    xyz_full[:, 1] = _read_float32_field(rows, y_field)
    xyz_full[:, 2] = _read_float32_field(rows, z_field)
    intensity_full = _read_intensity(rows, i_field, num_points)

    valid = np.isfinite(xyz_full).all(axis=1)
    valid &= np.linalg.norm(xyz_full, axis=1) > 1e-3
    xyz = xyz_full[valid]
    intensity = intensity_full[valid]
    if len(xyz) == 0:
        return None, f"no valid points after filtering original={num_points}"

    if intensity.max() > 1.0:
        intensity = intensity / max(float(intensity.max()), 1e-6)

    xyzi = np.zeros((len(xyz), 4), dtype=np.float32)
    xyzi[:, :3] = xyz
    xyzi[:, 3] = intensity

    diag = {
        "xyzi": xyzi,
        "xyz_full": xyz_full,
        "intensity_full": intensity_full,
        "valid": valid,
        "point_step": msg.point_step,
        "fields": msg.fields,
    }
    return diag, None


def print_diagnosis(diag):
    xyzi = diag["xyzi"]
    xyz = xyzi[:, :3]
    intensity = xyzi[:, 3]
    valid = diag["valid"]
    raw_intensity = diag["intensity_full"][valid]
    fields = [(_clean_name(f.name), f.offset, f.datatype) for f in diag["fields"]]

    print("\n" + "=" * 60)
    print("  PointCloud2 parse diagnosis")
    print("=" * 60)
    print(f"  original points: {len(diag['xyz_full'])}")
    print(f"  valid points:    {len(xyzi)}")
    print(f"  point_step:      {diag['point_step']}")
    print(f"  fields:          {fields}")
    print(f"  x range:         [{xyz[:, 0].min():.2f}, {xyz[:, 0].max():.2f}]")
    print(f"  y range:         [{xyz[:, 1].min():.2f}, {xyz[:, 1].max():.2f}]")
    print(f"  z range:         [{xyz[:, 2].min():.2f}, {xyz[:, 2].max():.2f}]")
    print(f"  raw intensity:   [{raw_intensity.min():.4f}, {raw_intensity.max():.4f}]")
    print(f"  norm intensity:  [{intensity.min():.4f}, {intensity.max():.4f}]")
    print("=" * 60 + "\n")


def save_cloud_msg(msg, count):
    diag, error = parse_pointcloud_xyzi(msg)
    if error is not None:
        return None, error, None

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"{count:06d}.bin")
    diag["xyzi"].tofile(out_path)
    return diag["xyzi"], out_path, diag


def extract_with_rosbag2_py():
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=BAG_DIR, storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    reader.open(storage_options, converter_options)

    topic_type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if TOPIC_NAME not in topic_type_map:
        print(f"[ERROR] topic {TOPIC_NAME!r} not found in bag")
        print(f"[INFO] available topics: {list(topic_type_map.keys())}")
        return False

    msg_class = get_message(topic_type_map[TOPIC_NAME])
    reader.set_filter(rosbag2_py.StorageFilter(topics=[TOPIC_NAME]))

    count = 0
    total_skipped = 0
    while reader.has_next():
        _, msg_data, _ = reader.read_next()
        try:
            msg = deserialize_message(msg_data, msg_class)
            xyzi, info, diag = save_cloud_msg(msg, count)
            if xyzi is None:
                total_skipped += 1
                if total_skipped <= 3:
                    print(f"  [SKIP] {info}")
                continue

            if count == 0:
                print_diagnosis(diag)
            print(f"  [{count:06d}] {len(xyzi):7d} points -> {info}")
            count += 1

            if MAX_FRAMES > 0 and count >= MAX_FRAMES:
                break
        except Exception as exc:
            total_skipped += 1
            if total_skipped <= 3:
                print(f"  [SKIP] parse failed: {exc}")

    try:
        reader.close()
    except AttributeError:
        pass

    if total_skipped:
        print(f"\n[WARN] skipped {total_skipped} frames")
    print(f"\nDone. Saved {count} frames to {OUTPUT_DIR}")
    return True


def extract_with_ros2_subscriber():
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import PointCloud2

    class BagExtractor(Node):
        def __init__(self):
            super().__init__("bin_extractor")
            self.sub = self.create_subscription(PointCloud2, TOPIC_NAME, self.callback, 10)
            self.count = 0
            self.last_msg_time = self.get_clock().now()
            self.shutdown_timer = self.create_timer(5.0, self.check_shutdown)
            self.get_logger().info(f"Waiting for {TOPIC_NAME} messages...")

        def callback(self, msg):
            self.last_msg_time = self.get_clock().now()
            if MAX_FRAMES > 0 and self.count >= MAX_FRAMES:
                raise SystemExit(0)

            xyzi, info, diag = save_cloud_msg(msg, self.count)
            if xyzi is None:
                if self.count < 20:
                    self.get_logger().warn(f"frame skipped: {info}")
                return

            if self.count == 0:
                print_diagnosis(diag)
            print(f"  [{self.count:06d}] {len(xyzi):7d} points -> {info}")
            self.count += 1

        def check_shutdown(self):
            elapsed = (self.get_clock().now() - self.last_msg_time).nanoseconds * 1e-9
            if elapsed > 5.0 and self.count > 0:
                self.get_logger().info(f"bag playback ended; extracted {self.count} frames")
                raise SystemExit(0)

    rclpy.init()
    node = BagExtractor()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        rclpy.shutdown()


def main():
    print("=" * 60)
    print("  ROS2 Bag -> Cylinder3D XYZI extractor")
    print("=" * 60)
    print(f"[INFO] bag:        {BAG_DIR}")
    print(f"[INFO] output:     {OUTPUT_DIR}")
    print(f"[INFO] topic:      {TOPIC_NAME}")
    print(f"[INFO] max frames: {MAX_FRAMES if MAX_FRAMES > 0 else 'all'}")

    try:
        import rosbag2_py  # noqa: F401
        print("\n[MODE] rosbag2_py direct reader")
        if extract_with_rosbag2_py():
            return
    except ImportError:
        print("[WARN] rosbag2_py unavailable; falling back to subscriber mode")
    except Exception as exc:
        print(f"[WARN] rosbag2_py mode failed: {exc}")

    print("\n[MODE] rclpy subscriber fallback")
    print(f"Run in another terminal: ros2 bag play {BAG_DIR}")
    extract_with_ros2_subscriber()


if __name__ == "__main__":
    main()
