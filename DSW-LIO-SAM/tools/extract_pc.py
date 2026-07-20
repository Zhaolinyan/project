"""Extract Ouster PointCloud2 frames from ROS1/ROS2 bags for Cylinder3D demo input."""

import argparse
import os
import sqlite3
from pathlib import Path

import numpy as np
from rosbags.rosbag1 import Reader as Rosbag1Reader
from rosbags.typesys import Stores, get_typestore


DEFAULT_BAG_PATH = "/mnt/e/Code_reproduction/DSW-LIO-SAM/datasets/rooster_2020-07-10-09-19-26_2-001_ros2"
DEFAULT_TOPIC = "/os1_cloud_node/points"
DEFAULT_OUTPUT_DIR = "/home/zly/Cylinder3D/demo_data"
DEFAULT_MAX_FRAMES = 5

FLOAT32 = 7


def pointcloud2_to_xyzi(msg, normalize_intensity=True, drop_invalid=True):
    """Convert ROS PointCloud2 data to Cylinder3D's N x 4 float32 bin format."""
    field_by_name = {field.name: field for field in msg.fields}
    missing = [name for name in ("x", "y", "z") if name not in field_by_name]
    if missing:
        raise ValueError(f"PointCloud2 missing required fields: {missing}")

    intensity_name = next(
        (name for name in ("intensity", "signal", "reflectivity") if name in field_by_name),
        None,
    )

    names = ["x", "y", "z"]
    if intensity_name:
        names.append(intensity_name)

    endian = ">" if msg.is_bigendian else "<"
    dtype = np.dtype(
        {
            "names": names,
            "formats": [endian + "f4"] * len(names),
            "offsets": [field_by_name[name].offset for name in names],
            "itemsize": msg.point_step,
        }
    )

    for name in names:
        field = field_by_name[name]
        if field.datatype != FLOAT32:
            raise ValueError(
                f"Field '{name}' is datatype {field.datatype}, expected FLOAT32 ({FLOAT32})"
            )

    num_points = int(msg.height) * int(msg.width)
    points = np.frombuffer(msg.data, dtype=dtype, count=num_points)

    xyzi = np.zeros((num_points, 4), dtype=np.float32)
    xyzi[:, 0] = points["x"]
    xyzi[:, 1] = points["y"]
    xyzi[:, 2] = points["z"]
    if intensity_name:
        xyzi[:, 3] = points[intensity_name]

    if drop_invalid:
        valid = np.isfinite(xyzi).all(axis=1)
        valid &= np.linalg.norm(xyzi[:, :3], axis=1) > 1e-3
        xyzi = xyzi[valid]

    if normalize_intensity and xyzi.size:
        max_intensity = float(np.max(xyzi[:, 3]))
        if max_intensity > 1.0:
            xyzi[:, 3] /= max_intensity
    return xyzi

def find_ros2_db3(bag_path):
    bag_path = Path(bag_path)
    if bag_path.is_file() and bag_path.suffix == ".db3":
        return bag_path
    if bag_path.is_dir():
        db3_files = sorted(bag_path.glob("*.db3"))
        if db3_files:
            return db3_files[0]
    raise RuntimeError(f"Could not find a rosbag2 .db3 file under {bag_path}")


def extract_ros2_frames(bag_path, topic, output_dir, max_frames, normalize_intensity, drop_invalid):
    os.makedirs(output_dir, exist_ok=True)

    db3_path = find_ros2_db3(bag_path)
    typestore = get_typestore(Stores.ROS2_HUMBLE)

    with sqlite3.connect(db3_path) as con:
        cur = con.cursor()

        topic_row = cur.execute(
            "select id, type, serialization_format from topics where name = ?",
            (topic,),
        ).fetchone()

        if topic_row is None:
            raise RuntimeError(f"Topic not found: {topic}")

        topic_id, msgtype, serialization_format = topic_row

        if serialization_format != "cdr":
            raise RuntimeError(f"Expected cdr, got {serialization_format}")

        count = 0
        for (rawdata,) in cur.execute(
            "select data from messages where topic_id = ? order by timestamp",
            (topic_id,),
        ):
            msg = typestore.deserialize_cdr(rawdata, msgtype)

            pc = pointcloud2_to_xyzi(
                msg,
                normalize_intensity=normalize_intensity,
                drop_invalid=drop_invalid,
            )

            out_path = os.path.join(output_dir, f"{count:06d}.bin")
            pc.astype(np.float32).tofile(out_path)

            print(f"[{count}] saved {pc.shape[0]} points -> {out_path}")

            count += 1
            if max_frames > 0 and count >= max_frames:
                break

    print(f"Done. Extracted {count} ROS2 frames to {output_dir}")

def write_xyzi_frame(xyzi, output_dir, count):
    out_path = os.path.join(output_dir, f"{count:06d}.bin")
    xyzi.tofile(out_path)
    print(
        f"[{count}] saved {xyzi.shape[0]} points -> {out_path} "
        f"x=[{xyzi[:, 0].min():.2f},{xyzi[:, 0].max():.2f}] "
        f"y=[{xyzi[:, 1].min():.2f},{xyzi[:, 1].max():.2f}] "
        f"z=[{xyzi[:, 2].min():.2f},{xyzi[:, 2].max():.2f}] "
        f"i=[{xyzi[:, 3].min():.2f},{xyzi[:, 3].max():.2f}]"
    )


def find_ros2_db3(bag_path):
    bag_path = Path(bag_path)
    if bag_path.is_file() and bag_path.suffix == ".db3":
        return bag_path
    if bag_path.is_dir():
        db3_files = sorted(bag_path.glob("*.db3"))
        if db3_files:
            return db3_files[0]
    raise RuntimeError(f"Could not find a rosbag2 .db3 file under {bag_path}")


def infer_bag_format(bag_path, bag_format):
    if bag_format != "auto":
        return bag_format
    path = Path(bag_path)
    if path.is_dir() or path.suffix == ".db3":
        return "ros2"
    if path.suffix == ".bag":
        return "ros1"
    raise RuntimeError(f"Cannot infer bag format from {bag_path}; pass --bag-format ros1 or ros2")


def extract_ros1_frames(bag_path, topic, output_dir, max_frames, normalize_intensity, drop_invalid):
    typestore = get_typestore(Stores.ROS1_NOETIC)

    with Rosbag1Reader(bag_path) as reader:
        connections = [conn for conn in reader.connections if conn.topic == topic]
        if not connections:
            available = "\n".join(f"  {conn.topic} ({conn.msgtype})" for conn in reader.connections)
            raise RuntimeError(f"Topic '{topic}' not found. Available topics:\n{available}")

        count = 0
        for connection, _, rawdata in reader.messages(connections=connections):
            msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
            xyzi = pointcloud2_to_xyzi(
                msg,
                normalize_intensity=normalize_intensity,
                drop_invalid=drop_invalid,
            )
            write_xyzi_frame(xyzi, output_dir, count)

            count += 1
            if max_frames > 0 and count >= max_frames:
                break

    return count


def extract_ros2_frames(bag_path, topic, output_dir, max_frames, normalize_intensity, drop_invalid):
    output_dir = os.path.expanduser(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    db3_path = find_ros2_db3(bag_path)
    typestore = get_typestore(Stores.ROS2_HUMBLE)

    with sqlite3.connect(db3_path) as con:
        cur = con.cursor()
        topic_row = cur.execute(
            "select id, type, serialization_format from topics where name = ?",
            (topic,),
        ).fetchone()
        if topic_row is None:
            available = "\n".join(
                f"  {name} ({msgtype}, {fmt})"
                for name, msgtype, fmt in cur.execute(
                    "select name, type, serialization_format from topics order by id"
                )
            )
            raise RuntimeError(f"Topic '{topic}' not found. Available topics:\n{available}")

        topic_id, msgtype, serialization_format = topic_row
        if serialization_format != "cdr":
            raise RuntimeError(
                f"Topic '{topic}' uses serialization '{serialization_format}', expected 'cdr'"
            )

        count = 0
        for (rawdata,) in cur.execute(
            "select data from messages where topic_id = ? order by timestamp",
            (topic_id,),
        ):
            msg = typestore.deserialize_cdr(rawdata, msgtype)
            xyzi = pointcloud2_to_xyzi(
                msg,
                normalize_intensity=normalize_intensity,
                drop_invalid=drop_invalid,
            )
            write_xyzi_frame(xyzi, output_dir, count)

            count += 1
            if max_frames > 0 and count >= max_frames:
                break

    return count


def extract_frames(bag_path, topic, output_dir, max_frames, normalize_intensity, drop_invalid, bag_format):
    os.makedirs(output_dir, exist_ok=True)
    resolved_format = infer_bag_format(bag_path, bag_format)

    if resolved_format == "ros1":
        count = extract_ros1_frames(
            bag_path,
            topic,
            output_dir,
            max_frames,
            normalize_intensity,
            drop_invalid,
        )
    else:
        count = extract_ros2_frames(
            bag_path,
            topic,
            output_dir,
            max_frames,
            normalize_intensity,
            drop_invalid,
        )

    print(f"Done. Extracted {count} {resolved_format} frames to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", default=DEFAULT_BAG_PATH, help="ROS1 .bag, ROS2 bag directory, or ROS2 .db3 path")
    parser.add_argument("--bag-format", choices=["auto", "ros1", "ros2"], default="auto")
    parser.add_argument("--topic", default=DEFAULT_TOPIC, help="PointCloud2 topic")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Folder for .bin files")
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES, help="0 means all frames")
    parser.add_argument("--keep-invalid", action="store_true", help="Keep zero/invalid organized-cloud points")
    parser.add_argument("--raw-intensity", action="store_true", help="Do not normalize intensity to 0..1")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extract_ros2_frames(
        os.path.expanduser(args.bag),
        args.topic,
        os.path.expanduser(args.output_dir),
        args.max_frames,
        normalize_intensity=not args.raw_intensity,
        drop_invalid=not args.keep_invalid,
    )