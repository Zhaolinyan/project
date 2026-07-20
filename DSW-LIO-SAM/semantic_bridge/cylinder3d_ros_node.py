#!/usr/bin/env python3
"""
Cylinder3D ROS2 realtime semantic segmentation node.

This node owns ROS communication and delegates model inference to
cylinder3d_inference.py through a binary stdin/stdout protocol.
"""

import argparse
import os
import select
import struct
import subprocess
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField


def read_exact(fd, n_bytes, timeout=30):
    """Read exactly n_bytes from fd, or return None on EOF/timeout."""
    buf = b""
    start_time = time.time()
    while len(buf) < n_bytes:
        if timeout > 0 and (time.time() - start_time) > timeout:
            return None

        remaining = n_bytes - len(buf)
        try:
            if hasattr(fd, "fileno"):
                r, _, _ = select.select([fd], [], [], min(1.0, timeout if timeout > 0 else 1.0))
                if not r:
                    continue
            chunk = fd.read(min(remaining, 65536))
        except (OSError, ValueError):
            return None

        if not chunk:
            return None
        buf += chunk
    return buf


class Cylinder3DNode(Node):
    """ROS2 wrapper for Cylinder3D inference."""

    def __init__(self, cli_args=None):
        super().__init__("cylinder3d_semantic_node")
        args = cli_args or {}

        self.declare_parameter("config_path", args.get("config_path", "config/newer_college.yaml"))
        self.declare_parameter("device", args.get("device", "cuda:0"))
        self.declare_parameter("input_topic", args.get("input_topic", "/os1_cloud_node/points"))
        self.declare_parameter("output_topic", args.get("output_topic", "/semantic_cloud"))
        self.declare_parameter("z_min", float(args.get("z_min", -5.0)))
        self.declare_parameter("z_max", float(args.get("z_max", 15.0)))
        self.declare_parameter("rho_max", float(args.get("rho_max", 50.0)))
        self.declare_parameter("rho_filter", float(args.get("rho_filter", 55.0)))
        self.declare_parameter("use_config_voxel_profile", bool(args.get("use_config_voxel_profile", True)))

        config_path = self.get_parameter("config_path").value
        device_str = self.get_parameter("device").value
        input_topic = self.get_parameter("input_topic").value
        output_topic = self.get_parameter("output_topic").value

        self.z_min = self.get_parameter("z_min").value
        self.z_max = self.get_parameter("z_max").value
        self.rho_max = self.get_parameter("rho_max").value
        self.rho_filter = self.get_parameter("rho_filter").value
        self.use_config_voxel_profile = self.get_parameter("use_config_voxel_profile").value

        self._proc = self._start_inference_subprocess(config_path, device_str)

        qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self.sub = self.create_subscription(PointCloud2, input_topic, self.cloud_callback, qos)
        self.pub = self.create_publisher(PointCloud2, output_topic, 10)

        self._frame_count = 0
        self.get_logger().info(
            f"Cylinder3D IPC node started | Input: {input_topic} | Output: {output_topic} | "
            f"voxel_profile={'config/checkpoint' if self.use_config_voxel_profile else 'launch override'} | "
            f"rho_filter={self.rho_filter}m"
        )

    def _find_conda_python(self):
        for conda_base in [
            os.path.expanduser("~/miniconda3"),
            os.path.expanduser("~/anaconda3"),
            "/opt/conda",
        ]:
            python_bin = os.path.join(conda_base, "envs", "cylinder3d", "bin", "python3")
            if os.path.isfile(python_bin):
                return python_bin
        return None

    def _get_inference_script(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(script_dir, "cylinder3d_inference.py"),
            os.path.join(script_dir, "..", "..", "src", "dsw_lio_sam", "semantic_bridge", "cylinder3d_inference.py"),
        ]
        for path in candidates:
            if os.path.isfile(path):
                return os.path.abspath(path)
        return candidates[0]

    def _start_inference_subprocess(self, config_path, device):
        python_bin = self._find_conda_python()
        if python_bin is None:
            raise RuntimeError("[FATAL] cylinder3d conda env not found")

        inf_script = self._get_inference_script()
        if not os.path.isfile(inf_script):
            raise RuntimeError(f"[FATAL] inference script not found: {inf_script}")

        cmd = [
            python_bin,
            inf_script,
            "--config_path", config_path,
            "--device", device,
        ]
        if not self.use_config_voxel_profile:
            cmd.extend([
                "--z_min", str(self.z_min),
                "--z_max", str(self.z_max),
                "--rho_max", str(self.rho_max),
            ])

        log_dir = os.path.expanduser("~/dsw_lio_sam_logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "cylinder3d_inference.log")
        stderr_file = open(log_path, "w")

        self.get_logger().info(f"Spawning inference: {python_bin}")
        self.get_logger().info(f"Script: {inf_script}")
        self.get_logger().info(f"Stderr log: {log_path}")

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr_file,
        )

        max_wait = 60
        ready = False
        self.get_logger().info("Waiting for engine to load model...")
        for i in range(max_wait):
            ret = proc.poll()
            if ret is not None:
                stderr_file.close()
                with open(log_path, "r") as f:
                    log_content = f.read()
                raise RuntimeError(f"Engine crashed during init (rc={ret}):\n{log_content[-2000:]}")

            stderr_file.flush()
            try:
                with open(log_path, "r") as f:
                    log_content = f.read()
                if "ready for inference" in log_content.lower():
                    ready = True
                    self.get_logger().info(f"Engine ready after {i + 1}s")
                    break
            except Exception as exc:
                self.get_logger().warn(f"Log read error: {exc}")
            time.sleep(1)

        if not ready:
            stderr_file.close()
            with open(log_path, "r") as f:
                log_content = f.read()
            raise RuntimeError(f"Engine not ready after {max_wait}s:\n{log_content[-2000:]}")

        self._verify_pipe(proc)
        self._stderr_file = stderr_file
        return proc

    def _verify_pipe(self, proc):
        dummy_n = struct.pack("<q", 1)
        dummy_xyz = np.array([[0.0, 0.0, 0.0]], dtype="<f4").tobytes()
        dummy_int = np.array([0.0], dtype="<f4").tobytes()
        proc.stdin.write(dummy_n + dummy_xyz + dummy_int)
        proc.stdin.flush()

        resp = read_exact(proc.stdout, 8, timeout=5)
        if resp is None:
            raise RuntimeError("Engine not responding to dummy frame")
        n_labels = struct.unpack("<q", resp)[0]
        label_data = read_exact(proc.stdout, n_labels * 4, timeout=5)
        if label_data is None:
            raise RuntimeError("Engine label read failed on dummy frame")
        self.get_logger().info(f"Pipe verified (dummy n_labels={n_labels})")

    def _send_and_recv(self, xyz, intensity):
        n_pts = len(xyz)
        if n_pts != len(intensity):
            raise ValueError("xyz/intensity length mismatch")

        data = (
            struct.pack("<q", n_pts)
            + np.asarray(xyz, dtype="<f4").tobytes()
            + np.asarray(intensity, dtype="<f4").tobytes()
        )

        try:
            self._proc.stdin.write(data)
            self._proc.stdin.flush()
        except BrokenPipeError:
            self.get_logger().error("[IPC] BrokenPipe on write")
            return None

        resp_header = read_exact(self._proc.stdout, 8, timeout=30)
        if resp_header is None:
            self.get_logger().error("[IPC] EOF/timeout on header read")
            return None

        n_labels = struct.unpack("<q", resp_header)[0]
        label_data = read_exact(self._proc.stdout, n_labels * 4, timeout=30)
        if label_data is None:
            self.get_logger().error("[IPC] EOF/timeout on label data read")
            return None

        labels = np.frombuffer(label_data, dtype="<u4").copy()
        if n_labels != n_pts:
            self.get_logger().warn(f"[IPC] Label count mismatch: expected {n_pts}, got {n_labels}")
            return None
        return labels

    def _find_field_offset(self, msg, field_names):
        field_map = {}
        for field in msg.fields:
            name = field.name.strip("\x00").strip()
            if name in field_names:
                field_map[name] = field.offset
        return [field_map.get(name) for name in field_names]

    def _parse_ouster_cloud(self, msg):
        num_points = msg.width * msg.height
        point_step = msg.point_step

        x_off, y_off, z_off = self._find_field_offset(msg, ["x", "y", "z"])
        i_offsets = self._find_field_offset(msg, ["intensity", "signal", "reflectivity"])
        i_off = next((off for off in i_offsets if off is not None), None)

        if x_off is None or y_off is None or z_off is None:
            self.get_logger().error(f"Cannot find xyz fields. Available: {[f.name for f in msg.fields]}")
            return None, None

        raw = np.frombuffer(msg.data, dtype=np.uint8, count=num_points * point_step)
        raw = raw.reshape(num_points, point_step)

        xyz = np.zeros((num_points, 3), dtype=np.float32)
        xyz[:, 0] = raw[:, x_off:x_off + 4].copy().view("<f4").reshape(-1)
        xyz[:, 1] = raw[:, y_off:y_off + 4].copy().view("<f4").reshape(-1)
        xyz[:, 2] = raw[:, z_off:z_off + 4].copy().view("<f4").reshape(-1)

        intensity = np.zeros(num_points, dtype=np.float32)
        if i_off is not None:
            intensity = raw[:, i_off:i_off + 4].copy().view("<f4").reshape(-1)

        return xyz, intensity

    def _build_semantic_cloud_msg(self, xyz, intensity, labels, header):
        n = len(xyz)
        cloud = np.zeros(
            n,
            dtype=[
                ("x", np.float32),
                ("y", np.float32),
                ("z", np.float32),
                ("intensity", np.float32),
                ("label", np.uint32),
            ],
        )
        cloud["x"] = xyz[:, 0]
        cloud["y"] = xyz[:, 1]
        cloud["z"] = xyz[:, 2]
        cloud["intensity"] = intensity
        cloud["label"] = labels

        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = n
        msg.fields = self._make_fields()
        msg.is_bigendian = False
        msg.point_step = 20
        msg.row_step = 20 * n
        msg.data = cloud.tobytes()
        msg.is_dense = bool(np.isfinite(xyz).all())
        return msg

    @staticmethod
    def _make_fields():
        fields = []
        for name, offset, datatype in [
            ("x", 0, PointField.FLOAT32),
            ("y", 4, PointField.FLOAT32),
            ("z", 8, PointField.FLOAT32),
            ("intensity", 12, PointField.FLOAT32),
            ("label", 16, PointField.UINT32),
        ]:
            field = PointField()
            field.name = name
            field.offset = offset
            field.datatype = datatype
            field.count = 1
            fields.append(field)
        return fields

    def cloud_callback(self, msg):
        num_points = msg.width * msg.height
        if num_points == 0:
            return

        xyz_all, intensity_all = self._parse_ouster_cloud(msg)
        if xyz_all is None:
            return

        dist = np.sqrt(xyz_all[:, 0] ** 2 + xyz_all[:, 1] ** 2 + xyz_all[:, 2] ** 2)
        valid = np.isfinite(xyz_all).all(axis=1) & (dist > 0.1) & (dist < self.rho_filter)
        xyz = xyz_all[valid]
        intensity = intensity_all[valid]
        if len(xyz) == 0:
            return

        if intensity.max() > 1.0:
            intensity = intensity / max(float(intensity.max()), 1e-6)

        inferred_labels = self._send_and_recv(xyz, intensity)
        if inferred_labels is None:
            self.get_logger().error("Cylinder3D inference failed; semantic frame was not published")
            return
        inferred_labels = inferred_labels.astype(np.uint32, copy=False)
        labels = np.zeros(num_points, dtype=np.uint32)
        labels[valid] = inferred_labels

        out_msg = self._build_semantic_cloud_msg(xyz_all, intensity_all, labels, msg.header)
        self.pub.publish(out_msg)

        self._frame_count += 1
        if self._frame_count % 50 == 0:
            nz = int((labels != 0).sum())
            self.get_logger().info(
                f"Frame {self._frame_count}: infer={len(xyz)} pts, "
                f"published={num_points} pts, non-zero={nz} ({nz / max(num_points, 1) * 100:.1f}%)"
            )


def main():
    def parse_bool(value):
        return str(value).strip().lower() in ("1", "true", "yes", "on")

    parser = argparse.ArgumentParser(description="Cylinder3D semantic segmentation ROS2 node (IPC)")
    parser.add_argument("--config_path", type=str, default="config/newer_college.yaml")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--input_topic", type=str, default="/os1_cloud_node/points")
    parser.add_argument("--output_topic", type=str, default="/semantic_cloud")
    parser.add_argument("--z_min", type=float, default=-5.0)
    parser.add_argument("--z_max", type=float, default=15.0)
    parser.add_argument("--rho_max", type=float, default=50.0)
    parser.add_argument("--rho_filter", type=float, default=55.0)
    parser.add_argument("--use_config_voxel_profile", type=parse_bool, default=True)
    args, _ = parser.parse_known_args()

    rclpy.init()
    node = Cylinder3DNode(cli_args=vars(args))
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if getattr(node, "_proc", None):
            try:
                node._proc.stdin.close()
                node._proc.terminate()
                node._proc.wait(timeout=5)
            except Exception as exc:
                node.get_logger().warn(f"Engine cleanup failed: {exc}")
        if hasattr(node, "_stderr_file"):
            node._stderr_file.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
