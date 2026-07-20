#!/usr/bin/env python3
"""计算 trajectory.pcd 的端点漂移（首帧到末帧的直线距离）。

用法:
    python3 drift.py <trajectory.pcd> [更多 trajectory.pcd ...]

纯标准库实现，无需 pypcd / open3d / numpy。
支持 PCD 的 binary 与 ascii 两种 DATA 格式，按字段名定位 x/y/z。
"""
import sys
import struct


def read_pcd(path):
    """读取 PCD，返回 [(x, y, z), ...] 列表。"""
    with open(path, "rb") as f:
        fields, types = [], []
        width = height = points = 0
        data_fmt = None
        while True:
            line = f.readline().decode("ascii", errors="ignore").strip()
            if not line:
                continue
            parts = line.split()
            key = parts[0]
            if key == "FIELDS":
                fields = parts[1:]
            elif key == "SIZE":
                pass  # 大小由 TYPE 推导，此处忽略
            elif key == "TYPE":
                types = parts[1:]
            elif key == "WIDTH":
                width = int(parts[1])
            elif key == "HEIGHT":
                height = int(parts[1])
            elif key == "POINTS":
                points = int(parts[1])
            elif key == "DATA":
                data_fmt = parts[1]
                break

        fmt_map = {"F": "f", "U": "I", "I": "i", "S": "h", "B": "B"}
        field_struct = "<" + "".join(fmt_map[t] for t in types)
        point_size = struct.calcsize(field_struct)
        if points == 0:
            points = width * height

        if data_fmt == "ascii":
            raw = f.read().decode().strip().split("\n")
            pts = []
            for r in raw:
                vals = r.split()
                if len(vals) >= 3:
                    pts.append((float(vals[0]), float(vals[1]), float(vals[2])))
            return pts

        data = f.read(points * point_size)
        pts = []
        for i in range(points):
            chunk = data[i * point_size:(i + 1) * point_size]
            vals = struct.unpack(field_struct, chunk)
            x = vals[fields.index("x")] if "x" in fields else vals[0]
            y = vals[fields.index("y")] if "y" in fields else vals[1]
            z = vals[fields.index("z")] if "z" in fields else vals[2]
            pts.append((x, y, z))
        return pts


def main():
    paths = sys.argv[1:]
    if not paths:
        print("用法: python3 drift.py <trajectory.pcd> [更多 trajectory.pcd ...]")
        sys.exit(1)
    for p in paths:
        try:
            pts = read_pcd(p)
        except Exception as e:
            print(f"{p}\n  读取失败: {e}\n")
            continue
        if len(pts) < 2:
            print(f"{p}\n  关键帧数不足: {len(pts)}\n")
            continue
        s = pts[0]
        e = pts[-1]
        drift = ((e[0] - s[0]) ** 2 + (e[1] - s[1]) ** 2 + (e[2] - s[2]) ** 2) ** 0.5
        print(f"{p}")
        print(f"  关键帧数 = {len(pts)}")
        print(f"  起点 = ({s[0]:.2f}, {s[1]:.2f}, {s[2]:.2f})")
        print(f"  终点 = ({e[0]:.2f}, {e[1]:.2f}, {e[2]:.2f})")
        print(f"  端点漂移 = {drift:.3f} m")
        print()


if __name__ == "__main__":
    main()
