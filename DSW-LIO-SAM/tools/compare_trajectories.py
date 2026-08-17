#!/usr/bin/env python3
"""
轨迹对比可视化脚本
读取四组 trajectory.pcd 并绘制 3D 对比图

用法:
    python compare_trajectories.py
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# ========== 修改这里的路径 ==========
# 四组实验的 trajectory.pcd 路径
TRAJ_PATHS = {
    "Original LIO-SAM": "/home/zly/dsw_lio_sam_results/original_lio_sam_rooster_001/trajectory.pcd",
    "DSW Ablation (No Semantic)": "/home/zly/dsw_lio_sam_results/baseline_no_semantic_rooster_001/trajectory.pcd",
    "Pseudo-Semantic (alpha=0.5)": "/home/zly/dsw_lio_sam_results/pseudo_semantic_rooster_001/trajectory.pcd",
    "Cylinder3D Semantic (alpha=0.5)": "/home/zly/dsw_lio_sam_results/cylinder3d_true_semantic_rooster_001/trajectory.pcd",
}
# ===================================


def read_pcd(pcd_path):
    """
    读取 PCL 保存的 PCD 文件（支持 ASCII 和二进制格式）
    返回 (N, 3) 的 xyz 坐标数组
    """
    if not os.path.exists(pcd_path):
        print(f"  [警告] 文件不存在: {pcd_path}")
        return None

    with open(pcd_path, 'rb') as f:
        raw = f.read()

    # 找到 DATA 行的位置
    header_end = raw.find(b'DATA')
    if header_end == -1:
        print(f"  [警告] 无法解析 PCD 头部: {pcd_path}")
        return None

    # 找到 DATA 行结束的位置（换行符）
    data_start = raw.find(b'\n', header_end) + 1

    # 解析头部获取字段信息
    header = raw[:data_start].decode('ascii', errors='ignore')
    fields = []
    for line in header.split('\n'):
        if line.startswith('FIELDS'):
            fields = line.split()[1:]
        elif line.startswith('POINTS'):
            n_points = int(line.split()[1])
        elif line.startswith('DATA'):
            data_format = line.split()[1]

    # 找到 xyz 字段的索引
    x_idx = fields.index('x') if 'x' in fields else 0
    y_idx = fields.index('y') if 'y' in fields else 1
    z_idx = fields.index('z') if 'z' in fields else 2

    # 读取数据
    data = raw[data_start:]

    if data_format == 'ascii':
        # ASCII 格式
        points = []
        for line in data.decode('ascii', errors='ignore').strip().split('\n'):
            parts = line.strip().split()
            if len(parts) >= 3:
                points.append([
                    float(parts[x_idx]),
                    float(parts[y_idx]),
                    float(parts[z_idx])
                ])
        return np.array(points)

    elif data_format == 'binary':
        # 二进制格式：每个字段是 float32
        # 计算每个点占用的字节数
        point_step = len(fields) * 4  # 每个字段 4 字节 (float32)
        points = []
        for i in range(n_points):
            offset = i * point_step
            x = np.frombuffer(data[offset + x_idx*4:offset + x_idx*4 + 4], dtype=np.float32)[0]
            y = np.frombuffer(data[offset + y_idx*4:offset + y_idx*4 + 4], dtype=np.float32)[0]
            z = np.frombuffer(data[offset + z_idx*4:offset + z_idx*4 + 4], dtype=np.float32)[0]
            points.append([x, y, z])
        return np.array(points)

    else:
        print(f"  [警告] 不支持的 PCD 格式: {data_format}")
        return None


def plot_trajectories(trajectories):
    """
    绘制 3D 轨迹对比图
    """
    fig = plt.figure(figsize=(14, 10))

    # ---- 子图1: 3D 轨迹 ----
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    colors = ['#1f77b4', '#7f7f7f', '#ff7f0e', '#2ca02c']
    for (name, pts), color in zip(trajectories.items(), colors):
        if pts is not None:
            ax1.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                     color=color, linewidth=2, label=name)
            # 标记起点和终点
            ax1.scatter(*pts[0],  color=color, s=80, marker='^',
                        edgecolors='black', linewidth=1.5, zorder=10)
            ax1.scatter(*pts[-1], color=color, s=80, marker='s',
                        edgecolors='black', linewidth=1.5, zorder=10)

    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectory Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ---- 子图2: XY 俯视图 ----
    ax2 = fig.add_subplot(2, 2, 2)
    for (name, pts), color in zip(trajectories.items(), colors):
        if pts is not None:
            ax2.plot(pts[:, 0], pts[:, 1],
                     color=color, linewidth=2, label=name)
            ax2.scatter(pts[0, 0],  pts[0, 1],  color=color,
                        s=80, marker='^', edgecolors='black', linewidth=1.5)
            ax2.scatter(pts[-1, 0], pts[-1, 1], color=color,
                        s=80, marker='s', edgecolors='black', linewidth=1.5)
            # 标注终点偏移量
            ax2.annotate(f'End: ({pts[-1,0]:.2f}, {pts[-1,1]:.2f})',
                         xy=(pts[-1, 0], pts[-1, 1]),
                         xytext=(5, 5), textcoords='offset points',
                         fontsize=8, color=color)

    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Top-down View (XY Plane)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')

    # ---- 子图3: XZ 侧视图 ----
    ax3 = fig.add_subplot(2, 2, 3)
    for (name, pts), color in zip(trajectories.items(), colors):
        if pts is not None:
            ax3.plot(pts[:, 0], pts[:, 2],
                     color=color, linewidth=2, label=name)
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Z (m)')
    ax3.set_title('Side View (XZ Plane)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # ---- 子图4: 轨迹长度对比 ----
    ax4 = fig.add_subplot(2, 2, 4)
    lengths = []
    names = []
    for name, pts in trajectories.items():
        if pts is not None:
            # 计算累计路径长度
            diffs = np.diff(pts[:, :3], axis=0)
            seg_lengths = np.linalg.norm(diffs, axis=1)
            total_length = np.sum(seg_lengths)
            lengths.append(total_length)
            names.append(name)

    bars = ax4.bar(range(len(lengths)), lengths,
                   color=colors[:len(lengths)], alpha=0.7)
    ax4.set_xticks(range(len(names)))
    ax4.set_xticklabels(names, rotation=15, ha='right')
    ax4.set_ylabel('Trajectory Length (m)')
    ax4.set_title('Total Path Length')
    ax4.grid(True, alpha=0.3, axis='y')

    # 在柱子上标注数值
    for bar, length in zip(bars, lengths):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{length:.2f}m', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    # 保存图片
    output_path = os.path.expanduser("~/trajectory_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n图表已保存至: {output_path}")
    plt.show()


def print_stats(trajectories):
    """Print trajectory statistics"""
    print("\n" + "="*60)
    print("Trajectory Statistics Comparison")
    print("="*60)
    for name, pts in trajectories.items():
        if pts is None:
            print(f"\n[{name}]  File not found, skipped")
            continue
        # 计算累计路径长度
        diffs = np.diff(pts[:, :3], axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)
        total_length = np.sum(seg_lengths)

        # 起点到终点的直线距离 (闭环精度)
        start_pt = pts[0]
        end_pt = pts[-1]
        drift = np.linalg.norm(end_pt - start_pt)

        print(f"\n[{name}]")
        print(f"  Keyframes:     {len(pts)}")
        print(f"  Total Length:  {total_length:.3f} m")
        print(f"  Start:         ({start_pt[0]:.3f}, {start_pt[1]:.3f}, {start_pt[2]:.3f})")
        print(f"  End:           ({end_pt[0]:.3f}, {end_pt[1]:.3f}, {end_pt[2]:.3f})")
        print(f"  End Drift:     {drift:.3f} m")


if __name__ == "__main__":
    print("Loading trajectory files...")
    trajectories = {}
    for name, path in TRAJ_PATHS.items():
        print(f"  Loading: {name}")
        pts = read_pcd(path)
        trajectories[name] = pts

    print_stats(trajectories)
    plot_trajectories(trajectories)
