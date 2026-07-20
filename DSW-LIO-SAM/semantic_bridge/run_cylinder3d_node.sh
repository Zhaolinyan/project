#!/bin/bash
# Cylinder3D 语义分割节点启动脚本 (IPC 架构)
#
# 架构:
#   本进程: 系统 Python 3.10 → 运行 ROS2 节点 (rclpy/sensor_msgs)
#   子进程: conda Python 3.8   → 运行推理引擎 (torch/spconv)
#
# 此脚本由 colcon 安装到 lib/dsw_lio_sam/

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NODE_SCRIPT="$SCRIPT_DIR/cylinder3d_ros_node.py"

if [ ! -f "$NODE_SCRIPT" ]; then
    echo "[ERROR] $NODE_SCRIPT not found!" >&2
    exit 1
fi

# 强制使用系统 Python (ROS2 Humble 用 3.10 编译)
# 不能用 conda 的 python (可能是 3.8，缺少 rclpy)
SYSTEM_PYTHON="/usr/bin/python3"
if [ ! -x "$SYSTEM_PYTHON" ]; then
    # 备选: 找到非 conda 的系统 python
    SYSTEM_PYTHON=$(which -a python3 2>/dev/null | grep -v miniconda | grep -v anaconda | head -1)
fi
if [ -z "$SYSTEM_PYTHON" ] || [ ! -x "$SYSTEM_PYTHON" ]; then
    echo "[ERROR] Cannot find system python3!" >&2
    exit 1
fi

export PYTHONUNBUFFERED=1

exec "$SYSTEM_PYTHON" "$NODE_SCRIPT" "$@"
