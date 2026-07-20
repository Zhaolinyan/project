#!/bin/bash
# 语义回放节点启动脚本
# 自动退出 conda 环境，用系统 Python 3.10 运行
# 用法: SEMANTIC_DIR=/path/to/semantic bash run_replay.sh
#   或: bash run_replay.sh /path/to/semantic

# 退出所有 conda 环境（两次确保退出嵌套）
conda deactivate 2>/dev/null
conda deactivate 2>/dev/null

# 加载 ROS2 环境
source /opt/ros/humble/setup.bash

# 定位脚本所在目录，避免硬编码 $HOME/ros2_ws（I3）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NODE_PY="$SCRIPT_DIR/semantic_replay_node.py"

# 语义帧目录：优先命令行参数，其次环境变量
SEMANTIC_DIR="${1:-${SEMANTIC_DIR:-}}"
if [ -z "$SEMANTIC_DIR" ]; then
    echo "错误：未指定语义帧目录。用法: bash run_replay.sh /path/to/semantic" >&2
    exit 1
fi

# 用系统 Python 3.10 运行回放节点，通过 ROS2 参数传入语义目录
/usr/bin/python3 "$NODE_PY" --ros-args -p semantic_dir:="$SEMANTIC_DIR"
