#!/usr/bin/env python3
"""
Cylinder3D 语义分割节点 launch 文件 (IPC 子进程架构)

架构:
  ROS2 节点 (系统 Python 3.10): 订阅点云 → 解析/过滤 → 通过 pipe 发给子进程 → 发布语义点云
  推理引擎 (conda Python 3.8) : 从 pipe 收数据 → Cylinder3D 推理 → 通过 pipe 返回标签

启动方式:
  ros2 launch dsw_lio_sam semantic.launch.py
  ros2 launch dsw_lio_sam semantic.launch.py device:=cpu

依赖:
  - cylinder3d conda 环境 (PyTorch + spconv) - 用于推理子进程
  - 已执行 colcon build --packages-select dsw_lio_sam
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # ========== 参数声明 ==========
        DeclareLaunchArgument('config_path', default_value='config/newer_college.yaml',
                              description='Cylinder3D 配置文件路径(相对于 Cylinder3D 目录)'),
        DeclareLaunchArgument('device', default_value='cuda:0',
                              description='推理设备 (cuda:0 或 cpu)'),
        DeclareLaunchArgument('input_topic', default_value='/os1_cloud_node/points',
                              description='输入点云话题 (Ouster LiDAR)'),
        DeclareLaunchArgument('output_topic', default_value='/semantic_cloud',
                              description='输出语义点云话题'),
        DeclareLaunchArgument('z_min', default_value='-5.0',
                              description='体素 z 轴最小值 (Newer College 校园适配)'),
        DeclareLaunchArgument('z_max', default_value='15.0',
                              description='体素 z 轴最大值 (校园建筑高度)'),
        DeclareLaunchArgument('rho_max', default_value='50.0',
                              description='体素 rho 最大值 (有效感知范围)'),
        DeclareLaunchArgument('rho_filter', default_value='55.0',
                              description='噪声过滤距离 (Ouster 远距离噪声)'),

        # ---- 启动信息 ----
        LogInfo(msg=['\n' + '='*60]),
        LogInfo(msg=['  Cylinder3D Semantic Segmentation Node']),
        LogInfo(msg=['  Dataset: Newer College (Ouster OS1-64)']),
        LogInfo(msg=['  Device: ', LaunchConfiguration('device')]),
        LogInfo(msg=['  Input:  ', LaunchConfiguration('input_topic')]),
        LogInfo(msg=['  Output: ', LaunchConfiguration('output_topic')]),
        LogInfo(msg=['='*60 + '\n']),

        # ========== Cylinder3D 语义分割节点 ==========
        # IPC 架构: ROS2 节点自动启动 conda 推理子进程
        Node(
            package='dsw_lio_sam',
            executable='run_cylinder3d_node',
            name='cylinder3d_semantic_node',
            output='screen',
            arguments=[
                '--config_path', LaunchConfiguration('config_path'),
                '--device',      LaunchConfiguration('device'),
                '--input_topic', LaunchConfiguration('input_topic'),
                '--output_topic',LaunchConfiguration('output_topic'),
                '--z_min',       LaunchConfiguration('z_min'),
                '--z_max',       LaunchConfiguration('z_max'),
                '--rho_max',     LaunchConfiguration('rho_max'),
                '--rho_filter',  LaunchConfiguration('rho_filter'),
            ],
        ),
    ])
