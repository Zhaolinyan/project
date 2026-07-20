#!/usr/bin/env python3
"""
语义桥接节点：订阅原始LiDAR点云，用简单几何规则打伪语义标签，
发布 /semantic_cloud (sensor_msgs/PointCloud2, PointXYZIL格式)
"""

import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField

class SemanticBridge(Node):
    def __init__(self):
        super().__init__('semantic_bridge')

        # 订阅原始点云
        self.sub = self.create_subscription(
            PointCloud2, '/os1_cloud_node/points', self.cloud_callback, 10)
        
        # 发布带语义标签的点云
        self.pub = self.create_publisher(PointCloud2, '/semantic_cloud', 10)
        self.declare_parameter('pseudoGroundMaxZ', -0.5)
        self.declare_parameter('pseudoStructureMinZ', 2.0)
        self.declare_parameter('pseudoCanopyMinZ', 2.0)
        self.pseudo_ground_max_z = (
            self.get_parameter('pseudoGroundMaxZ').get_parameter_value().double_value)
        self.pseudo_structure_min_z = (
            self.get_parameter('pseudoStructureMinZ').get_parameter_value().double_value)
        self.pseudo_canopy_min_z = (
            self.get_parameter('pseudoCanopyMinZ').get_parameter_value().double_value)
        
        self.get_logger().info('语义桥接节点已启动')
    
    def cloud_callback(self, msg):
        """
        对每一点，根据高度和反射强度打伪标签：
        - z > 2.0  → TREE (11)        树木/高处物体
        - z < -0.5  → TERRAIN (10)     地面以下
        - intensity > 0.8 → BUILDING (7)  高反射 = 建筑物
        - 其他          → ROAD (8)         路面
        """
        # 解析 PointCloud2 数据（NumPy 向量化，替代逐点循环，性能提升 100 倍以上）
        point_step = msg.point_step
        num_points = msg.width * msg.height

        # 将原始字节按 point_step 重排为 (N, point_step) 的 uint8 矩阵
        raw = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(num_points, point_step)
        # PointXYZI 格式：x/y/z/intensity 位于前 16 字节，各 float32
        floats = raw[:, 0:16].copy().view(np.float32).reshape(num_points, 4)
        x = floats[:, 0]
        y = floats[:, 1]
        z = floats[:, 2]
        intensity = floats[:, 3]

        # 规则式打标签（向量化）：优先级 TREE > TERRAIN > BUILDING > ROAD
        labels = np.full(num_points, 8, dtype=np.uint32)          # 默认 ROAD
        labels[(z > self.pseudo_structure_min_z) | (intensity > 0.8)] = 7
        labels[z < self.pseudo_ground_max_z] = 10
        labels[z > self.pseudo_canopy_min_z] = 11

        # 构建输出结构化数组：x, y, z, intensity(float32) + label(uint32)
        out = np.zeros(num_points, dtype=[
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('intensity', 'f4'), ('label', 'u4')])
        out['x'] = x
        out['y'] = y
        out['z'] = z
        out['intensity'] = intensity
        out['label'] = labels
        new_data = out.tobytes()

        # 发布新的 PointCloud2
        new_msg = PointCloud2()
        new_msg.header = msg.header
        new_msg.height = msg.height
        new_msg.width = msg.width
        new_msg.fields = [
            PointField(name='x',         offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y',         offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z',         offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name='label',     offset=16, datatype=PointField.UINT32,  count=1),
        ]
        new_msg.is_bigendian = msg.is_bigendian
        new_msg.point_step = 20  # x+y+z+intensity+label = 4*4+4 = 20 bytes
        new_msg.row_step = new_msg.point_step * msg.width
        new_msg.data = bytes(new_data)
        new_msg.is_dense = msg.is_dense
        
        self.pub.publish(new_msg)

def main():
    rclpy.init()
    node = SemanticBridge()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
