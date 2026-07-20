# DSW-LIO-SAM 团队技术提升指南

> 基于实际代码审查，由资深工程师整理。结合 ROS2 + SLAM 开发最佳实践。

---

## 第一部分：当前代码质量问题清单

### 🔴 已修复（P0 高危）

#### 1. params.yaml 重复键值定义
**问题**：同一个 YAML 文件中，`sensor`、`N_SCAN`、`Horizon_SCAN`、`imuAccBiasN`、`imuGyrBiasN` 被定义了两次，且值不一致。
```yaml
# 错误示例（已修复）
sensor: velodyne   ← 第一次定义，错误值
...
sensor: ouster     ← 第二次定义，正确值
```
**后果**：ROS2 的 yaml-cpp 解析器行为未定义，实际生效的值不可预测，导致系统用 Velodyne 格式解析 Ouster 数据，全部点云静默失败。

**规范**：YAML 配置文件每个键只能定义一次。修改参数前用 `grep -n "sensor:" params.yaml` 检查是否有重复。

---

### 🟡 待修复（P1 中危）

#### 2. 诊断日志缺失 —— 静默失败是调试噩梦

**问题**：`deskewInfo()` 函数在无法处理点云帧时只打印一条通用 INFO：

```cpp
// 当前代码（imageProjection.cpp 第410行）
if (imuQueue.empty() ||
    stamp2Sec(imuQueue.front().header.stamp) > timeScanCur ||
    stamp2Sec(imuQueue.back().header.stamp) < timeScanEnd)
{
    RCLCPP_INFO(get_logger(), "Waiting for IMU data ...");
    return false;
}
```

**问题**：三个条件触发的原因完全不同，但打印同一条日志，无法区分：
- `imuQueue.empty()` → IMU话题根本没数据
- `front().stamp > timeScanCur` → IMU比点云晚（时间戳超前）
- `back().stamp < timeScanEnd` → IMU比点云早（时间戳落后）

**推荐改法**：

```cpp
{
    std::lock_guard<std::mutex> lock1(imuLock);
    std::lock_guard<std::mutex> lock2(odoLock);

    if (imuQueue.empty()) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
            "[deskew] imuQueue is EMPTY. Is '%s' topic publishing?",
            imuTopic.c_str());
        return false;
    }

    double imu_front = stamp2Sec(imuQueue.front().header.stamp);
    double imu_back  = stamp2Sec(imuQueue.back().header.stamp);

    if (imu_front > timeScanCur) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
            "[deskew] IMU ahead of lidar: imu_front=%.3f > scan_cur=%.3f (diff=+%.3fs)",
            imu_front, timeScanCur, imu_front - timeScanCur);
        return false;
    }

    if (imu_back < timeScanEnd) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
            "[deskew] IMU behind lidar: imu_back=%.3f < scan_end=%.3f (diff=%.3fs). "
            "Are use_sim_time settings consistent?",
            imu_back, timeScanEnd, imu_back - timeScanEnd);
        return false;
    }
}
```

**`RCLCPP_WARN_THROTTLE` 的好处**：限制打印频率（每2秒最多一条），不刷屏，但问题出现时立刻可见。

---

#### 3. `is_dense` 检查直接 shutdown —— 过于激进

```cpp
// 当前代码（imageProjection.cpp 第355行）
if (laserCloudIn->is_dense == false) {
    RCLCPP_ERROR(get_logger(), "Point cloud is not in dense format...");
    rclcpp::shutdown();  // ← 直接杀死整个系统
}
```

Ouster OS1-64 在某些驱动版本下会设置 `is_dense = false`（即使点云中没有NaN）。直接shutdown会导致系统在第一帧就崩溃。

**推荐**：改为警告 + 跳过该帧：
```cpp
if (laserCloudIn->is_dense == false) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
        "Point cloud is not dense (is_dense=false). "
        "NaN points were already removed. Continuing...");
    // 不 shutdown，继续处理（NaN已在上一行 removeNaN 中去掉了）
}
```

---

#### 4. 拼写错误影响可维护性

`mapOptmization.cpp` → 应为 `mapOptimization.cpp`（少了一个 'i'）
`z_tollerance` / `rotation_tollerance` → 应为 `tolerance`（多了一个 'l'）

虽然功能不受影响，但这类错误会在代码搜索时造成困扰，体现出代码审查不严格。

---

## 第二部分：ROS2 开发核心规范

### 2.1 参数管理规范

```yaml
# ✅ 正确：每个参数只定义一次，有注释说明含义和单位
sensor: ouster           # lidar type: velodyne / ouster / livox
N_SCAN: 64               # lidar channels, Ouster OS1-64 = 64
Horizon_SCAN: 512        # horizontal resolution, OS1 mode: 512/1024/2048

# ❌ 错误：重复定义
sensor: velodyne
...
sensor: ouster
```

**检查脚本**（可加入CI/CD）：
```bash
# 检查YAML是否有重复key
python3 -c "
import yaml, sys, collections

def find_duplicates(d, parent=''):
    dups = []
    if isinstance(d, dict):
        seen = {}
        for k, v in d.items():
            if k in seen:
                dups.append(f'{parent}.{k}')
            seen[k] = v
            dups.extend(find_duplicates(v, f'{parent}.{k}'))
    return dups

with open(sys.argv[1]) as f:
    data = yaml.safe_load(f)
print('Duplicates:', find_duplicates(data) or 'None')
" config/params.yaml
```

---

### 2.2 QoS（服务质量）规范

**ROS2 中 QoS 不匹配是最常见的"数据无法接收"问题**：

| Publisher QoS | Subscriber QoS | 能否连接 |
|---|---|---|
| RELIABLE | RELIABLE | ✅ |
| BEST_EFFORT | BEST_EFFORT | ✅ |
| RELIABLE | BEST_EFFORT | ✅（降级） |
| **BEST_EFFORT** | **RELIABLE** | **❌ 不连接** |

**rosbag replay 默认用 RELIABLE**，而 LIO-SAM 原始代码订阅用的是 `BEST_EFFORT`，导致 bag 数据无法被接收。

**解决原则**：播放 bag 时，订阅方需要用 `RELIABLE`（已在 `utility.hpp` 第540行修复）。

---

### 2.3 use_sim_time 规范

使用 rosbag 复现实验时，**必须**确保：

```python
# launch 文件中每个 Node 都要加
parameters=[parameter_file, {'use_sim_time': True}]
```

```bash
# 播放 bag 时必须加 --clock
ros2 bag play your_bag/ --clock --rate 0.2
```

两者缺一不可。缺少任何一个，IMU 和点云的时间戳对不上，`deskewInfo()` 永远返回 false。

---

## 第三部分：LIO-SAM 算法流程图解

### 3.1 整体数据流

```
 /os1_cloud_node/points ──→  imageProjection  ──→  /deskew/cloud_info
 /os1_cloud_node/imu    ──┘                              ↓
                                               featureExtraction
                                                         ↓
                                              /feature/cloud_corner
                                              /feature/cloud_surface
                                                         ↓
                       imuPreintegration ──→  mapOptimization
                       (IMU预积分里程计)        ↓
                                          /mapping/odometry
                                          /mapping/cloud_registered (RViz显示)
```

### 3.2 为什么需要 `use_sim_time`？

```
系统时钟（wall clock）:  10:00:00.000
bag 内时间戳:            1780755069.561  ← 2026年的Unix时间戳

差距 ≈ 56年

imageProjection 查询: "给我 1780755069.0 ~ 1780755069.1 之间的IMU"
imuQueue 中的数据时间: 10:00:00.100 (系统时间, 约 1749462xxx)

→ 完全找不到 → "Waiting for IMU data..."
```

启用 `use_sim_time: true` 后，所有节点的时钟都改为监听 `/clock` 话题（由 bag 播放器发布），时间戳统一，问题消失。

---

### 3.3 去畸变（Deskewing）原理

激光雷达一帧扫描需要 100ms（10Hz），在这 100ms 内机器人在运动，导致点云"拉伸变形"：

```
帧开始 t=0ms:   ●●●● （这些点记录的是机器人在位置A时的方向）
帧中间 t=50ms:  ●●●● （这些点记录的是机器人在位置B时的方向）
帧结束 t=100ms: ●●●● （这些点记录的是机器人在位置C时的方向）

不去畸变：A/B/C 的点混在一起，坐标系不一致 → 点云模糊
去畸变：把B、C的点变换回A的坐标系 → 点云清晰
```

IMU 数据（200Hz）在每 0.5ms 内积分，得到 t=0 到 t=100ms 的精确旋转量，用于做这个变换。

---

## 第四部分：调试工作流（SLAM 专项）

### 4.1 标准调试顺序（Bottom-Up）

永远从数据源开始向上排查，不要从症状开始：

```
步骤1: 确认 bag 数据正确
  ros2 bag info <bag_path>
  → 检查 topic 名、message 数量、持续时间

步骤2: 确认数据在流动
  ros2 topic hz /os1_cloud_node/points   → 应该约 10 Hz
  ros2 topic hz /os1_cloud_node/imu      → 应该约 200 Hz

步骤3: 确认 QoS 匹配
  ros2 topic info /os1_cloud_node/points -v
  → Publisher 和 Subscriber 的 Reliability 必须匹配

步骤4: 确认时间戳对齐
  ros2 topic echo /os1_cloud_node/imu --once | grep "sec:"
  ros2 topic echo /os1_cloud_node/points --once | grep "sec:"
  → 两个 sec 值应该相差 < 1 秒

步骤5: 确认中间结果
  ros2 topic hz /dsw_lio_sam/deskew/cloud_deskewed  → 应该约 10 Hz
  ros2 topic hz /dsw_lio_sam/feature/cloud_corner   → 应该约 10 Hz
  → 哪一步断了，问题就在那一步
```

### 4.2 RViz2 点云不显示的排查树

```
RViz2 没有点云
├── Fixed Frame 报错？
│   ├── 是 → TF 链断了 → 检查 mapOptimization 是否在运行
│   └── 否 → 继续往下
├── PointCloud2 Status: Warn/Error？
│   ├── "No msgs received" → 话题没数据 → ros2 topic hz 检查
│   └── "Transform Failed" → frame_id 不匹配 → 检查 lidarFrame 参数
└── 什么都正常但就是不显示？
    └── 数据在，但相机视角不对 → RViz "Reset" 或调整 Target Frame
```

---

## 第五部分：实验记录规范（论文级）

### 5.1 目录结构

```
experiments/
├── 2026-06-09_newer_college_short/
│   ├── config_snapshot.yaml    ← 实验时的完整参数备份（不是软链接）
│   ├── run.log                 ← 终端输出完整日志
│   ├── results/
│   │   ├── trajectory.tum      ← 输出轨迹（TUM格式）
│   │   └── map.pcd             ← 输出地图
│   └── eval/
│       ├── ape.txt             ← evo 评估结果（APE）
│       └── rpe.txt             ← evo 评估结果（RPE）
```

### 5.2 轨迹评估工具：evo

```bash
# 安装
pip install evo

# APE 评估（绝对位姿误差，衡量全局精度）
evo_ape tum ground_truth.tum estimated.tum -a --plot

# RPE 评估（相对位姿误差，衡量局部一致性）
evo_rpe tum ground_truth.tum estimated.tum -a --plot --delta 1 --delta_unit m
```

Newer College ground truth 在 `registered_poses.csv`，需转换格式：
```python
# csv转TUM格式：sec.nsec tx ty tz qx qy qz qw
import pandas as pd
df = pd.read_csv("registered_poses.csv")
df["timestamp"] = df["sec"] + df["nsec"] * 1e-9
df[["timestamp","x","y","z","qx","qy","qz","qw"]].to_csv(
    "ground_truth.tum", sep=" ", index=False, header=False)
```

---

## 第六部分：下一步代码开发计划

按照论文方案，接下来要实现的核心改动：

### Phase 1（当前）：跑通原始 LIO-SAM ✅ 进行中
- [x] 包名重命名 dsw_lio_sam
- [x] 语义标签接口（PointTypeL, CloudInfo.msg）
- [x] IMU 6轴适配（identity quaternion）
- [x] QoS 修复（RELIABLE）
- [x] use_sim_time 配置
- [ ] **跑通第一帧点云显示** ← 当前卡点

### Phase 2：语义权重融入 ICP
- featureExtraction 输出带 label 的 PointTypeL
- mapOptimization 中逐点加权 `w_i = 1 + α × (S(c_i) - 0.5)`

### Phase 3：退化检测与处理
- 双因子退化检测（点云退化 + 几何退化）
- 退化时融合更多 IMU 约束

### Phase 4：实验评估
- Newer College 完整序列跑通
- evo 评估 ATE/RPE
- 与原始 LIO-SAM 对比

---

*文档版本：v1.0 | 2026-06-09 | 基于实际代码审查生成*
