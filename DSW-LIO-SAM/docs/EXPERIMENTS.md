# Newer College 四组对照实验

四组实验必须使用同一个 ROS2 bag、播放速率和代码版本。统一入口会自动选择参数文件和语义来源，避免同时启动两个语义发布器。

运行前可检查四套参数是否出现非预期差异：

```bash
python3 tools/validate_experiment_configs.py
```

## Newer College extrinsic calibration

The four experiment files use the same Ouster OS1 Gen1 calibration. In the
official Frontier/Ouster URDF, `os_sensor -> os_lidar` is
`xyz=(0,0,0.03618), rpy=(0,0,pi)` and `os_sensor -> os_imu` is
`xyz=(0.006253,-0.011775,0.007645), rpy=(0,0,0)`. Therefore the IMU-to-LiDAR
transform used by this package is `T_LI = T_SL^-1 * T_SI`:

```text
translation: (-0.006253, 0.011775, -0.028535) m
rotation:    [ -1  0  0; 0 -1  0; 0 0 1 ]
```

Before replaying a bag, check that the driver has not already applied this
transform to the IMU message. The frame IDs and static transform must be
consistent with the selected parameters:

```bash
ros2 topic echo /os1_cloud_node/imu --once
ros2 topic echo /os1_cloud_node/points --once
ros2 run tf2_ros tf2_echo os1_lidar os1_imu
```

The repository's `robot.urdf.xacro` provides the internal `base_link` to
`lidar_link` lookup required by LIO-SAM; the numeric IMU/LiDAR calibration is
applied in `imuConverter` and `imuPreintegration.cpp`, so an additional static
publisher for the same pair would apply the calibration twice.

## 1. Original LIO-SAM

使用 PCL 质心体素滤波、等权最小二乘和原始二值退化传递：

```bash
ros2 launch dsw_lio_sam experiment.launch.py experiment:=original
ros2 bag play <bag-dir> --clock --rate 0.2
```

## 2. DSW 无语义消融基线

运行 DSW 公共算法路径，但关闭语义订阅与权重，用于隔离非语义改动：

```bash
ros2 launch dsw_lio_sam experiment.launch.py experiment:=baseline
ros2 bag play <bag-dir> --clock --rate 0.2
```

## 3. Pseudo semantic

启动几何规则伪语义发布器，使用与真实语义组相同的 `alpha=0.5` 和退化阈值：

```bash
ros2 launch dsw_lio_sam experiment.launch.py experiment:=pseudo
ros2 bag play <bag-dir> --clock --rate 0.2
```

## 4. Cylinder3D semantic

使用预计算且连续编号的 `000000.bin ...` 语义帧：

```bash
ros2 launch dsw_lio_sam experiment.launch.py \
  experiment:=true \
  semantic_dir:=/absolute/path/to/rooster_semantic_001
ros2 bag play <bag-dir> --clock --rate 0.2
```

语义回放节点必须先于 bag 播放启动。该模式要求每帧语义匹配率至少为 95%，否则对应 LiDAR 帧会被拒绝，避免无标签帧混入真实语义结果。

## 结果解释

四组比较分别表示：`original vs baseline` 衡量 DSW 非语义改动，`baseline vs pseudo` 衡量几何伪语义，`baseline vs true` 衡量 Cylinder3D 语义，`pseudo vs true` 衡量语义来源差异。排序只能作为待验证假设，不能由代码预设。至少报告 APE、RPE、轨迹完整率、被拒绝帧数和三次重复运行的均值/标准差。`tools/drift.py` 的首尾距离仅适用于起终点重合的闭环序列，不能替代带真值的 APE/RPE。

当前 Cylinder3D 权重文件是 SemanticKITTI 预训练模型；因此论文中宜称为“Cylinder3D 预测语义”，不能称为 Newer College 真值语义，除非后续使用 Newer College 标注完成微调或提供人工真值。

## ATE/RPE 评估

保存地图后，使用 Newer College 官方 `registered_poses.csv` 和 LIO-SAM 输出的
`transformations.pcd` 计算轨迹误差：

```bash
python3 tools/evaluate_newer_college.py \
  /path/to/registered_poses.csv \
  ~/dsw_lio_sam_results/baseline_no_semantic_rooster_001/transformations.pcd \
  --output-json ~/dsw_lio_sam_results/baseline_no_semantic_rooster_001/metrics.json
```

第一个参数也可以直接传 Newer College 官方 `.zip` 数据包，脚本会读取其中唯一的
`ground_truth/registered_poses.csv`。评估会在估计轨迹的时间戳处插值真值，执行无尺度
SE(3) 对齐，并报告平移 ATE 和默认间隔为 1 米的 SE(3) RPE（平移米、旋转度）。
可用 `--max-time-gap`、`--rpe-delta` 和 `--rpe-tolerance` 调整关联参数；四组实验必须使用
完全相同的参数。

对同一个 Newer College 序列，依次运行四组实验后，可用下面的命令做统一对比（只替换
`<bag-dir>`、`<registered_poses.csv>` 和 `semantic_dir`）：

```bash
python3 tools/validate_experiment_configs.py

ros2 launch dsw_lio_sam experiment.launch.py experiment:=original
ros2 bag play <bag-dir> --clock --rate 0.2

ros2 launch dsw_lio_sam experiment.launch.py experiment:=baseline
ros2 bag play <bag-dir> --clock --rate 0.2

ros2 launch dsw_lio_sam experiment.launch.py experiment:=pseudo
ros2 bag play <bag-dir> --clock --rate 0.2

ros2 launch dsw_lio_sam experiment.launch.py \
  experiment:=true semantic_dir:=/absolute/path/to/rooster_semantic_001
ros2 bag play <bag-dir> --clock --rate 0.2
```

```bash
GT=<registered_poses.csv>
python3 tools/evaluate_newer_college.py "$GT" \
  ~/dsw_lio_sam_results/original_lio_sam_rooster_001/transformations.pcd \
  --output-json ~/dsw_lio_sam_results/original_lio_sam_rooster_001/metrics.json
python3 tools/evaluate_newer_college.py "$GT" \
  ~/dsw_lio_sam_results/baseline_no_semantic_rooster_001/transformations.pcd \
  --output-json ~/dsw_lio_sam_results/baseline_no_semantic_rooster_001/metrics.json
python3 tools/evaluate_newer_college.py "$GT" \
  ~/dsw_lio_sam_results/pseudo_semantic_rooster_001/transformations.pcd \
  --output-json ~/dsw_lio_sam_results/pseudo_semantic_rooster_001/metrics.json
python3 tools/evaluate_newer_college.py "$GT" \
  ~/dsw_lio_sam_results/cylinder3d_true_semantic_rooster_001/transformations.pcd \
  --output-json ~/dsw_lio_sam_results/cylinder3d_true_semantic_rooster_001/metrics.json
```

上述评估结果才是 `original`、`baseline`、`pseudo` 和 `true` 的可比指标；不要只用
`tools/compare_trajectories.py` 的首尾距离替代 ATE/RPE。
