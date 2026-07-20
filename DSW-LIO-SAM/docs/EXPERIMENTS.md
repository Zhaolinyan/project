# Newer College 三组对照实验

三组实验必须使用同一个 ROS2 bag、播放速率和代码版本。统一入口会自动选择参数文件和语义来源，避免同时启动两个语义发布器。

运行前可检查三套参数是否出现非预期差异：

```bash
python3 tools/validate_experiment_configs.py
```

## 1. Baseline

原始 LIO-SAM 特征与等权最小二乘，保留原始二值退化投影：

```bash
ros2 launch dsw_lio_sam experiment.launch.py experiment:=baseline
ros2 bag play <bag-dir> --clock --rate 0.2
```

## 2. Pseudo semantic

启动几何规则伪语义发布器，使用与真实语义组相同的 `alpha=0.5` 和退化阈值：

```bash
ros2 launch dsw_lio_sam experiment.launch.py experiment:=pseudo
ros2 bag play <bag-dir> --clock --rate 0.2
```

## 3. Cylinder3D semantic

使用预计算且连续编号的 `000000.bin ...` 语义帧：

```bash
ros2 launch dsw_lio_sam experiment.launch.py \
  experiment:=true \
  semantic_dir:=/absolute/path/to/rooster_semantic_001
ros2 bag play <bag-dir> --clock --rate 0.2
```

语义回放节点必须先于 bag 播放启动。该模式要求每帧语义匹配率至少为 95%，否则对应 LiDAR 帧会被拒绝，避免无标签帧混入真实语义结果。

## 结果解释

预期排序（baseline 漂移最大、pseudo 次之、Cylinder3D 最小）只能作为待验证假设，不能由代码预设。至少报告 APE、RPE、轨迹完整率、被拒绝帧数和三次重复运行的均值/标准差。`tools/drift.py` 的首尾距离仅适用于起终点重合的闭环序列，不能替代带真值的 APE/RPE。

当前 Cylinder3D 权重文件是 SemanticKITTI 预训练模型；因此论文中宜称为“Cylinder3D 预测语义”，不能称为 Newer College 真值语义，除非后续使用 Newer College 标注完成微调或提供人工真值。
