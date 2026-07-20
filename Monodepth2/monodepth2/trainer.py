## 1、头部导入与版权声明
# Copyright Niantic 2019. Patent Pending. All rights reserved.
# 版权归 Niantic 所有 2019。专利申请中。保留所有权利。
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# 本软件根据 Monodepth2 许可条款授权，仅允许非商业用途，完整条款
# available in the LICENSE file.
# 可在 LICENSE 文件中查看。

from __future__ import absolute_import, division, print_function  # __future__的三个导入是为了兼容 Python2 的语法（Monodepth2 开发时 Python2 还在使用）

import numpy as np  # 数值计算
import time  # 计时

# PyTorch 库：核心深度学习框架，包含张量操作、优化器、数据加载、函数式接口（F）
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import json

from utils import *  # 工具函数（如深度误差计算、KITTI 数据处理）
from kitti_utils import *
from layers import *  # 自定义层（如反投影、3D 投影、SSIM 损失层）

import datasets  # 数据集类（KITTIRAW/KITTIOdom）
import networks  # 网络模型（ResNet 编码器、深度解码器、位姿解码器）
from IPython import embed  # embed 是 IPython 的调试工具，可在代码中插入embed()进入交互式调试

## 2、Trainer 类的初始化方法__init__
class Trainer:  # Trainer 类
    def __init__(self, options):  # 初始化
        ## （1）基础配置与校验
        self.opt = options  # 外部传入的训练配置（如学习率、批次大小、图像尺寸、数据集路径等），通常由argparse解析得到
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)  # 日志和模型保存的路径（拼接日志目录 + 模型名）

        # checking height and width are multiples of 32
        # 图像的高和宽必须是 32 的倍数
        # 因为 ResNet 编码器会进行 5 次下采样（2^5=32），保证解码器上采样时尺寸匹配
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        ## （2）模型相关初始化
        self.models = {}  # models字典：存储所有模型（编码器、深度解码器、位姿网络等），统一管理所有子模型，方便后续训练 / 验证时切换模式（train/eval）
        self.parameters_to_train = []  # 收集所有需要优化的参数，后续传入优化器

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")  # 自动选择 GPU（如果可用）

        # 多尺度 / 帧相关参数：Monodepth2 采用多尺度训练提升精度，同时使用连续帧的位姿估计来完成重投影损失计算
        self.num_scales = len(self.opt.scales)  # 多尺度训练的尺度数
        self.num_input_frames = len(self.opt.frame_ids)  # 输入帧的数量
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames  # 位姿网络的输入帧数量

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"  # 帧ID必须以0（当前帧）开头

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])  # 位姿网络的使用条件，立体视觉模式下如果只输入当前帧，不需要位姿网络（因为立体视觉的视差可以直接计算深度）

        ## （3）加载深度网络（编码器 + 解码器）
        if self.opt.use_stereo:  
            self.opt.frame_ids.append("s")  # 立体视觉扩展：如果使用立体视觉，在帧 ID 中添加 "s"，表示左 / 右相机的帧--添加"s"表示立体对的另一帧

        # ResnetEncoder：深度网络的编码器，基于 ResNet18/50，输出不同尺度的特征图。
        self.models["encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained")  # ResNet编码器（18/50层，预训练/随机初始化）
        self.models["encoder"].to(self.device)  # 移到指定设备
        self.parameters_to_train += list(self.models["encoder"].parameters())  # 添加参数到训练列表

        # DepthDecoder：深度网络的解码器，接收编码器的特征图，输出多尺度的视差（disp）图，视差与深度成反比。
        self.models["depth"] = networks.DepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales)  # 深度解码器（输入编码器的通道数，输出多尺度视差）
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        ## （4）加载位姿网络
        if self.use_pose_net:
            if self.opt.pose_model_type == "separate_resnet":  # separate_resnet：独立的 ResNet 编码器 + 位姿解码器，性能最好，计算量稍大。
                # 独立的ResNet编码器用于位姿估计
                self.models["pose_encoder"] = networks.ResnetEncoder(
                    self.opt.num_layers,
                    self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames)

                self.models["pose_encoder"].to(self.device)
                self.parameters_to_train += list(self.models["pose_encoder"].parameters())

                # 位姿解码器
                self.models["pose"] = networks.PoseDecoder(
                    self.models["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)

            elif self.opt.pose_model_type == "shared":  # shared：共享深度编码器的特征，减少参数，但性能略降。
                # 共享深度编码器的特征用于位姿估计（Monodepth1的方式）
                self.models["pose"] = networks.PoseDecoder(
                    self.models["encoder"].num_ch_enc, self.num_pose_frames)

            elif self.opt.pose_model_type == "posecnn":  # posecnn：轻量化的 CNN，适合快速训练 / 推理。
                # 简单的CNN用于位姿估计（轻量化）
                self.models["pose"] = networks.PoseCNN(
                    self.num_input_frames if self.opt.pose_model_input == "all" else 2)

            self.models["pose"].to(self.device)
            self.parameters_to_train += list(self.models["pose"].parameters())
        
        ## （5）加载预测掩码网络（可选）
        if self.opt.predictive_mask:  # predictive_mask：可选的掩码网络，用于抑制重投影损失中的无效区域（如运动物体），与自动掩码（automasking）互斥。
            assert self.opt.disable_automasking, \
                "When using predictive_mask, please disable automasking with --disable_automasking"

            # Our implementation of the predictive masking baseline has the the same architecture
            # as our depth decoder. We predict a separate mask for each source frame.
            # 预测掩码的解码器（与深度解码器结构相同，输出每个源帧的掩码）
            self.models["predictive_mask"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc, self.opt.scales,
                num_output_channels=(len(self.opt.frame_ids) - 1))  # 掩码解码器的输出通道数等于源帧的数量（除当前帧外的帧），每个通道对应一个源帧的掩码。
            self.models["predictive_mask"].to(self.device)
            self.parameters_to_train += list(self.models["predictive_mask"].parameters())

        ## （6）优化器与学习率调度器
        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)  # Adam 优化器：深度学习中最常用的优化器，收敛速度快。
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(  # StepLR 调度器：按固定步数衰减学习率，是最基础的学习率调整策略--每隔指定步数学习率乘以0.1
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        ## （7）加载预训练模型
        if self.opt.load_weights_folder is not None:  # 支持加载已训练的模型权重，继续训练或微调。
            self.load_model()  # 调用自定义的load_model方法加载预训练权重

        ## （8）打印训练信息
        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)
        
        ## （9）数据集加载
        # data
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset}  # 数据集字典：支持KITTI原始数据和KITTI里程计数据
        self.dataset = datasets_dict[self.opt.dataset]
        
        # 读取训练/验证集的文件列表（splits文件夹下的txt文件）
        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.opt.png else '.jpg'  # 图像格式（png/jpg）

        # 计算总训练步数
        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        # 训练集加载器
        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext=img_ext)
        self.train_loader = DataLoader(  # DataLoader：PyTorch 的数据集加载器，支持多线程、批次加载、随机打乱（训练集）。
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        # 验证集加载器
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)  # val_iter：验证集的迭代器，用于在训练过程中随机取一个批次进行验证，而不是遍历整个验证集（节省时间）。

        ## （10）日志工具与损失层初始化
        # TensorBoard日志写入器（训练/验证分开）
        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))  # SummaryWriter：将训练过程中的损失、图像写入 TensorBoard，方便可视化。

        # SSIM损失层（用于重投影损失）
        if not self.opt.no_ssim:
            self.ssim = SSIM()  # SSIM 层：结构相似性指数，用于计算重投影损失（比单纯的 L1 损失更鲁棒）。
            self.ssim.to(self.device)
        
        # 反投影和3D投影层（多尺度）
        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)
            # BackprojectDepth/Project3D：自定义层，实现深度图到 3D 点云的反投影，以及 3D 点云到另一帧像素的投影（核心操作，用于重投影损失）。
            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)
        
        # 深度度量指标名称（用于评估深度精度）--绝对相对误差（abs_rel）、均方根误差（rms）、准确率（a1/a2/a3）。
        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        ## （11）打印数据集信息并保存配置
        # 打印数据集的分割方式和样本数量。
        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        # save_opts：将训练配置保存到日志目录，方便后续复现实验。
        self.save_opts()  # 保存配置到json文件

    ## 3、模型模式切换方法（train/eval）
    # 这是 PyTorch 中训练 / 验证的标准操作，必须在对应阶段调用。
    def set_train(self):  # set_train：将所有模型切换到训练模式（启用 Dropout、BatchNorm 的训练行为）
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):  # set_eval：将所有模型切换到验证模式（禁用 Dropout、使用 BatchNorm 的移动平均）。
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    ## 4、核心训练流程（train/run_epoch）
    ## （1）train 方法：总训练循环
    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()  # 初始化 epoch、step、开始时间。
        for self.epoch in range(self.opt.num_epochs):  # 遍历所有 epoch，调用run_epoch执行单轮训练。
            self.run_epoch()  # 运行单轮epoch
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()  # 每隔指定 epoch 保存模型权重。
    
    ## （2）run_epoch 方法：单轮 epoch 的训练与验证
    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        self.model_lr_scheduler.step()  # 学习率调度器更新（注意：原代码中step放在epoch开头，可能与预期不符，通常放在结尾）

        print("Training")
        self.set_train()  # 切换到训练模式

        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)  # 处理批次数据：前向传播+损失计算

            # 反向传播
            self.model_optimizer.zero_grad()  # 梯度清零
            losses["loss"].backward()  # 损失反向传播
            self.model_optimizer.step()  # 优化器更新参数

            duration = time.time() - before_op_time  # 计算批次处理时间

            # log less frequently after the first 2000 steps to save time & disk space
            # 日志记录策略：前2000步每log_frequency记录，之后每2000步记录（节省时间和磁盘空间）
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)  # 打印终端日志

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)  # 计算深度度量指标

                self.log("train", inputs, outputs, losses)  # 写入TensorBoard日志
                self.val()  # 执行单批次验证

            self.step += 1  # 全局步数+1

    ## 5. 批次处理核心方法（process_batch）
    def process_batch(self, inputs):  # 前向传播的核心方法
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        if self.opt.pose_model_type == "shared":
            # 共享编码器模式：所有帧都通过深度编码器提取特征
            # If we are using a shared encoder for both depth and pose (as advocated
            # in monodepthv1), then all images are fed separately through the depth encoder.
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])
            all_features = self.models["encoder"](all_color_aug)
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features]

            features = {}
            for i, k in enumerate(self.opt.frame_ids):
                features[k] = [f[i] for f in all_features]

            outputs = self.models["depth"](features[0])  # 深度解码器处理当前帧特征
        else:
            # Otherwise, we only feed the image with frame_id 0 through the depth encoder
            # 非共享模式：仅当前帧通过深度编码器
            features = self.models["encoder"](inputs["color_aug", 0, 0])
            outputs = self.models["depth"](features)

        # 预测掩码
        if self.opt.predictive_mask:
            outputs["predictive_mask"] = self.models["predictive_mask"](features)

        # 位姿预测
        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, features))

        # 生成重投影图像
        self.generate_images_pred(inputs, outputs)
        # 计算损失
        losses = self.compute_losses(inputs, outputs)

        return outputs, losses

    ## 6、位姿预测方法（predict_poses）
    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # 成对输入模式：每个源帧与当前帧成对输入位姿网络，输出两两之间的位姿。
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    # 保持时间顺序：负帧（上一帧）则交换顺序
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    # 不同位姿网络的输入处理
                    if self.opt.pose_model_type == "separate_resnet":
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)

                    # 位姿网络输出：轴角（旋转）+ 平移
                    axisangle, translation = self.models["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    # 转换为位姿变换矩阵（如果是负帧，反转矩阵）
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            # 多帧输入模式：所有帧一起输入位姿网络，输出当前帧与所有源帧的位姿。
            if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
                pose_inputs = torch.cat(
                    [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)

                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs = [self.models["pose_encoder"](pose_inputs)]

            elif self.opt.pose_model_type == "shared":
                pose_inputs = [features[i] for i in self.opt.frame_ids if i != "s"]  # 立体视觉跳过：帧 ID 为 "s" 的立体帧不参与位姿预测（因为立体帧的位姿是已知的相机外参）。

            axisangle, translation = self.models["pose"](pose_inputs)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(  # 位姿参数转换：轴角（表示旋转）和平移向量转换为 4x4 的位姿变换矩阵（transformation_from_parameters），这是 3D 几何中的标准操作。
                        axisangle[:, i], translation[:, i])

        return outputs

    ## 7、验证方法（val）
    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()  # 切换到验证模式
        try:
            ###############################
            # inputs = self.val_iter.next()
            ###############################
            # 原代码的next()写法在Python3中已更新，这里修复为next()
            inputs = next(self.val_iter)
            
        except StopIteration:
            # 验证集迭代器耗尽时，重新初始化
            self.val_iter = iter(self.val_loader)
            inputs = next(self.val_iter)

        with torch.no_grad():  # 验证阶段不需要计算梯度，这是 PyTorch 的标准操作，能大幅减少内存占用和计算时间。
            outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses  # 释放内存

        self.set_train()  # 切回训练模式

    ## 8、重投影图像生成方法（generate_images_pred）
    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]  # 多尺度视差
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                # 非v1模式：将视差上采样到原始尺寸（源帧使用原始尺寸）
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            # 视差转深度（disp与depth成反比）
            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[("depth", 0, scale)] = depth  # 保存深度图

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]  # 立体帧的位姿（已知的相机外参）
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]  # 位姿网络预测的变换矩阵

                # from the authors of https://arxiv.org/abs/1712.00175
                # PoseCNN的特殊处理：平移向量乘以平均逆深度
                if self.opt.pose_model_type == "posecnn":

                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                # 步骤1：深度图反投影为3D点云（相机坐标系）
                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                # 步骤2：3D点云投影到源帧的像素坐标系
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords  # 保存像素坐标

                # 步骤3：根据像素坐标采样源帧图像，得到重投影图像
                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border")

                # 自动掩码：保存原始源帧图像（用于后续的掩码计算）
                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]

    ## 9、损失计算相关方法
    ## （1）重投影损失计算（compute_reprojection_loss）
    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        # 重投影损失：衡量重投影图像与当前帧图像的差异，是 Monodepth2 的核心损失。
        abs_diff = torch.abs(target - pred)  # L1损失的基础
        l1_loss = abs_diff.mean(1, True)  # 通道维度求平均

        if self.opt.no_ssim:
            reprojection_loss = l1_loss  # 仅使用L1损失
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)  # SSIM损失
            # SSIM+L1 加权：SSIM 捕捉结构相似性，L1 捕捉像素级差异，加权组合能提升损失的鲁棒性。
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss  # 加权组合

        return reprojection_loss

    ## （2）总损失计算（compute_losses）
    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]  # 当前帧图像（多尺度）
            target = inputs[("color", 0, source_scale)]  # 目标图像（原始尺寸）

            # 计算每个源帧的重投影损失
            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]  # 重投影图像
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)  # 拼接所有源帧的损失

            if not self.opt.disable_automasking:  # 自动掩码：使用原始源帧的重投影损失作为掩码
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]  # 原始源帧图像
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

            # 预测掩码：使用掩码网络的输出加权损失
            elif self.opt.predictive_mask:
                # use the predicted mask
                mask = outputs["predictive_mask"]["disp", scale]
                if not self.opt.v1_multiscale:
                    mask = F.interpolate(
                        mask, [self.opt.height, self.opt.width],
                        mode="bilinear", align_corners=False)

                reprojection_losses *= mask  # 掩码加权

                # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                # 掩码的正则化损失：推动掩码接近1
                weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
                loss += weighting_loss.mean()

            # 平均重投影损失（可选）
            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            # 自动掩码的核心逻辑：组合原始源帧和重投影的损失，取最小值
            if not self.opt.disable_automasking:
                # add random numbers to break ties
                # 添加微小随机数打破平局
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape, device=self.device) * 0.00001

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            # 取最小损失（自动掩码的关键：选择损失最小的区域）
            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            # 保存自动掩码的选择结果
            if not self.opt.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (
                    idxs > identity_reprojection_loss.shape[1] - 1).float()

            # 累加重投影损失
            loss += to_optimise.mean()

            # 平滑损失：推动视差图在空间上平滑（减少噪声）
            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)  # 归一化视差
            smooth_loss = get_smooth_loss(norm_disp, color)  # 基于图像梯度的平滑损失

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)  # 多尺度加权
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales  # 多尺度损失平均
        losses["loss"] = total_loss
        return losses

    ## （3）深度度量指标计算（compute_depth_losses）
    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]  # 原始尺度的深度预测
        # 上采样到KITTI原始尺寸（375x1242）
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()  # 分离梯度，不参与训练

        depth_gt = inputs["depth_gt"]  # 深度真值
        mask = depth_gt > 0  # 有效深度掩码（排除无真值的区域）

        # garg/eigen crop
        # Garg/Eigen裁剪：只计算图像中心区域的误差（KITTI的标准裁剪方式）
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        # 只保留有效区域的深度
        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        # 深度归一化：将预测深度的中值与真值对齐（Monodepth的标准操作，因为单目深度是尺度不确定的）
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)  # 限制深度范围

        depth_errors = compute_depth_errors(depth_gt, depth_pred)  # 计算深度误差指标

        for i, metric in enumerate(self.depth_metric_names):  # 将指标存入losses字典
            losses[metric] = np.array(depth_errors[i].cpu())

    ## 10、日志与模型保存 / 加载方法
    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration  # 每秒处理的样本数
        time_sofar = time.time() - self.start_time  # 已用时间
        # 剩余时间估计
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        # 打印训练的 epoch、批次、处理速度、损失、已用时间、剩余时间，方便用户监控训练进度。
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))  # sec_to_hm_str：自定义函数，将秒数转换为小时：分钟：秒的格式。

    ## （2）TensorBoard 日志写入（log）
    def log(self, mode, inputs, outputs, losses):  # 日志
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        # 写入损失值
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        # 写入图像（最多4张，避免日志过大）
        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids:
                    # 写入原始图像
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, self.step)
                    if s == 0 and frame_id != 0:
                        # 写入重投影图像
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, self.step)

                # 写入视差图（归一化到0-1）
                writer.add_image(  # 视差图需要归一化到 0-1 才能正确显示。
                    "disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), self.step)

                # 写入预测掩码
                if self.opt.predictive_mask:
                    for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
                        writer.add_image(
                            "predictive_mask_{}_{}/{}".format(frame_id, s, j),
                            outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
                            self.step)

                # 写入自动掩码
                elif not self.opt.disable_automasking:
                    writer.add_image(
                        "automask_{}/{}".format(s, j),
                        outputs["identity_selection/{}".format(s)][j][None, ...], self.step)

    ## （3）配置保存（save_opts）
    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()  # 复制配置的字典

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)  # 保存为json文件

    ## （4）模型保存（save_model）
    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        # 保存每个模型的权重
        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                # 保存编码器的尺寸和立体视觉配置（预测时需要）
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        # 保存优化器的状态（方便继续训练）
        # 保存每个模型的权重（state_dict），以及优化器的状态（方便继续训练）。
        # 编码器额外保存图像尺寸和立体视觉配置，因为预测时需要这些参数来处理输入图像。
        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    ## （5）模型加载（load_model）
    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        # 加载指定的模型权重，过滤掉不匹配的参数（避免因模型结构变化导致的错误）。
        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            # 过滤掉不匹配的参数（如尺寸不同的层）
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        # 加载优化器状态，方便继续训练。
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")

#  核心设计思路总结：
#    1、多尺度训练：使用多个尺度的特征图和损失，提升深度估计的精度和鲁棒性。
#    2、重投影损失：利用连续帧的位姿估计，将源帧重投影到当前帧，计算重投影损失（SSIM+L1），这是单目深度估计的核心监督信号。
#    3、自动掩码（Automasking）：自动忽略运动物体、遮挡区域的重投影误差，解决动态场景下的损失计算问题。
#    4、位姿网络的多样性：支持独立 ResNet、共享编码器、PoseCNN 三种位姿网络，兼顾性能和效率。
#    5、高效的训练策略：单批次验证、日志记录频率调整、内存优化（torch.no_grad ()），提升训练效率。
#    6、尺度归一化：单目深度估计的尺度不确定性通过中值对齐解决，保证深度指标的可比较性。
