# -*- coding: utf-8 -*-
# 基于官方 test_simple.py 的最小改动版
# 1. 支持任意 model 路径   2. 结果固定输出到指定目录

from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torchvision import transforms

import networks
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist
from evaluate_depth import STEREO_SCALE_FACTOR


def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing function for Monodepthv2 models.')

    parser.add_argument('--image_path', type=str, required=True,
                        help='path to a test image or folder of images')
    # ========== 改动① 去掉 choices 限制 ==========
    parser.add_argument('--model_name', type=str, required=True,
                        help='name of a pretrained model to use, or absolute path to your own weights folder')
    # ============================================
    parser.add_argument('--ext', type=str, default='jpg',
                        help='image extension to search for in folder')
    parser.add_argument('--no_cuda', action='store_true',
                        help='if set, disables CUDA')
    parser.add_argument('--pred_metric_depth', action='store_true',
                        help='if set, predicts metric depth instead of disparity. '
                             '(This only makes sense for stereo-trained KITTI models).')
    return parser.parse_args()


def test_simple(args):
    assert args.model_name is not None, \
        "You must specify the --model_name parameter; see README.md for an example"

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    # ========== 改动② 支持绝对路径权重文件夹 ==========
    model_name = args.model_name
    if os.path.isdir(model_name):
        model_path = model_name          # 用户直接给路径
    else:
        download_model_if_doesnt_exist(model_name)
        model_path = os.path.join("models", model_name)
    # ===================================================

    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))
    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)
    depth_decoder.to(device)
    depth_decoder.eval()

    # FINDING INPUT IMAGES
    if os.path.isfile(args.image_path):
        paths = [args.image_path]
    elif os.path.isdir(args.image_path):
        paths = glob.glob(os.path.join(args.image_path, '*.{}'.format(args.ext)))
    else:
        raise Exception("Can not find args.image_path: {}".format(args.image_path))
    print("-> Predicting on {:d} test images".format(len(paths)))

    # ========== 改动③ 固定输出目录 ==========
    output_directory = "/home/zly/tmp/test"
    os.makedirs(output_directory, exist_ok=True)
    # =======================================

    with torch.no_grad():
        for idx, image_path in enumerate(paths):
            if image_path.endswith("_disp.jpg"):
                continue

            input_image = pil.open(image_path).convert('RGB')
            original_width, original_height = input_image.size
            input_image_resized = input_image.resize((feed_width, feed_height), pil.LANCZOS)
            input_tensor = transforms.ToTensor()(input_image_resized).unsqueeze(0).to(device)

            features = encoder(input_tensor)
            outputs = depth_decoder(features)

            disp = outputs[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)

            # save numpy
            output_name = os.path.splitext(os.path.basename(image_path))[0]
            scaled_disp, depth = disp_to_depth(disp, 0.1, 100)
            if args.pred_metric_depth:
                name_dest_npy = os.path.join(output_directory, "{}_depth.npy".format(output_name))
                metric_depth = STEREO_SCALE_FACTOR * depth.cpu().numpy()
                np.save(name_dest_npy, metric_depth)
            else:
                name_dest_npy = os.path.join(output_directory, "{}_disp.npy".format(output_name))
                np.save(name_dest_npy, scaled_disp.cpu().numpy())

            # save colormapped depth image
            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)

            
            # ===== 新增：生成原图 + 深度图对比图（不修改原注释）=====
            # depth_im = pil.fromarray(colormapped_im)
            # orig_im = input_image.resize((original_width, original_height), pil.LANCZOS)

            # compare_im = pil.new(
            #     "RGB",
            #     (original_width * 2, original_height)
            # )
            # compare_im.paste(orig_im, (0, 0))
            # compare_im.paste(depth_im, (original_width, 0))

            # name_dest_im = os.path.join(
            #     output_directory,
            #     "{}_compare.jpeg".format(output_name)
            # )
            # compare_im.save(name_dest_im)
            # ======================================================
            
            # ===== 新增：生成原图 + 深度图上下对比图（不修改原注释）=====
            depth_im = pil.fromarray(colormapped_im)
            orig_im = input_image.resize((original_width, original_height), pil.LANCZOS)

            # 上下拼接
            compare_im = pil.new(
                "RGB",
                (original_width, original_height * 2)  # 宽度不变，高度翻倍
            )
            compare_im.paste(orig_im, (0, 0))  # 原图在上
            compare_im.paste(depth_im, (0, original_height))  # 深度图在下

            name_dest_im = os.path.join(
                output_directory,
                "{}_compare.jpeg".format(output_name)
            )
            compare_im.save(name_dest_im)
            # ======================================================    

            print("   Processed {:d} of {:d} images - saved predictions to:".format(idx + 1, len(paths)))
            print("   - {}".format(name_dest_im))
            print("   - {}".format(name_dest_npy))

    print('-> Done!')


if __name__ == '__main__':
    args = parse_args()
    test_simple(args)
