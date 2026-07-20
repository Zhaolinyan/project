# -*- coding:utf-8 -*-
# author: Xinge
# @file: load_save_util.py 

import torch


def adapt_spconv_weight(value, target):
    if len(value.shape) == 5 and len(target.shape) == 5:
        converted = value.permute(4, 0, 1, 2, 3).contiguous()
        if converted.shape == target.shape:
            return converted
    return value


def load_checkpoint(model_load_path, model):
    my_model_dict = model.state_dict()
    pre_weight = torch.load(model_load_path, map_location='cpu')

    part_load = {}
    match_size = 0
    nomatch_size = 0
    for k in pre_weight.keys():
        value = pre_weight[k]
        if k in my_model_dict:
            value = adapt_spconv_weight(value, my_model_dict[k])
        if k in my_model_dict and my_model_dict[k].shape == value.shape:
            # print("loading ", k)
            match_size += 1
            # Fix NaN/Inf weights in pretrained checkpoint
            if torch.isnan(value).any() or torch.isinf(value).any():
                value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
            part_load[k] = value
        else:
            nomatch_size += 1

    print("matched parameter sets: {}, and no matched: {}".format(match_size, nomatch_size))

    my_model_dict.update(part_load)
    model.load_state_dict(my_model_dict)

    return model


def load_checkpoint_1b1(model_load_path, model):
    my_model_dict = model.state_dict()
    pre_weight = torch.load(model_load_path, map_location='cpu')

    part_load = {}
    match_size = 0
    nomatch_size = 0

    pre_weight_list = [*pre_weight]
    my_model_dict_list = [*my_model_dict]

    for idx in range(len(pre_weight_list)):
        key_ = pre_weight_list[idx]
        key_2 = my_model_dict_list[idx]
        value_ = pre_weight[key_]
        value_ = adapt_spconv_weight(value_, my_model_dict[key_2])
        if my_model_dict[key_2].shape == value_.shape:
            # print("loading ", k)
            match_size += 1
            part_load[key_2] = value_
        else:
            print(key_)
            print(key_2)
            nomatch_size += 1

    print("matched parameter sets: {}, and no matched: {}".format(match_size, nomatch_size))

    my_model_dict.update(part_load)
    model.load_state_dict(my_model_dict)

    return model
