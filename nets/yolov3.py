# -*- coding: utf-8 -*-
# @Time : 2020/6/8 9:31
# @Author : Jiwei Qin
# @FileName: yolov3.py
# @Software: PyCharm

import torch
import torch.nn as nn
from collections import OrderedDict
from nets.darknet import darknet53


# DBL模块
def conv2d(filter_in, filter_out, kernel_size):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=1, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))


# 输出层
def make_last_layers(filter_list, in_filters, out_filter):
    m = nn.ModuleList([
        conv2d(in_filters, filter_list[0], 1),
        conv2d(filter_list[0], filter_list[1], 3),
        conv2d(filter_list[1], filter_list[0], 1),
        conv2d(filter_list[0], filter_list[1], 3),
        conv2d(filter_list[1], filter_list[0], 1),  # 5层DBL  -> outbranch
        conv2d(filter_list[0], filter_list[1], 3),  # 1层DBL
        nn.Conv2d(filter_list[1], out_filter, kernel_size=1, stride=1, padding=0, bias=True)  # 最后的卷积
    ])
    return m


# yolo主体
class YoloBody(nn.Module):
    def __init__(self, config):
        super(YoloBody, self).__init__()
        self.config = config

        self.backbone = darknet53(None)

        out_filters = self.backbone.layers_out_filters

        # 3 * (4 + 1 + 20) = 75
        final_out_filter0 = len(config["yolo"]["anchors"][0]) * (5 + config["yolo"]["classes"])
        # 13 * 13 * 1024 -> 13 * 13 * 75
        self.last_layer0 = make_last_layers([512, 1024], out_filters[-1], final_out_filter0)  # 第一层输出

        # 3 * (4 + 1 + 20) = 75
        final_out_filter1 = len(config["yolo"]["anchors"][1]) * (5 + config["yolo"]["classes"])
        # input=512 -> out=256
        self.last_layer1_conv = conv2d(512, 256, 1)
        # 13 * 13 -> 26 * 26
        self.last_layer1_upsample = nn.Upsample(scale_factor=2, mode="nearest")
        # 26 * 26 * (512+256=768) -> 26 * 26 * 256 -> 26 * 26 * 75
        self.last_layer1 = make_last_layers([256, 512], out_filters[-2] + 256, final_out_filter1)  # 第二层输出

        # 3 * (4 + 1 + 20) = 75
        final_out_filter2 = len(config["yolo"]["anchors"][2] * (5 + config["yolo"]["classes"]))
        # input=256 -> out=128
        self.last_layer2_conv = conv2d(256, 128, 1)
        # 26 * 26 -> 52 * 52
        self.last_layer2_upsample = nn.Upsample(scale_factor=2, mode="nearest")
        # 52 * 52 * (256+128=384) -> 52 * 52 * 128 -> 52 * 52 * 75
        self.last_layer2 = make_last_layers([128, 256], out_filters[-3] + 128, final_out_filter2)  # 第三层输出

    # 把make_last_layers()的七层卷积分为5层+2层，前5层用于下一次的输出
    def forward(self, x):
        def _branch(last_layer, layer_in):
            for i, e in enumerate(last_layer):
                layer_in = e(layer_in)
                if i == 4:
                    out_branch = layer_in
            return layer_in, out_branch

        x2, x1, x0 = self.backbone(x)

        # yolo branch0

        out0, out0_branch = _branch(self.last_layer0, x0)
        #

        # yolo branch1
        x1_in = self.last_layer1_conv(out0_branch)
        x1_in = self.last_layer1_upsample(x1_in)
        x1_in = torch.cat([x1_in, x1], 1)
        out1, out1_branch = _branch(self.last_layer1, x1_in)

        # yolo branch2
        x2_in = self.last_layer2_conv(out1_branch)
        x2_in = self.last_layer2_upsample(x2_in)
        x2_in = torch.cat([x2_in, x2], 1)
        out2, _ = _branch(self.last_layer2, x2_in)

        return out0, out1, out2


