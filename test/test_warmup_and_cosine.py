#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : elfin
# @Time     : 2021/5/12 18:07
# @File     : test_warmup_and_cosine.py
# @Project  : WarmUpLR

import os
import sys
sys.path.append(os.path.abspath(__file__).split("test")[0])

import numpy as np
import matplotlib.pyplot as plt
from torch.optim import SGD
from torchvision.models import VGG, vgg
from utils.LearningRate import CosineAnnealingWarmRestarts
from utils.LearningRate import WarmUpLR
from utils.update_learning_rate import update_learning_rate

model = VGG(vgg, num_classes=2)
optimizer = SGD(model.parameters(), lr=0.9, momentum=0.9)

param = {
    "eta_max": 0.01,
    "eta_min": 1e-5,
    "t0": 6,
    "ti": 2,
    "steps": 500,
}


def show_lr(epoch=0, start_epoch=0, end_epoch=42):
    assert epoch >= start_epoch, f"excepted epoch >= start_epoch, " \
                                 f"got epoch:{epoch}--start_epoch:{start_epoch}"
    cosine = CosineAnnealingWarmRestarts(**param)
    warm_up_lr = WarmUpLR(cosine)
    cosine2 = CosineAnnealingWarmRestarts(**param)

    y1 = []
    y2 = []
    x1 = np.linspace(start_epoch, end_epoch, (end_epoch - start_epoch) * param["steps"])
    for epoch_index in range(start_epoch + 1, end_epoch + 1):
        for step_num in range(param["steps"]):
            y1.append(cosine2.get_learning_rate(epoch_index, step_num))

    plt.plot(x1, y1, color="pink")

    if epoch != 0:
        warm_up_lr.origin_lr.epoch_step_modify(epoch)

    for epoch_index in range(epoch, epoch + end_epoch - start_epoch):
        for step_num in range(param["steps"]):
            update_learning_rate(optimizer, warm_up_lr, epoch_index, step_num)
            y2.append(optimizer.param_groups[0]['lr'])
            optimizer.step()

    x2 = np.linspace(epoch, epoch + end_epoch - start_epoch, (end_epoch - start_epoch) * param["steps"])
    if not epoch:
        x3 = [2 for _ in range(22)]
        y3 = [0.0005*i for i in range(22)]
        plt.plot(x3, y3, color="red", linestyle="--")

    plt.plot(x2, y2, color="green")
    plt.xlabel("epoch")
    plt.ylabel("learning rate")
    plt.title("CosineAnnealingWarmRestarts")
    plt.show()
    pass


if __name__ == '__main__':
    show_lr()
    show_lr(13)
