#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : elfin
# @Time     : 2021/5/13 10:16
# @File     : update_learning_rate.py
# @Project  : WarmUpLR


def update_learning_rate(opt, warm_lr_scheduler, epoch, step=0):
    """
    更新优化器的学习率配置
    :param warm_lr_scheduler: 学习率迭代器
    :param opt: 优化器 optimizers
    :param epoch: 训练的批次
    :param step: 当前epoch的迭代步数(第多少个batch)
    :return: 更新后的学习率
    """
    lr = warm_lr_scheduler.get_learning_rate(epoch, step)
    for group in opt.param_groups:
        group["lr"] = lr
