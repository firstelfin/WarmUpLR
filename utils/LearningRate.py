#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : elfin
# @Time     : 2021/5/12 17:34
# @File     : LearningRate.py
# @Project  : WarmUpLR

import math


class CosineAnnealingWarmRestarts(object):
    """
    带重启的余弦退火模型

    Attribute::
        t0: 第一次发生学习率重置的epoch;
        ti: 学习率之恩因子;
        eta_max: 重启的学习率大小;
        steps: 当前训练的每个epoch的step数量
    """

    def __init__(self, eta_max, t0, steps,
                 eta_min=1e-5, ti: int = 1):
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.steps = steps
        self.step = 0
        self.cycle_num = 0
        self.t0 = t0
        self.ti = ti
        assert self.ti >= 1
        self.cycle = self.t0
        pass

    def epoch_step_modify(self, epoch):
        """Iteration reset when loading pre training model."""
        delta_epoch = self.get_used_epochs_in_cycle(epoch-1)
        if delta_epoch > 0 and self.step == 0:
            self.step = delta_epoch * self.steps
        pass

    def get_used_epochs_in_cycle(self, epoch):
        """Gets the number of epochs used in the current cycle."""
        return epoch - self.get_start_epoch_in_cycle(epoch)

    def get_start_epoch_in_cycle(self, epoch):
        """Gets the start_epoch for the current cycle."""
        status = 1
        start_epoch = 1
        while status:
            if self.ti > 1:
                end_epoch = self.t0 * (1 - self.ti**status) / (1 - self.ti)
            else:
                end_epoch = status * self.t0
            if start_epoch <= epoch <= end_epoch:
                status = 0
            else:
                start_epoch = end_epoch + 1
                status += 1
        return start_epoch

    def get_ti(self, epoch):
        """Gets the total number of iterations for the current cycle."""
        if self.ti == 1:
            return self.t0
        else:
            if epoch > self.get_multiply(self.cycle_num) * self.t0:
                self.cycle *= self.ti
                self.cycle_num += 1
        return self.cycle

    def get_multiply(self, n):
        """Times of self.t0 after n warm restart."""
        if n == 0:
            return 1
        return self.ti ** n + self.get_multiply(n - 1)

    def get_learning_rate(self, epoch: [int], step=0):
        """
        Update for lr.
        :param epoch: Current training cycle.
        :param step: Which iteration of the current cycle.
        :return: lr.
        """
        cosine_cycle = self.get_ti(epoch) * self.steps

        lr = self.eta_min + 0.5 * (self.eta_max - self.eta_min) * \
             (1 + math.cos(self.step / cosine_cycle * math.pi))
        self.step += 1
        if self.step == cosine_cycle:
            # warm restart lr
            self.step = 0
        return lr


class WarmUpLR(object):
    """
    Warm up learning rate.
    """

    def __init__(self, origin_lr, warm_epochs=2):
        self.warm_up_epoch = warm_epochs
        self.origin_lr = origin_lr
        self.warm_step = 0

    def get_learning_rate(self, epoch, step):
        if epoch < self.warm_up_epoch:
            return self.get_warm_lr()
        return self.origin_lr.get_learning_rate(epoch - 1, step)

    def get_warm_lr(self):
        """Warm up the first two epochs by default"""
        new_lr = self.origin_lr.eta_min + \
                 (self.origin_lr.eta_max - self.origin_lr.eta_min) * \
                 self.warm_step / self.origin_lr.steps * 0.5
        self.warm_step += 1
        return new_lr


if __name__ == '__main__':
    param = {
        "eta_max": 0.01,
        "eta_min": 1e-5,
        "to": 6,
        "ti": 2,
        "steps": 40,
    }
    cosine = CosineAnnealingWarmRestarts(**param)
