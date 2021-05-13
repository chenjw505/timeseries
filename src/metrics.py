# -*- coding:utf-8 -*-
"""
业务指标
@author: chenjw
@time: 2021/5/6 10:00
"""
import numpy as np


def rse(y_true: np.ndarray, y_pred: np.ndarray):
    """
    RSE：可以衡量不同单位数据下模型的效果，分母有减去均值消除偏差
    RSE = sqrt(sum[(y_true - y_pred)^2]) / sqrt(sum[(y_true - y_mean)^2])
    y_mean = np.mean(y_true)

    注意是针对整个batch计算

    Args:
        y_true (np.ndarray):  shape: (batch, input_size),  （128， 862）
        y_pred (np.ndarray):  shape: (batch, input_size),

    Returns:
        rse: shape: (batch, 1)
    """
    y_mean = y_true.mean(axis=0)  # 每一列的均值，(128, 1)
    fraq_up = np.sqrt(np.sum(np.power((y_true - y_pred), 2)))
    fraq_down = np.sqrt(np.sum(np.power((y_true - y_mean), 2)))
    return fraq_up / fraq_down
