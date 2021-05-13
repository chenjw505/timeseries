# -*- coding:utf-8 -*-
"""

@author: chenjw
@time: 2021/4/29 09:15
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class Constant:
    FORECAST_TYPE_ONE2ONE = 0
    FORECAST_TYPE_MANY2ONE = 1
    FORECAST_TYPE_MANY2MANY = 2

    NORMAL_TYPE_NONE = 0
    NORMAL_TYPE_MATRIX = 1

    MODEL_NAME_LSTNET = 'LSTNet'
    MODEL_NAME_INFORMER = 'Informer'

    MODEL_METRIC_RSE = 'RSE'


class StandardScaler:
    """针对numpy.ndarray数据进行归一化"""
    def __init__(self):
        self.mean = 0
        self.std = 1

    def fit(self, data):
        self.mean = np.mean(data, axis=0)
        self.std = np.mean(data, axis=0)

    def transform(self, data):
        return (data - self.mean)/self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean
