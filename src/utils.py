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

    MODEL_METRIC_RSE = 'RSE'



