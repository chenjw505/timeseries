# -*- coding:utf-8 -*-
"""
NBeats Model

@author: chenjw
@time: 2021/3/4 10:13

References:
    [1] N-BEATS: Neural basis expansion analysis for interpretable time series forecasting
    [2] https://github.com/ElementAI/N-BEATS
    [3] https://github.com/philipperemy/n-beats/blob/master/nbeats_pytorch/model.py

"""
import numpy as np
from typing import Tuple
import torch
import torch.nn as nn


class NBeats(nn.Module):
    """定义整个NBeats模型结构"""

    def __init__(self,
                 blocks: nn.ModuleList):
        super(NBeats, self).__init__()
        # 整个NBeats虽然有stack和block区分，但其实就是一连串block
        self.blocks = blocks

    def forward(self, x, input_mask):
        residuals = x
        forecast = x[:, -1:]
        for i, block in enumerate(self.blocks):
            backcast, block_forecast = block(residuals)
            residuals = (residuals - backcast) * input_mask  # 模型输入都要长度统一，计算后续得分要去掉mask的影响
            forecast = forecast + block_forecast
        return forecast


class NBeatsBlock(nn.Module):
    """定义一个Block结构"""

    def __init__(self,
                 input_size,
                 theta_size,
                 basis_function: nn.Module,
                 layers,
                 layer_size):
        """

        Args:
            input_size: 即时序窗口长度，和backcast_size一样
            theta_size:
            basis_function:
            layers:
            layer_size:
        """
        super(NBeatsBlock, self).__init__()

        # 4层全连接，第一层维度需要和input对齐
        self.layers = nn.ModuleList([nn.Linear(input_size, layer_size)] +
                                    [nn.Linear(layer_size, layer_size) for _ in range(layers - 1)])

        # 维度转换，方便拆分backcast和forecast
        self.basis_fc = nn.Linear(layer_size, theta_size)

        # backcast/forecast分别对应的网络层
        self.basis_layer = basis_function

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        block_input = x
        for layer in self.layers:
            block_input = torch.relu(layer(block_input))
        basis_input = self.basis_fc(block_input)
        backcast, forecast = self.basis_layer(basis_input)  # 最终的输出是由block最后一层决定的
        return backcast, forecast


# ------------------
# 最后一层只是定义一个操作，将上一步学习到的向量转成想要的格式
# 然后使前面的模型学出来的向量是满足定义的操作的
# 最后一层是没有学习参数的
# ------------------
class GenericBasis(nn.Module):
    """定义block最后一层: 全连接模型，backcast和forecast拼在一个全连接学习参数"""

    def __init__(self, backcast_size, forecast_size):
        super(GenericBasis, self).__init__()
        self.backcast = backcast_size
        self.forecast = forecast_size

    def forward(self, theta):
        # 这里 theta  是block里面basis_fc转换维度得到的向量
        # 直接把前部分作为backcast向量，后部分作为forecast
        # 注意最后一层，
        backcast = theta[:, :self.backcast]
        forecast = theta[:, -self.forecast:]
        return backcast, forecast


class TrendBasis(nn.Module):
    """
    定义block最后一层，trend类型，拟合序列的趋势
    """

    def __init__(self,
                 degree_of_polynomial,
                 backcast_size,
                 forecast_size):
        """
        针对backcast和forecast都要定义一个多项式线性组合来拟合趋势
        degree_of_polynomial是定义了用多少种多项式来拟合，比如3

        trendbasis 和seasonlity输入前的theta的维度和generic维度是不一的，需要根据结构定义
        比如 trendbasis 输入的theta的维度是batch x 2*（degree_poly）， 分别是backcast的 batch x degree_poly 和forecast的

        再来看如何学习趋势：就backcast来说

        theta： batch x degree_poly
        time： degree_poly x backcast_size
        最后得到： batch x backcast_size， 得到每个batch，蕴含 trend 信息的特征 backcast_size

        对于theta，batch x degree_poly， 每一行就是一个样本trend的系数，有3个。
        对于time，degree_poly x backcast_size， 每一列是trend的变量t，一列就是从0，1，2次方的变化，一行就是每个时刻的多项式特征
         列就是多项式，行就是时间步。 backcast_size 就是历史多少步 time的每一行就是时间步不同多项式的组合。

         这样就促使前面的FC学习的theta为不同多项式上时间步的系数，
        Args:
            degree_of_polynomial: 用多项式去拟合序列的趋势
            backcast_size:
            forecast_size:
        """
        super(TrendBasis, self).__init__()
        self.polynomial_size = degree_of_polynomial + 1  # 还有一个偏置项

        # 定义不同多项式下的时间步长
        # polynomial_size x timestep
        # 归一化了一下
        self.backcast_timestep = torch.tensor(
            np.concatenate([np.power(np.arange(backcast_size, dtype=np.float) / backcast_size, i)[None, :]
                            for i in range(self.polynomial_size)]), requires_grad=False)

        self.forecast_timestep = torch.tensor(
            np.concatenate([np.power(np.arange(forecast_size, dtype=np.float) / forecast_size, i)[None, :]
                            for i in range(self.polynomial_size)]), requires_grad=False)

    def forward(self, theta: torch.Tensor):
        backcast = torch.einsum('bp, pt -> bt', theta[:, :self.polynomial_size], self.backcast_timestep)
        forecast = torch.einsum('bp, pt -> bt', theta[:, self.polynomial_size:], self.forecast_timestep)
        return backcast, forecast


class SeasonalityBasis(nn.Module):
    def __init__(self,
                 harmonics,
                 backcast_size,
                 forecast_size):
        """
        构建一个  batch x  harmonics
                harmonics x backcast_size  # 每一列代表一个time步长的多谐波特征

        这里的 harmonics 不是参数指定的，是代码里面的frequency
        不用纠结是否和参数对应，只是一系列谐波的倍数而已

        Args:
            harmonics:
            backcast_size:
            forecast_size:
        """

        super(SeasonalityBasis, self).__init__()

        # 9个谐波，(1 x 9)
        # 谐波倍数从0开始
        # 除以 harmonics 使后面部分从1开始，同时也是归一化
        self.frequency = np.append(np.zeros(1, dtype=np.float32),
                                   np.arange(harmonics, harmonics * forecast_size / 2,
                                             dtype=np.float32) / harmonics)[None, :]

        # 时间步长：(12 x 1) 和 (6 x 1)
        backcast_line = np.arange(backcast_size, dtype=np.float)[:, None] / backcast_size
        forecast_line = np.arange(forecast_size, dtype=np.float)[:, None] / forecast_size

        # 构建参数网络
        backcast_grid = -2 * np.pi * backcast_line * self.frequency
        forecast_grid = -2 * np.pi * forecast_line * self.frequency

        # (9 x 12), (9 x 6)
        # 转置使其成为每列表示1个步长的多谐波组合
        backcast_cos_values = np.transpose(np.cos(backcast_grid))
        backcast_sin_values = np.transpose(np.sin(backcast_grid))
        forecast_cos_values = np.transpose(np.cos(forecast_grid))
        forecast_sin_values = np.transpose(np.sin(forecast_grid))

        # 参数层，用于和theta计算
        self.backcast_cos_template = torch.tensor(backcast_cos_values, dtype=torch.float32, requires_grad=False)
        self.backcast_sin_template = torch.tensor(backcast_sin_values, dtype=torch.float32, requires_grad=False)
        self.forecast_cos_template = torch.tensor(forecast_cos_values, dtype=torch.float32, requires_grad=False)
        self.forecast_sin_template = torch.tensor(forecast_sin_values, dtype=torch.float32, requires_grad=False)

    def forward(self, theta: torch.Tensor):
        """
        theta的维度： 4 * int(np.ceil(harmonical*forecast/2 - (harmonical+1)
        Args:
            theta: theta前部分为backcast，后部分为forecast
        """
        params_per_harmonic = theta.shape[1] // 4

        backcast_harmonics_cos = torch.einsum('bp,pt->bt',
                                              theta[:, :params_per_harmonic], self.backcast_cos_template)
        backcast_harmonics_sin = torch.einsum('bp,pt->bt', theta[:, params_per_harmonic:2 * params_per_harmonic],
                                              self.backcast_sin_template)
        backcast = backcast_harmonics_sin + backcast_harmonics_cos

        forecast_harmonics_cos = torch.einsum('bp,pt->bt', theta[:, 2 * params_per_harmonic:3 * params_per_harmonic],
                                              self.forecast_cos_template)
        forecast_harmonics_sin = torch.einsum('bp,pt->bt', theta[:, 3 * params_per_harmonic:],
                                              self.forecastcast_sin_template)
        forecast = forecast_harmonics_sin + forecast_harmonics_cos

        return backcast, forecast