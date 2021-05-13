# -*- coding:utf-8 -*-
"""

@author: chenjw
@time: 2021/3/4 10:14
"""
import numpy as np
from typing import Iterator
import gin
import os
import torch
import torch.nn as nn
from torch import optim
from model import NBeats, NBeatsBlock, GenericBasis, TrendBasis, SeasonalityBasis
from utils import divide_no_nan
from sampler import TimeseriesSampler


def smape2loss(forecast, target, mask) -> torch.float:
    """
    sMAPE loss function, https://robjhyndman.com/hyndsight/smape/
    data shape: batch, time
    """
    return 200 * torch.mean(divide_no_nan(torch.abs(forecast - target),
                                          torch.abs(forecast.data) + torch.abs(target.data)) * mask)


def rmse(forecast: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.float:
    """
    RMSE:  因为有mask，损失函数需要自己定义
    Args:
        forecast:  batch x time
        target:
        mask:
    """
    return torch.sqrt(torch.mean(torch.pow((forecast - target) * mask, 2)))


class NBeatNetwork:
    """
    定义以全连接作为Block最后一层的网络结构
    Args:
        input_size:
        output_size:
        stacks:
        layers:
        layer_size:
    """

    def __init__(self,
                 training_values,
                 test_values,
                 input_size,
                 horizon,
                 window_sample_limit,
                 batch_size,
                 learning_rate,
                 iterations,
                 basis_name='generic',
                 test_windows=7,
                 config_path='../configs'
                 ):
        self.training_values = training_values
        self.test_values = test_values
        self.input_size = input_size
        self.horizon = horizon
        self.window_sample_limit = window_sample_limit
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.test_windows = test_windows
        self.basis_name = basis_name

        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu'
        )
        print('Use accelerator: ', self.device)

        print(f'config = [{basis_name}]')
        gin.parse_config_file(os.path.join(config_path, basis_name + '.gin'))

        # data
        self.training_set_sampler = TimeseriesSampler(timeseries=self.training_values,
                                                      insample_size=self.input_size,
                                                      outsample_size=self.horizon,
                                                      window_sample_limit=self.window_sample_limit,
                                                      batch_size=self.batch_size)

        self.test_set_sampler = TimeseriesSampler(timeseries=self.test_values,
                                                  insample_size=self.input_size,
                                                  outsample_size=self.horizon,
                                                  window_sample_limit=self.window_sample_limit,
                                                  batch_size=self.batch_size)

        # model
        if basis_name == 'generic':
            self.model = generic(backcast_size=input_size,
                                 forecast_size=horizon).to(self.device)
        else:
            self.model = interpretable(backcast_size=input_size,
                                       forecast_size=horizon)
        # criterion
        print('criterion = RMSE')
        self.criterion = rmse
        # optimizer
        print(f'optimizer = Adam with lr={learning_rate}')
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def _to_tensor(self, array: np.ndarray) -> torch.Tensor:
        return torch.tensor(array, dtype=torch.float32).to(self.device)

    def train(self):
        # 数据集生成器
        training_set = iter(self.training_set_sampler)

        # 训练过程
        for i in range(self.iterations):
            self.model.train()
            x, x_mask, y, y_mask = map(self._to_tensor, next(training_set))  # 一个next是一个batch

            forecast = self.model(x, x_mask)
            loss = self.criterion(forecast, y, y_mask)

            if np.isnan(float(loss)):
                break

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if i % 10000 == 0:
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.learning_rate * 0.9

            if i % 500 == 0:
                test_loss = self.test()
                print(f'Iter: {i}, Train loss: {loss.item()}, Test loss: {test_loss}')

        print('==> Training done!')

    def test(self):
        test_set = iter(self.test_set_sampler)
        # 打印一个batch的损失
        self.model.eval()
        with torch.no_grad():
            x, x_mask, y, y_mask = map(self._to_tensor, next(test_set))
            forecast = self.model(x, x_mask)
            loss = self.criterion(forecast, y, y_mask)
            return loss.item()

    def predict(self, test_windows):
        """predict"""
        forecasts = []
        self.model.eval()
        with torch.no_grad():
            # 预测的1次是预测所有序列的 horizon 长度，表示24h，表示1天，
            # 这里test_windows=7，表示预测一周的长度，以小时为单位拼接
            # 验证不用验证1周，就验证一次预测24h的结果即可
            for i in range(test_windows):
                window_input_set = np.concatenate([self.training_values,
                                                   self.test_values[:, :i * self.horizon]],
                                                  axis=1)

                test_set = TimeseriesSampler(timeseries=window_input_set,
                                             insample_size=self.input_size,
                                             outsample_size=self.horizon,
                                             window_sample_limit=self.window_sample_limit,
                                             batch_size=self.batch_size)

                # 这里只取最后一个window
                x, x_mask = map(self._to_tensor, test_set.last_insample_window())
                window_forecast = self.model(x, x_mask).cpu().detach().numpy()

                forecasts = window_forecast if len(forecasts) == 0 else np.concatenate([forecasts, window_forecast],
                                                                                       axis=1)
            return forecasts


@gin.configurable()
def generic(backcast_size,
            forecast_size,
            layers,
            layer_size,
            blocks):
    """
    gin是其读取配置文件作为默认参数，如果调用时有传入则修改参数
    Args:
        backcast_size:
        forecast_size:
        ---
        # 以下参数参考gin文件
        layers:
        layer_size:
        blocks:

    Returns:

    """
    # 因为是全连接，只有一种类型的stack
    stack_list = nn.ModuleList([NBeatsBlock(input_size=backcast_size,
                                            theta_size=backcast_size + forecast_size,
                                            basis_function=GenericBasis(backcast_size=backcast_size,
                                                                        forecast_size=forecast_size),
                                            layers=layers,
                                            layer_size=layer_size)
                                for _ in range(blocks)])
    return NBeats(stack_list)


@gin.configurable()
def interpretable(backcast_size: int,
                  forecast_size: int,
                  trend_blocks: int,
                  trend_layers: int,
                  trend_layer_size: int,
                  degree_of_polynomial: int,
                  seasonality_blocks: int,
                  seasonality_layers: int,
                  seasonality_layer_size: int,
                  num_of_harmonics: int):
    """
    可解释模型，一个trend的stack，一个seasonality的stack
    Args:
        backcast_size: 输入维度
        forecast_size: 输出维度
        ---
        # 以下参数参考gin文件
        trend_blocks: 3, trend stack包含多少个block
        trend_layers: 4,
        trend_layer_size: 256
        degree_of_polynomial: 3
        seasonality_blocks: 3
        seasonality_layers: 4
        seasonality_layer_size: 2048
        num_of_harmonics: 1

    Returns:

    """
    # 声明trend类型的block
    trend_block = NBeatsBlock(input_size=backcast_size,
                              theta_size=2 * (degree_of_polynomial + 1),
                              basis_function=TrendBasis(degree_of_polynomial=degree_of_polynomial,
                                                        backcast_size=backcast_size,
                                                        forecast_size=forecast_size),
                              layers=trend_layers,
                              layer_size=trend_layer_size)

    # 声明seasonality类型的block
    seasonality_block = NBeatsBlock(input_size=backcast_size,
                                    theta_size=4 * int(
                                        np.ceil(num_of_harmonics / 2 * forecast_size) - (num_of_harmonics - 1)),
                                    basis_function=SeasonalityBasis(harmonics=num_of_harmonics,
                                                                    backcast_size=backcast_size,
                                                                    forecast_size=forecast_size),
                                    layers=seasonality_layers,
                                    layer_size=seasonality_layer_size
                                    )

    # 一个stack包含多少个block
    trend_stack = [trend_block for _ in range(trend_blocks)]
    seasonality_stack = [seasonality_block for _ in range(seasonality_blocks)]

    # 返回整个网络结构
    return NBeats(torch.nn.ModuleList(trend_stack + seasonality_stack))
