# -*- coding:utf-8 -*-
"""

@author: chenjw
@time: 2021/4/29 09:22
"""
import os
import pickle
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from src import cst


class PytorchDataset(Dataset):
    def __init__(self, x_data, y_data):
        super(PytorchDataset, self).__init__()
        self.x_data = x_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.x_data[index, :, :], self.y_data[index, :]

    def __len__(self):
        if self.x_data is None:
            return 0
        return self.x_data.shape[0]


class DatasetDeepAR:
    """
    DeepAR模型使用数据集
    1. 加载数据
    2. 数据集需要有时间信息，需要利用时间信息构建衍生变量
    """

    def __init__(self,
                 file_name,
                 p_train,
                 p_test,
                 start_date,
                 end_date,
                 seq_length,
                 stride_size,
                 horizon,
                 normal_type,
                 num_covariates
                 ):
        self.file_name = os.path.join(file_name)
        self.p_train = p_train
        self.p_test = p_test
        self.start_date = start_date
        self.end_date = end_date
        self.seq_length = seq_length
        self.stride_size = stride_size  # 窗口滑动步长, 一般是按1滑动
        self.horizon = horizon
        self.normal_type = normal_type
        self.num_covariates = num_covariates

    def _load_text(self):
        # 1. 数据加载
        print(f'==> load raw data: {self.file_name}')
        raw_data = np.loadtxt(self.file_name, delimiter=',')
        n, m = raw_data.shape
        data_start = (raw_data != 0).argmax(axis=0)  # 每一列不为0的起始位置

        scale_data = np.zeros((n, m))  # 归一化后的值，用 max normal
        scale_val = np.ones((1, m))  # 每一列归一化所用的max val， 用于反归一化

        # 2. 是否进行归一化处理
        print(f'==> normalize: {self.normal_type}')
        if self.normal_type == cst.NORMAL_TYPE_MATRIX:
            # 按列进行归一化
            for i in range(m):
                scale_val[0, i] = np.max(np.abs(raw_data[:, i]))
                scale_data[:, i] = raw_data[:, i] / scale_val[0, i]
        else:
            scale_data = raw_data

        self.scale_data = scale_data
        self.scale_val = scale_val
        self.n = n  # 序列长度
        self.m = m  # 序列个数
        self.raw_data = raw_data

        # 3. 构建衍生变量，当前日期的小时，周，月份，然后归一化
        ts = pd.date_range(self.start_date, self.end_date, freq='H', closed='left')
        self.covariates = self._gen_covariates(ts)

        # 4. 划分数据集, 从p_train中划分和p_test同样长度的数据作为valid, 调参用
        train_idx = int(n * self.p_train)

        # train需要保证采样一个window数据[seq_length + horizon]
        train_range = range(self.seq_length + self.horizon - 1, train_idx)
        valid_range = range(train_idx, n)

    def _generate_window_data(self, range_idx):
        """
        构建窗口数据, [data_range, seq_length+horizon, m] -> [data_range, m]
        对于每一个时间点，都采样一个window的数据，因此是 data_rang x window_size
        基于多变量预测，每一个时间点的x都是m维，y也是m维
        Args:
            range_idx:
        Returns:

        """
        n = len(range_idx)

        win_x = np.zeros((n, self.seq_length, self.m))
        win_y = np.zeros((n, self.m))

        # horizon表示预测间隔，等于1表示预测next one，idx=5 horizon=1， seq_length=3
        # end: 5 - 1 + 1 = 5
        # start： 5 - 3 = 2，  [2,5), 即 2，3，4 预测 5
        for i in tqdm(range(n)):
            end = range_idx[i] - self.horizon + 1
            start = end - self.seq_length
            win_x[i, :, :] = self.scale_data[start: end, :]
            win_y[i, :] = self.scale_data[range_idx[i], :]
        return [win_x, win_y]

    def _gen_covariates(self, times):
        """构建时间相关协变量特征，并进行归一化"""
        covariates = np.zeros((times.shape[0], self.num_covariates))
        # 列从1开始，0位置用来保存序列实际数据
        for i, input_time in enumerate(times):
            covariates[i, 1] = input_time.weekday()
            covariates[i, 2] = input_time.hour
            covariates[i, 3] = input_time.month
        for i in range(1, self.num_covariates):
            covariates[:, i] = stats.zscore(covariates[:, i])
        return covariates[:, :self.num_covariates]


class DatasetLSTNet:
    """
    LSTNet所使用数据集
    many2many dataset, 多维x预测多维y，即multivariable时序一次
    """

    def __init__(self,
                 data_dir,
                 file_name,
                 p_train,
                 p_valid,
                 p_test,
                 seq_length,
                 horizon=1,
                 normal_type=cst.NORMAL_TYPE_NONE):
        """

        Args:
            file_name:  数据集地址
            p_train: 训练集长度比例
            p_test: 测试集长度比例，
            seq_length: 模型训练所需序列长度
            horizon: 预测间隔，即预测N+horizon
            normal_type: 归一化类型
        """
        self.data_dir = data_dir
        self.data_path = os.path.join(data_dir, file_name)
        self.dataset_name = file_name.split('.')[0]
        self.p_train = p_train
        self.p_valid = p_valid
        self.p_test = p_test
        self.seq_length = seq_length
        self.horizon = horizon
        self.normal_type = normal_type
        self._load_text()

    def _load_text(self):

        # 1. 数据加载
        print(f'==> load raw data: {self.data_path}')
        raw_data = np.loadtxt(self.data_path, delimiter=',')
        n, m = raw_data.shape
        scale_data = np.zeros((n, m))  # 归一化后的值，用 max normal
        scale_val = np.ones((1, m))  # 每一列归一化所用的max val， 用于反归一化

        # 2. 是否进行归一化处理
        print(f'==> normalize: {self.normal_type}')
        if self.normal_type == cst.NORMAL_TYPE_MATRIX:
            # 按列进行归一化
            for i in range(m):
                scale_val[0, i] = np.max(np.abs(raw_data[:, i]))
                scale_data[:, i] = raw_data[:, i] / scale_val[0, i]
        else:
            scale_data = raw_data

        self.scale_data = scale_data
        self.scale_val = scale_val
        self.n = n  # 序列长度
        self.m = m  # 序列个数

        # 3. 划分数据集, 从p_train中划分和p_test同样长度的数据作为valid, 调参用
        train_idx = int(n * self.p_train)
        valid_idx = int(n * (self.p_train + self.p_valid))

        # train需要保证采样一个window数据[seq_length + horizon]
        train_range = range(self.seq_length + self.horizon - 1, train_idx)
        valid_range = range(train_idx, valid_idx)
        test_range = range(valid_idx, self.n)

        # 4. 构建窗口数据
        print('==> generate train data')
        x_train, y_train = self._generate_window_data(train_range)
        print('==> generate valid data')
        x_valid, y_valid = self._generate_window_data(valid_range)
        print('==> generate test data')
        x_test, y_test = self._generate_window_data(valid_range)

        print(f'==> train data: x: {x_train.shape}, y: {y_train.shape}')
        print(f'==> valid data: x: {x_valid.shape}, y: {y_valid.shape}')
        print(f'==> test  data: x: {x_test.shape}, y: {y_test.shape}')
        self.train_dataset = PytorchDataset(x_train, y_train)
        self.valid_dataset = PytorchDataset(x_valid, y_valid)
        self.test_dataset = PytorchDataset(x_test, y_test)

    def _generate_window_data(self, range_idx):
        """
        构建窗口数据, [data_range, seq_length+horizon, m] -> [data_range, m]
        对于每一个时间点，都采样一个window的数据，因此是 data_rang x window_size
        基于多变量预测，每一个时间点的x都是m维，y也是m维
        Args:
            range_idx:
        Returns:

        """
        n = len(range_idx)

        win_x = np.zeros((n, self.seq_length, self.m))
        win_y = np.zeros((n, self.m))

        # horizon表示预测间隔，等于1表示预测next one，idx=5 horizon=1， seq_length=3
        # end: 5 - 1 + 1 = 5
        # start： 5 - 3 = 2，  [2,5), 即 2，3，4 预测 5
        for i in tqdm(range(n)):
            end = range_idx[i] - self.horizon + 1
            start = end - self.seq_length
            win_x[i, :, :] = self.scale_data[start: end, :]
            win_y[i, :] = self.scale_data[range_idx[i], :]
        return win_x, win_y

    def build_test_data(self, test_time):
        """
        构建在线推理数据
        """
        # traffic 数据集时间索引
        start_date = pd.to_datetime('2015-01-01')
        end_date = pd.to_datetime('2017-01-01')
        ts = pd.date_range(start_date, end_date, freq='H', closed='left')
        ts = list(ts)

        test_time = pd.to_datetime(test_time)
        test_index = ts.index(test_time)

        win_x = np.zeros((1, self.seq_length, self.m))
        win_y = np.zeros((1, self.m))

        end = test_index - self.horizon + 1
        start = end - self.seq_length
        win_x[0, :, :] = self.scale_data[start: end, :]

        if test_index < len(self.scale_data):
            win_y[0, :] = self.scale_data[test_index, :]
        else:
            win_y[0, :] = 0  # 在线情况下是没有y值的
        return win_x, win_y

#
# if __name__ == '__main__':
#     data_file = '../../benchmarks/traffic.txt'
#     p_train = 0.8
#     p_test = 0.2
#     seq_length = 10
#     horizon = 1
#     normal_type = cst.NORMAL_TYPE_MATRIX
#     data = DatasetM2M(file_name=data_file,
#                       p_train=p_train,
#                       p_test=p_test,
#                       seq_length=seq_length,
#                       horizon=horizon,
#                       normal_type=normal_type)
#
#     train_loader = DataLoader(dataset=data.train_dataset,
#                               batch_size=128,
#                               shuffle=True)
#     valid_loader = DataLoader(dataset=data.valid_dataset,
#                               batch_size=64,
#                               shuffle=True)
#     test_loader = DataLoader(dataset=data.test_dataset,
#                              batch_size=64,
#                              shuffle=True)
#
#     for x, y in train_loader:
#         print('train:', x.size(), y.size())
#         break
#
#     for x, y in valid_loader:
#         print('valid:', x.size(), y.size())
#         break
#
#     for x, y in test_loader:
#         print('test:', x.size(), y.size())
#         break
