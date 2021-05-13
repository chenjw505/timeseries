# -*- coding:utf-8 -*-
"""

@author: chenjw
@time: 2021/5/12 10:22
"""
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from src.utils.entities import StandardScaler
from src.utils.time_features import build_time_feat


class ETTHour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv', target='OT',
                 scale=True, inverse=False, time_encode=0, freq='h', cols=None):
        super(ETTHour, self).__init__()

        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        type_map = {'train': 0, 'val': 1, 'test': 2}

        self.set_type = type_map[flag]
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.time_encode = time_encode
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self._read_data()

    def _read_data(self):
        self.scaler = StandardScaler()

        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # 2个数组对应元素形成一个pair，总共3个pair
        # 分别表示train，valid，test的数据范围长度
        # train: 0 - 8640,  12个月数据
        # valid: 8544 - 11520, 4个月
        # test: 11424 - 14400, 4个月
        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]

        # 12*30*24 = 24小时，30天，12个月， 即1年的长度
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # 确定数据集
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        # 确定归一化
        if self.scale:
            train_data = df_data.iloc[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)  # 归一化后数据
        else:
            data = df_data.values

        # 时间戳特征
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        self.data_stamp = build_time_feat(df_stamp, time_encode=self.time_encode, freq=self.freq)

        self.data_x = data[border1:border2]
        # y值是否反归一化
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len  # start token length of Informer decoder
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
