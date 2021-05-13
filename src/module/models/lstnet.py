# -*- coding:utf-8 -*-
"""

@author: chenjw
@time: 2021/4/29 20:19
"""
import numpy as np
import torch
import torch.nn as nn


class LstNet(nn.Module):
    def __init__(self,
                 seq_length,
                 input_dim,
                 cnn_kernel,
                 cnn_hidden_dim,
                 rnn_hidden_dim,
                 skip_period,
                 skip_hidden_dim,
                 highway_window,
                 p_dropout):
        """

        Args:
            seq_length: default 24 * 7
            input_dim:
            rnn_hidden_dim: default 100
            cnn_hidden_dim: default 100
            skip_hidden_dim: default 10
            cnn_kernel: default 6
            skip_period: default 24
            highway_window: default 24
            p_dropout: default 0.2
        """
        super(LstNet, self).__init__()
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.cnn_hidden_dim = cnn_hidden_dim
        self.skip_hidden_dim = skip_hidden_dim
        self.cnn_kernel = cnn_kernel
        self.skip_period = skip_period
        self.highway_window = highway_window
        self.p_dropout = p_dropout

        self.pt = int((seq_length - self.cnn_kernel) / skip_period)  # 卷积后的序列提取多少个周期数据

        # CNN: 提取局部信息
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=self.cnn_hidden_dim,
                               kernel_size=(self.cnn_kernel, self.input_dim))

        # GRU：提取序列信息
        self.gru1 = nn.GRU(self.cnn_hidden_dim, self.rnn_hidden_dim)

        self.dropout = nn.Dropout(p=p_dropout)

        # skip rnn
        if self.skip_period > 0:
            self.gruskip = nn.GRU(self.cnn_hidden_dim, self.skip_hidden_dim)
            self.linear_layer = nn.Linear(self.rnn_hidden_dim + self.skip_period * skip_hidden_dim,
                                          self.input_dim)
        else:
            self.linear_layer = nn.Linear(self.rnn_hidden_dim, self.input_dim)

        if self.highway_window > 0:
            self.highway_layer = nn.Linear(self.highway_window, 1)

        self.sigmoid = torch.sigmoid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (tensor): shape (batch, seq_length, input_dim)
        """
        batch_size = x.size(0)

        # cnn: short multi variable term
        c = x.view(batch_size, 1, self.seq_length, self.input_dim)  # [128, 1, 168, 862]
        c = torch.relu(self.conv1(c))
        c = self.dropout(c)
        c = torch.squeeze(c, 3)

        # rnn: long term
        r = c.permute(2, 0, 1).contiguous()
        _, r = self.gru1(r)
        r = torch.squeeze(r, 0)
        r = self.dropout(r)

        # skip-rnn： cycle term， very long term
        if self.skip_period > 0:
            s = c[:, :, -int(self.pt * self.skip_period):].contiguous()
            s = s.view(batch_size, self.cnn_hidden_dim, self.pt, self.skip_period)
            s = s.permute(2, 0, 3, 1).contiguous()
            s = s.view(self.pt, batch_size * self.skip_period, self.cnn_hidden_dim)
            _, s = self.gruskip(s)
            s = s.view(batch_size, self.skip_period * self.skip_hidden_dim)
            s = self.dropout(s)
            r = torch.cat((r, s), 1)  # 增加rnn

        res = self.linear_layer(r)

        # AR: trend
        if self.highway_window > 0:
            z = x[:, -self.highway_window:, :]
            z = z.permute(0, 2, 1).contiguous().view(-1, self.highway_window)
            z = self.highway_layer(z)
            z = z.view(-1, self.input_dim)
            res = res + z

        # output
        res = self.sigmoid(res)

        return res



