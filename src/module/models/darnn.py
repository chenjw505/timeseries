# -*- coding:utf-8 -*-
"""

@author: chenjw
@time: 2021/5/13 15:54
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, seq_length, input_size, encoder_num_hidden):
        """

        Args:
            seq_length: 窗口步长，序列长度为 seq_length-1, 预测 next one，
            input_size: 衍生变量size
            encoder_num_hidden:
        """
        super(Encoder, self).__init__()
        # param
        self.encoder_num_hidden = encoder_num_hidden
        self.seq_length = seq_length
        self.input_size = input_size

        # input attn
        # Eq.8
        # input attn不是seq2seq结构，就是一个非线性变化得到权重，只是在temporal attn中每一个时刻都要计算当前时刻的input attn
        # TODO: 这里是线性，论文是非线性
        self.encoder_input_attn = nn.Linear(
            in_features=2 * self.encoder_num_hidden + self.seq_length - 1,
            out_features=1
        )

        # temporal attn
        self.encoder_temporal_attn = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.encoder_num_hidden,
            num_layers=1
        )

    def forward(self, X):
        """
        Args:
            X: 64 x 9 x 81
        """
        # input attn output, (batch, seq, hidden_size), as temporal attn input
        X_tilde = torch.zeros(X.size(0), self.seq_length - 1, self.input_size, dtype=torch.float).to(X.device)

        # temporal attn output
        X_encoded = torch.zeros(X.size(0), self.seq_length - 1, self.encoder_num_hidden, dtype=torch.float).to(X.device)

        # init h_n, s_n of Eq.8
        h_n = self._init_states(X)
        s_n = self._init_states(X)

        # encoder process
        for t in range(self.seq_length - 1):
            # input attn的encoder结果就是X，一个batch里面所有seq所有衍生变量input_size的数据
            # input attn的decoder结合在temporal attn的encoder步骤里面，每次LSTM的输出h和s作为input attn的decoder输入
            # decoder：计算socre，softmax计算weights

            # input attn decoder
            # 这里h_n只有1个step, (1, batch, dim), 复制input_size个，表示衍生变量都用同一个h_n进行计算
            # x=(batch, input_size, seq)
            x = torch.cat((h_n.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           s_n.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           X.permute(0, 2, 1)), dim=2)

            # Eq.8
            x = self.encoder_input_attn(
                x.view(-1, self.encoder_num_hidden * 2 + self.seq_length - 1)
            )

            # get weights, (batch, input_size)
            alpha = F.softmax(x.view(-1, self.input_size), dim=1)

            # attn: (batch, input_size) * (batch, input_size)
            # input attn只有权重系数，没有进行加权求和处理
            x_tilde = torch.mul(alpha, X[:, t, :])

            # temporal attn encoder
            _, final_state = self.encoder_temporal_attn(x_tilde.unsqueeze(0), (h_n, s_n))
            h_n = final_state[0]
            s_n = final_state[1]

            X_tilde[:, t, :] = x_tilde
            X_encoded[:, t, :] = h_n

        return X_tilde, X_encoded

    def _init_states(self, X) -> torch.Tensor:
        """
        init h
        """
        return torch.zeros(1, X.size(0), self.encoder_num_hidden, dtype=torch.float).to(X.device)


class Decoder(nn.Module):
    def __init__(self, seq_length, encoder_num_hidden, decoder_num_hidden):
        super(Decoder, self).__init__()

        self.encoder_num_hidden = encoder_num_hidden
        self.decoder_num_hidden = decoder_num_hidden
        self.seq_lenght = seq_length

        # 这里是decoder的lstm和encoder的lstm计算score
        self.decoder_score_layer = nn.Sequential(
            nn.Linear(2 * decoder_num_hidden + encoder_num_hidden, encoder_num_hidden),
            nn.Tanh(),
            nn.Linear(encoder_num_hidden, 1)
        )

        # 计算隐层特征
        self.decoder_lstm_layer = nn.LSTM(
            input_size=1,
            hidden_size=decoder_num_hidden
        )

        self.fc = nn.Linear(encoder_num_hidden + 1, 1)
        self.fc_final = nn.Linear(decoder_num_hidden + encoder_num_hidden, 1)

    def forward(self, X_encoded, y_prev):
        """

        Args:
            X_encoded:  [64, 9, 128]
            y_prev: [64, 9]

        Returns:

        """
        # shape和输入的X相关
        d_n = self._init_states(X_encoded)
        c_n = self._init_states(X_encoded)

        # decode process
        for t in range(self.seq_lenght - 1):
            # x = (batch, seq, dim)， X_encoded是已经经过LSTM得到的 encoder_output
            # 这里把d和s都复制到seq长度一起进行计算，这里d和s的每一个seq都是一样的，计算出来的系数就是当前step的系数
            # 这里直接计算当前step下的d和s，和encoder所有step的score
            x = torch.cat((d_n.repeat(self.seq_lenght - 1, 1, 1).permute(1, 0, 2),
                           c_n.repeat(self.seq_lenght - 1, 1, 1).permute(1, 0, 2),
                           X_encoded), dim=2)

            # 计算weight
            beta = F.softmax(self.decoder_score_layer(
                x.view(-1, 2 * self.decoder_num_hidden + self.encoder_num_hidden)).view(-1, self.seq_lenght - 1), dim=1)

            # 计算当前step的context vector
            context = torch.bmm(beta.unsqueeze(1), X_encoded)[:, 0, :]

            if t < self.seq_lenght - 1:
                # Eq.15
                y_tilde = self.fc(
                    torch.cat((context, y_prev[:, t].unsqueeze(1)), dim=1)
                )

                # Eq.16
                _, final_states = self.decoder_lstm_layer(y_tilde.unsqueeze(0), (d_n, c_n))

                d_n = final_states[0]
                c_n = final_states[1]

        # Eq.22
        y_pred = self.fc_final(torch.cat((d_n[0], context), dim=1))
        return y_pred

    def _init_states(self, X):
        # hidden state and cell state [num_layers*num_directions, batch_size, hidden_size]
        # https://pytorch.org/docs/master/nn.html?#lstm
        return torch.zeros(1, X.size(0), self.decoder_num_hidden, dtype=torch.float).to(X.device)


class DA_RNN(nn.Module):
    def __init__(self, X, y, seq_length,
                 encoder_num_hidden,
                 decoder_num_hidden,
                 batch_size,
                 learning_rate,
                 epochs):
        """

        Args:
            X:
            y:
            seq_length:
            encoder_num_hidden:
            decoder_num_hidden:
            batch_size:
            learning_rate:
            epochs:
        """
        super(DA_RNN, self).__init__()
        self.encoder_num_hidden = encoder_num_hidden  # 128
        self.decoder_num_hidden = decoder_num_hidden  # 128
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.shuffle = False
        self.epochs = epochs
        self.seq_length = seq_length
        self.X = X
        self.y = y

        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu'
        )
        print('==> Use accelerator: ', self.device)

        self.Encoder = Encoder(input_size=X.shape[1],
                               encoder_num_hidden=encoder_num_hidden,
                               seq_length=seq_length).to(self.device)

        self.Decoder = Decoder(encoder_num_hidden=encoder_num_hidden,
                               decoder_num_hidden=decoder_num_hidden,
                               seq_length=seq_length).to(self.device)

        # Loss
        self.criterion = nn.MSELoss()

        # opt
        self.encoder_optimizer = optim.Adam(self.Encoder.parameters(), lr=self.learning_rate)
        self.decoder_optimizer = optim.Adam(self.Decoder.parameters(), lr=self.learning_rate)

        # Training set
        self.train_timesteps = int(self.X.shape[0] * 0.7)
        self.y = self.y - np.mean(self.y[: self.train_timesteps])
        self.input_size = self.X.shape[1]

    def train_forward(self, X, y_prev, y_gt):
        """
        DA_RNN训练过程
        Args:
            X: 衍生变量
            y_prev: 序列变量
            y_gt: 真实值，即序列变量的下一个时刻的label值

        """
        # zero gradients
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        # encoder
        X_weighted, X_encoded = self.Encoder(
            torch.tensor(X, dtype=torch.float).to(self.device)
        )

        # decoder
        y_pred = self.Decoder(
            X_encoded,
            torch.tensor(y_prev, dtype=torch.float).to(self.device)
        )

        # label
        y_true = torch.tensor(y_gt, dtype=torch.float).to(self.device)
        y_true = y_true.view(-1, 1)

        loss = self.criterion(y_pred, y_true)
        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.item()

    def train(self):
        """Model training process"""
        iter_per_epoch = int(np.ceil(self.train_timesteps + 1. / self.batch_size))

        self.iter_losses = np.zeros(self.epochs * iter_per_epoch)
        self.epoch_losses = np.zeros(self.epochs)

        n_iter = 0
        for epoch in range(self.epochs):
            if self.shuffle:
                ref_idx = np.random.permutation(self.train_timesteps - self.seq_length)
            else:
                ref_idx = np.array(range(self.train_timesteps - self.seq_length))

            idx = 0

            # 循环提取windown数据，
            while idx < self.train_timesteps:
                # nasdaq是模拟多个衍生变量+序列变量来预测单个序列变量，如果是多变量这里修改即可
                indices = ref_idx[idx: idx + self.batch_size]  # 一个batch大小
                x = np.zeros((len(indices), self.seq_length - 1, self.input_size))  # 一个batch大小的seq数据, 衍生变量
                y_prev = np.zeros((len(indices), self.seq_length - 1))  # 序列数据
                y_gt = self.y[indices + self.seq_length]  # 往后1一个seq的label，即每个seq时间步对应的label

                # 构建数据成3D tensor
                for bs in range(len(indices)):
                    x[bs, :, :] = self.X[indices[bs]: indices[bs] + self.seq_length - 1]
                    y_prev[bs, :] = self.y[indices[bs]: indices[bs] + self.seq_length - 1]

                # 构建好一个batch数据就进行训练，
                loss = self.train_forward(x, y_prev, y_gt)

                self.iter_losses[int(epoch * iter_per_epoch + idx / self.batch_size)] = loss

                idx += self.batch_size
                n_iter += 1

                if n_iter % 10000 == 0 and n_iter != 0:
                    for param_group in self.encoder_optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.9
                    for param_group in self.decoder_optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.9

                self.epoch_losses[epoch] = np.mean(self.iter_losses[range(
                    epoch * iter_per_epoch, (epoch + 1) * iter_per_epoch)])

            if epoch % 2 == 0:
                print("Epochs: ", epoch, " Iterations: ", n_iter,
                      " Loss: ", self.epoch_losses[epoch])

            if epoch % 10 == 0:
                y_train_pred = self.test(on_train=True)
                y_test_pred = self.test(on_train=False)
                plt.ioff()
                plt.figure()
                plt.plot(range(1, 1 + len(self.y)), self.y, label="True")
                plt.plot(range(self.seq_length, len(y_train_pred) + self.seq_length),
                         y_train_pred, label='Predicted - Train')
                plt.plot(range(self.seq_length + len(y_train_pred), len(self.y) + 1),
                         y_test_pred, label='Predicted - Test')
                plt.legend(loc='upper left')
                plt.show()

    def test(self, on_train=False):
        """Prediction"""
        if on_train:
            y_pred = np.zeros(self.train_timesteps - self.seq_length + 1)
        else:
            y_pred = np.zeros(self.X.shape[0] - self.train_timesteps)

        i = 0
        while i < len(y_pred):
            batch_idx = np.array(range(len(y_pred)))[i: i + self.batch_size]
            X = np.zeros((len(batch_idx), self.seq_length - 1, self.X.shape[1]))
            y_history = np.zeros((len(batch_idx), self.seq_length - 1))

            for j in range(len(batch_idx)):
                if on_train:
                    X[j, :, :] = self.X[range(batch_idx[j], batch_idx[j] + self.seq_length - 1), :]
                    y_history[j, :] = self.y[range(batch_idx[j], batch_idx[j] + self.seq_length - 1)]

                else:
                    X[j, :, :] = self.X[range(
                        batch_idx[j] + self.train_timesteps - self.seq_length,
                        batch_idx[j] + self.train_timesteps - 1),
                                 :]
                    y_history[j, :] = self.y[range(
                        batch_idx[j] + self.train_timesteps - self.seq_length,
                        batch_idx[j] + self.train_timesteps - 1)]

            X = torch.tensor(X, dtype=torch.float).to(self.device)
            y_history = torch.tensor(y_history, dtype=torch.float).to(self.device)

            _, X_encoded = self.Encoder(X)
            y_pred[i: i + self.batch_size] = self.Decoder(X_encoded, y_history).cpu().detach().numpy()[:, 0]
            i += self.batch_size
        return y_pred
