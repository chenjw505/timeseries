# -*- coding:utf-8 -*-
"""

@author: chenjw
@time: 2021/5/13 15:54
"""
import torch
import torch.nn as nn
import numpy as np


def gaussian_sample(mu, sigma):
    gaussian = torch.distributions.normal.Normal(mu, sigma)
    ypred = gaussian.sample(mu.size())
    return ypred


class Gaussian(nn.Module):
    def __init__(self,
                 input_size,
                 output_size):
        """

        Args:
            input_size:
            output_size: embedding size
        """
        super(Gaussian, self).__init__()
        self.mu_layer = nn.Linear(input_size, output_size)
        self.sigma_layer = nn.Linear(input_size, output_size)

    def forward(self, h):
        _, hidden_size = h.size()
        sigma_t = torch.log(1 + torch.exp(self.sigma_layer(h))) + 1e-6
        sigma_t = sigma_t.squeeze(0)
        mu_t = self.mu_layer(h).squeeze(0)
        return mu_t, sigma_t


class DeepAR(nn.Module):
    def __init__(self,
                 num_class,
                 embedding_size,
                 input_size,
                 lstm_hidden_size,
                 lstm_num_layers):
        super(DeepAR, self).__init__()
        self.input_size = input_size
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers

        # id做embedding，直接拿Linear
        self.embedding = nn.Embedding(num_class, embedding_size)

        # lstm
        self.lstm = nn.LSTM(embedding_size + input_size-1,  # -1是idx，已经用embdding代替
                            lstm_hidden_size,
                            lstm_num_layers)
        self.relu = nn.ReLU()

        # gaussian
        self.likelihood_layer = Gaussian(input_size=lstm_hidden_size * lstm_num_layers,
                                         output_size=1)

    def forward(self, x, idx, hidden, cell):
        """

        """
        onehot_embed = self.embedding(idx)
        lstm_input = torch.cat((x, onehot_embed), dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        hidden_permute = hidden.permute(1, 2, 0).contiguous().view(hidden.shape[1], -1)
        mu, sigma = self.likelihood_layer(hidden_permute)
        return torch.squeeze(mu), torch.squeeze(sigma), hidden, cell

    def init_hidden(self, batch_size):
        return torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size)
