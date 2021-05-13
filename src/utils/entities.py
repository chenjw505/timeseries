# -*- coding:utf-8 -*-
"""

@author: chenjw
@time: 2021/4/29 09:15
"""
import torch
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


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path+'/'+'checkpoint.pth')
        self.val_loss_min = val_loss


class TriangularCausalMask:
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class ProbMask:
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask