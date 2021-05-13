# -*- coding:utf-8 -*-
"""

@author: chenjw
@time: 2021/4/29 10:41
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from src.dataloader import DatasetM2M


class Network:
    def __init__(self, args):
        """
        按顺序声明：
            dataset
            models
            optimizer
            criterion
        Args:
            args:
        """
        self.model_name = args.model_name
        self.optimizer_name = args.optimizer_name
        self.ckp_path = args.ckp_path

        # dataset
        data = DatasetM2M(file_name=args.data_file,
                          p_train=args.p_train,
                          p_test=args.p_test,
                          seq_length=args.seq_length,
                          horizon=args.horizon,
                          normal_type=args.normal_type)

        self.train_loader = DataLoader(dataset=data.train_dataset,
                                       batch_size=args.train_batch_size,
                                       shuffle=True)
        self.valid_loader = DataLoader(dataset=data.valid_dataset,
                                       batch_size=args.test_batch_size,
                                       shuffle=True)
        self.test_loader = DataLoader(dataset=data.test_dataset,
                                      batch_size=args.test_batch_size,
                                      shuffle=True)

        self.model = None
        self.optimizer = optim.SGD()

    def save_checkpoint(self, ckp_name):
        """
        Args:
            ckp_name: ckp_modelname_epoch_loss.pt
        """
        names = ckp_name.split('_')
        epoch = int(names[-2])
        loss = float(names[-1])
        save_path = os.path.join(self.ckp_path, ckp_name)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, save_path)

    def load_checkpoint(self, ckp_name):
        """

        Args:
            ckp_name: ckp_modelname_epoch_loss.pt
        """
        save_path = os.path.join(self.ckp_path, ckp_name)
        checkpoint = torch.load(save_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = int(checkpoint['epoch'])
        loss = float(checkpoint['loss'])
        return epoch, loss
