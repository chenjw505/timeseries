# -*- coding:utf-8 -*-
"""

@author: chenjw
@time: 2021/5/12 14:12
"""
import os
import numpy as np
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from src.dataloader.ett import ETTHour
from src.utils.entities import EarlyStopping
from src.module.models.informer.model import Informer


class Exp:
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _acquire_device(self):
        """
        args.use_gpu: 是否使用gpu
        args.use_multi_gpu: 是否使用多gpu
        args.gpu: 单gpu时gpu编号, 多gpu时主gpu编号
        args.devices: 多gpu时可用gpu编号, 包括主gpu
        """
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))  # 主gpu
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _build_dataset(self, flag):
        args = self.args

        data_dict = {
            'ETTh1': ETTHour,
            'other': None
        }
        # 选择对应的数据集
        Data = data_dict[args.data]

        # 参数设定
        time_encode = 1 if args.embed == 'timeF' else 0  # 构建时间戳特征
        if flag == 'train':
            shuffle_flag = True
            drop_last = True
            batch_size, freq = args.batch_size, args.freq
        elif flag == 'test':
            shuffle_flag = False
            drop_last = True
            batch_size, freq = args.batch_size, args.freq
        elif flag == 'pred':
            shuffle_flag = False
            drop_last = False
            batch_size = 1
            freq = args.detail_freq
            Data = None
        else:
            raise RuntimeError(f'build dataset error, flag: {flag}')

        dataset = Data(
            root_path=args.root_path,  # 数据目录
            data_path=args.data_path,  # 数据集名称
            flag=flag,  # train/valid/test
            size=[args.seq_len, args.label_len, args.pred_len],  # 数据集各部分长度
            features=args.features,  # 多变量/单变量
            target=args.target,  # ground_truth
            inverse=args.inverse,  # ground_truth是否用归一化的值，inverse=True表示y值用原始值，默认和x是一起归一化使用
            time_encode=time_encode,  # 时间戳特征
            freq=freq,  # 时间序列频率
            cols=args.cols
        )

        print(flag, len(dataset))

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)  # 最后不能形成batch size的数据drop掉

        return dataset, dataloader

    def _build_model(self):
        e_layers = self.args.e_layers if self.args.model == 'informer' else self.args.s_layers
        model = Informer(self.args.enc_in,
                         self.args.dec_in,
                         self.args.c_out,
                         self.args.seq_len,
                         self.args.label_len,
                         self.args.pred_len,
                         self.args.factor,
                         self.args.d_model,
                         self.args.n_heads,
                         e_layers,  # self.args.e_layers,
                         self.args.d_layers,
                         self.args.d_ff,
                         self.args.dropout,
                         self.args.attn,
                         self.args.embed,
                         self.args.freq,
                         self.args.activation,
                         self.args.output_attention,
                         self.args.distil,
                         self.args.mix,
                         self.device)

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _set_optimizer(self):
        return Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _set_criterion(self):
        return nn.MSELoss()

    def train(self, setting='default'):
        train_dataset, train_dataloader = self._build_dataset(flag='train')
        valid_dataset, valid_dataloader = self._build_dataset(flag='valid')
        test_dataset, test_dataloader = self._build_dataset(flag='test')

        # checkpoint path
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_dataloader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        optimizer = self._set_optimizer()
        criterion = self._set_criterion()

        # 混合精度
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_dataloader):
                iter_count += 1

                optimizer.zero_grad()
                pred, true = self._process_one_batch(
                    train_dataset, batch_x, batch_y, batch_x_mark, batch_y_mark)
                loss = criterion(pred, true)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
        #     train_loss = np.average(train_loss)
        #     vali_loss = self.vali(vali_data, vali_loader, criterion)
        #     test_loss = self.vali(test_data, test_loader, criterion)
        #
        #     print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
        #         epoch + 1, train_steps, train_loss, vali_loss, test_loss))
        #     early_stopping(vali_loss, self.model, path)
        #     if early_stopping.early_stop:
        #         print("Early stopping")
        #         break
        #
        #     adjust_learning_rate(optimizer, epoch + 1, self.args)
        #
        # best_model_path = path + '/' + 'checkpoint.pth'
        # self.model.load_state_dict(torch.load(best_model_path))
