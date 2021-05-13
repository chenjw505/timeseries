# -*- coding:utf-8 -*-
"""

@author: chenjw
@time: 2021/4/29 10:41
"""
import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from src.module.models import LstNet
from src.dataloader import get_dataset
from src.utils.metrics import rse
from src import cst


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
        # common params
        self.model_name = args.model_name
        self.ckp_path = args.ckp_path
        self.train_print_epoch = args.train_print_epoch
        self.valid_print_epoch = args.valid_print_epoch
        self.early_stop_patience = args.early_stop_patience
        self.early_stop_delta = args.early_stop_delta
        self.best_loss = float('inf')
        self.early_stop_count = 0
        self.early_stop = False
        self.lr_decay_iter = args.lr_decay_iter
        self.init_epoch = 0

        # device
        if args.cuda >= 0:
            self.device = torch.device(
                f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu'
            )
        else:
            self.device = torch.device('cpu')
        print(f'Use accelerator: {self.device}')

        # dataset
        print('Load dataset')
        self.dataset = get_dataset(args)
        self.train_loader = DataLoader(dataset=self.dataset.train_dataset,
                                       batch_size=args.train_batch_size,
                                       shuffle=True)
        self.valid_loader = DataLoader(dataset=self.dataset.valid_dataset,
                                       batch_size=args.valid_batch_size,
                                       shuffle=True)
        self.test_loader = DataLoader(dataset=self.dataset.test_dataset,
                                      batch_size=args.valid_batch_size)

        # model, optimizer, criterion
        print(f'Build model: {self.model_name}')
        self.model, self.optimizer, self.criterion = self._build_model(args)

        # load checkpoint model
        if args.is_load_ckp:
            epoch, loss = self._load_checkpoint(args.ckp_name)
            print(f'Checkpoint: Epoch: {epoch}, Loss: {loss}')
            self.init_epoch = epoch

    def training(self, args):

        print('Start training')
        total_iter = 0

        for epoch in range(self.init_epoch, args.epochs):
            self.model.train()

            iter_loss = []
            for x_batch, y_batch in self.train_loader:

                # batch data
                x = x_batch.to(torch.float).to(self.device)
                y_true = y_batch.to(torch.float).to(self.device)
                y_pred = self.model(x)

                # reverse scale
                if args.normal_type == cst.NORMAL_TYPE_MATRIX:
                    scale = np.tile(self.dataset.scale_val, (y_true.size(0), 1))
                    scale = torch.tensor(scale, dtype=torch.float).to(self.device)
                    loss = self.criterion(y_pred * scale, y_true * scale)
                else:
                    loss = self.criterion(y_pred, y_true)

                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # record
                total_iter += 1
                iter_loss.append(loss.cpu().item())

                # TODO: lr decay
                if total_iter % self.lr_decay_iter == 0 and total_iter != 0:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.9

            mean_batch_loss = np.mean(iter_loss)

            # print train loss
            if epoch % self.train_print_epoch == 0:
                print(f'Epochs: {epoch}, Iterations: {total_iter}, Train Loss: {mean_batch_loss:.5f}')

            # print valid loss and metric
            if epoch % self.valid_print_epoch == 0 and epoch != 0:
                valid_loss, valid_metric = self.validate(args)
                print(f'==> Epochs: {epoch}, Valid Loss: {valid_loss:.5f}, Valid {args.metric}: {valid_metric:.5f}')
                # TODO: early stop and checkpoint
                self._early_stopping(valid_loss, epoch, args.is_ckp)
                if self.early_stop:
                    print('==> Early Stop!')
                    break

        # test metric
        self.test(args)

    def validate(self, args):
        """
        验证集预测结果，返回 loss 以及对应的 metric
        用于调参, 早停
        """
        self.model.eval()
        with torch.no_grad():
            print('Start validate')
            valid_metric_list = []
            valid_loss_list = []
            for x_batch, y_batch in self.valid_loader:
                x = x_batch.to(torch.float).to(self.device)
                y_true = y_batch.to(torch.float).to(self.device)
                y_pred = self.model(x)

                # reverse scale
                if args.normal_type == cst.NORMAL_TYPE_MATRIX:
                    scale = np.tile(self.dataset.scale_val, (y_true.size(0), 1))
                    scale = torch.tensor(scale, dtype=torch.float).to(self.device)
                    y_true = y_true * scale
                    y_pred = y_pred * scale

                valid_loss = self.criterion(y_pred, y_true).item()

                y_true = y_true.detach().cpu().numpy()
                y_pred = y_pred.detach().cpu().numpy()

                if args.metric == cst.MODEL_METRIC_RSE:
                    valid_metric = rse(y_true, y_pred)
                else:
                    raise RuntimeError(f'No available metric: {args.metric}')

                valid_loss_list.append(valid_loss)
                valid_metric_list.append(valid_metric)

            mean_valid_loss = np.mean(valid_loss_list)
            mean_valid_metric = np.mean(valid_metric_list)

            return mean_valid_loss, mean_valid_metric

    def test(self, args):
        self.model.eval()
        with torch.no_grad():
            print('Start test')
            y_true_list = []
            y_pred_list = []
            for x_batch, y_batch in self.test_loader:
                x = x_batch.to(torch.float).to(self.device)
                y_true = y_batch.to(torch.float).to(self.device)
                y_pred = self.model(x)

                # reverse scale
                if args.normal_type == cst.NORMAL_TYPE_MATRIX:
                    scale = np.tile(self.dataset.scale_val, (y_true.size(0), 1))
                    scale = torch.tensor(scale, dtype=torch.float).to(self.device)
                    y_true = y_true * scale
                    y_pred = y_pred * scale

                y_true = y_true.detach().cpu().numpy()
                y_pred = y_pred.detach().cpu().numpy()

                y_true_list.append(y_true)
                y_pred_list.append(y_pred)

            trues = np.concatenate(y_true_list)
            preds = np.concatenate(y_pred_list)

            print(f'Test shape: {trues.shape}, {preds.shape}')

            if args.metric == cst.MODEL_METRIC_RSE:
                test_metric = rse(trues, preds)
                print(f'Test metric: {test_metric:.5f}')
            else:
                raise RuntimeError(f'No available metric: {args.metric}')

            # save test metric and y
            today = time.strftime("%Y-%m-%d", time.localtime())
            exp_path = os.path.join(args.exp_path, today)
            save_path = os.path.join(exp_path, f'{self.model_name}_rse_{test_metric:.5f}')
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            params_dict = vars(args)
            json_str = json.dumps(params_dict, indent=2)
            with open(os.path.join(save_path, 'params.json'), 'w') as f:
                f.write(json_str)
            np.save(os.path.join(save_path, 'true.npy'), trues)
            np.save(os.path.join(save_path, 'pred.npy'), preds)
            print(f'Test result save to: {save_path}')

    def _build_model(self, args):
        """
        init model
        """
        if self.model_name == cst.MODEL_NAME_LSTNET:
            model = LstNet(seq_length=args.seq_length,
                           input_dim=args.input_dim,
                           cnn_kernel=args.cnn_kernel,
                           cnn_hidden_dim=args.cnn_hidden_dim,
                           rnn_hidden_dim=args.rnn_hidden_dim,
                           skip_period=args.skip_period,
                           skip_hidden_dim=args.skip_hidden_dim,
                           highway_window=args.highway_window,
                           p_dropout=args.p_dropout).to(self.device)

            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
            # 返回标量：batches loss sum, 默认是取 mean
            criterion = nn.MSELoss(reduction='sum')
        else:
            raise RuntimeError('Error model name')

        return model, optimizer, criterion

    def _save_checkpoint(self, ckp_name):
        """
        Args:
            ckp_name: ckp_modelname_epoch_loss.pt
        """
        if not os.path.exists(self.ckp_path):
            os.makedirs(self.ckp_path)

        names = ckp_name.split('_')
        epoch = int(names[-2])
        loss = float(names[-1][:-3])
        save_path = os.path.join(self.ckp_path, ckp_name)

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, save_path)
        print(f'==> Save checkpoint to: {save_path}')

    def _load_checkpoint(self, ckp_name):
        """
        Args:
            ckp_name: ckp_modelname_epoch_loss.pt
        """
        save_path = os.path.join(self.ckp_path, ckp_name)
        if not os.path.exists(save_path):
            raise RuntimeError(f'No available ckp_path: {save_path}')
        print(f'==> Load checkpoint from: {save_path}')
        checkpoint = torch.load(save_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = int(checkpoint['epoch'])
        loss = float(checkpoint['loss'])
        return epoch, loss

    def _early_stopping(self, valid_loss, epoch, is_ckp=False):
        """
        if valid_loss dose not reduce 3 times， stop train
        else checkpoint the new model of the loss
        Args:
            valid_loss:
            epoch:

        Returns:

        """
        ckp_name = f'ckp_{self.model_name}_{epoch}_{valid_loss:.5f}.pt'

        if self.best_loss is None:
            self.best_loss = valid_loss
            if is_ckp:
                self._save_checkpoint(ckp_name)

        elif valid_loss >= self.best_loss + self.early_stop_delta:
            self.early_stop_count += 1
            if self.early_stop_count >= self.early_stop_patience:
                self.early_stop = True
        else:
            self.best_loss = valid_loss
            self.early_stop_count = 0
            if is_ckp:
                self._save_checkpoint(ckp_name)
