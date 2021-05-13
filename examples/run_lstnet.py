# -*- coding:utf-8 -*-
"""
configs/xxx.gin 用于实验环境下搜索最优参数
examples/args  用于直接跑最优参数

@author: chenjw
@time: 2021/5/6 10:32
"""

import argparse
from src.network import Network
from src import cst


def parse_args():
    """Parse arguments."""
    # Parameters settings
    parser = argparse.ArgumentParser(description="PyTorch implementation of LSTNet")

    # Common
    parser.add_argument('--model_name', type=str, default=cst.MODEL_NAME_LSTNET, help='path to dataset')
    parser.add_argument('--ckp_path', type=str, default='./checkpoints', help='dir to checkpoint')
    parser.add_argument('--is_ckp', type=bool, default=True)  # 训练过程是否checkpoint
    parser.add_argument('--data_path', type=str, default='./benchmarks', help='dir to dataset')
    parser.add_argument('--exp_path', type=str, default='./experiments', help='dir to checkpoint')

    parser.add_argument('--is_load_ckp', type=bool, default=False)  # 初始化模型or加载已训练模型
    parser.add_argument('--ckp_name', type=str, default='ckp_LSTNet_35_37.32135.pt')

    # Dataset setting
    parser.add_argument('--file_name', type=str, default="traffic.txt", help='name for dataset')
    parser.add_argument('--p_train', type=float, default=0.6)
    parser.add_argument('--p_valid', type=float, default=0.2)
    parser.add_argument('--p_test', type=float, default=0.2)
    parser.add_argument('--seq_length', type=int, default=24 * 7, help='seq_length')
    parser.add_argument('--horizon', type=int, default=1, help='horizon')
    parser.add_argument('--train_batch_size', type=int, default=128, help='train batch size [128]')
    parser.add_argument('--valid_batch_size', type=int, default=64, help='valid batch size [64]')
    parser.add_argument('--test_batch_size', type=int, default=64, help='test batch size [64]')
    parser.add_argument('--normal_type', type=int, default=cst.NORMAL_TYPE_MATRIX)

    # Model parameters setting
    parser.add_argument('--input_dim', type=int, default=862, help='input_dim')
    parser.add_argument('--cnn_kernel', type=int, default=6, help='cnn_kernel')
    parser.add_argument('--cnn_hidden_dim', type=int, default=100, help='cnn_num_hidden')
    parser.add_argument('--rnn_hidden_dim', type=int, default=100, help='rnn_num_hidden')
    parser.add_argument('--skip_period', type=int, default=24, help='skip_period')
    parser.add_argument('--skip_hidden_dim', type=int, default=10, help='skip_num_hidden')
    parser.add_argument('--highway_window', type=int, default=24, help='highway_window')
    parser.add_argument('--p_dropout', type=float, default=0.2,
                        help='dropout applied to layers (0 = no dropout)')

    # Training parameters setting
    parser.add_argument('--cuda', type=int, default=0, help='cuda id')
    parser.add_argument('--early_stop_patience', type=int, default=3, help='')
    parser.add_argument('--early_stop_delta', type=int, default=0, help='')
    parser.add_argument('--train_print_epoch', type=int, default=2, help='')
    parser.add_argument('--valid_print_epoch', type=int, default=5, help='')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate [0.001]')
    parser.add_argument('--lr_decay_iter', type=float, default=5000, help='learning rate reduce 0.1 by 5000 iterations')
    parser.add_argument('--metric', type=str, default=cst.MODEL_METRIC_RSE)


    # parse the arguments
    args = parser.parse_args()

    print('=' * 20)
    for k, v in vars(args).items():
        print(f'{k}: {v}')
    print('=' * 20)

    return args


def run():
    # Params
    args = parse_args()

    # Network
    network = Network(args)

    # Train
    network.training(args)

