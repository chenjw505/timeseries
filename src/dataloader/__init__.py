# -*- coding:utf-8 -*-
"""

@author: chenjw
@time: 2021/4/29 09:15
"""
from .generator import DatasetLSTNet
from src import cst


def get_dataset(args):
    if args.model_name == cst.MODEL_NAME_LSTNET:
        dataset = DatasetLSTNet(data_dir=args.data_path,
                                file_name=args.file_name,
                                p_train=args.p_train,
                                p_valid=args.p_valid,
                                p_test=args.p_test,
                                seq_length=args.seq_length,
                                horizon=args.horizon,
                                normal_type=args.normal_type)
    else:
        raise RuntimeError(f'No available model name: {args.model_name}')

    return dataset
