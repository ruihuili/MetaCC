#!/usr/bin/env python3

from utils.utils import create_json_experiment_log, update_json_experiment_log_dict
import os
import random
import time

import learn2learn as l2l
import numpy as np
import torch
import tqdm
import commpy.channelcoding.convcode as cc

from data_utils.datasets import get_tasksets
from utils.args_parser import get_args

import collections
from functools import partial
import pickle


def comms_ber(y_targ, y_pred):
    num_unequal = np.not_equal(
        np.round(y_targ), np.round(y_pred)).astype('float64')
    ber = sum(sum(num_unequal)) * 1.0 / (np.size(y_targ))

    return ber


def comms_bler(y_targ, y_pred):
    y_pred = np.round(y_pred)

    tp0 = abs(y_targ - y_pred)
    bler = sum(np.sum(tp0, axis=1).astype('float') > 0) * \
        1.0 / (y_pred.shape[0])

    return bler


def main(args, device):
    # process the args
    
    seed = args.train_seed
    meta_batch_size = args.batch_size

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    time_start = time.time()
    total_val_time = 0.0


    # specify some details manually - based on L2L implementation
    meta_lr = 0.001  # 0.003
    fast_lr = 0.1
    adaptation_steps = 5
    num_iterations = 20000
    meta_valid_freq = 10000

    # we could set num_tasks to something arbitrary like 20000
    num_tasks = num_iterations * meta_batch_size

    # Create a custom channel coding dataset
    tasksets = get_tasksets(num_tasks, args, device)

    num_val_tasks = tasksets.validation.images.shape[0] * \
        args.copies_of_vali_metrics


    f_name = os.path.join('results/test/', args.test_dataset, args.name) + '.json'


    create_json_experiment_log(f_name)

    val_time_start = time.time()
    meta_valid_error = 0.0
    meta_valid_ber = 0.0
    meta_valid_bler = 0.0
    meta_valid_ber_list = []
    meta_valid_bler_list = []

    print("viterbi_tb_depths", args.tb_depth)
    print("num_val_tasks", num_val_tasks)
    for task in range(num_val_tasks): #400 go oom for maml
        # Compute meta-validation loss
        batch = tasksets.validation.sample(task)

        _, _, evaluation_data, evaluation_labels = batch

        inputs = evaluation_data.cpu().detach().numpy()
        targets = evaluation_labels.cpu().detach().numpy()


        outputs = []
        for i, ins in enumerate(inputs):
            trellis = cc.Trellis(np.array([2]), np.array([[7,5]]))
            # print(i, trellis)
            outs = cc.viterbi_decode(np.ndarray.flatten(ins[0]), trellis, tb_depth=args.tb_depth, decoding_type='unquantized')
            outputs.append(outs)

        evaluation_ber = comms_ber(targets, outputs)
        evaluation_bler = comms_bler(targets, outputs)
        print(evaluation_ber, evaluation_bler)


        meta_valid_error += 0
        meta_valid_ber += evaluation_ber.item()
        meta_valid_bler += evaluation_bler.item()
        meta_valid_ber_list.append(evaluation_ber.item())
        meta_valid_bler_list.append(evaluation_bler.item())

    print('\n')
    # print('Iteration {}'.format(iteration))
    print('Meta Val Error: {:.8f}'.format(
        meta_valid_error / num_val_tasks))
    print('Meta Val BER: {:.8f}'.format(
        meta_valid_ber / num_val_tasks))
    print('Meta Val BLER: {:.8f}'.format(
        meta_valid_bler / num_val_tasks))
    experiment_update_dict = {'val_error': meta_valid_error / num_val_tasks,
                                'val_ber': meta_valid_ber / num_val_tasks,
                                'val_bler': meta_valid_bler / num_val_tasks,
                                # 'train_error': meta_train_error / meta_batch_size,
                                # 'train_ber': meta_train_ber / meta_batch_size,
                                # 'train_bler': meta_train_bler / meta_batch_size,
                                # 'train_ber_list': meta_train_ber_list,
                                # 'train_bler_list': meta_train_bler_list,
                                'val_ber_list': meta_valid_ber_list,
                                'val_bler_list': meta_valid_bler_list,
                                # 'train_ber_std': np.std(meta_train_ber_list),
                                # 'train_bler_std': np.std(meta_train_bler_list),
                                'val_ber_std': np.std(meta_valid_ber_list),
                                'val_bler_std': np.std(meta_valid_bler_list),
                                # 'iter': iteration + 1
                                }
    update_json_experiment_log_dict(experiment_update_dict, f_name)
    total_val_time += time.time() - val_time_start

    experiment_update_dict = {
        'total_time': 0, 'total_val_time': total_val_time}
    update_json_experiment_log_dict(experiment_update_dict, f_name)


if __name__ == '__main__':
    args, device = get_args()
    main(args, device)
