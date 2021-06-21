#!/usr/bin/env python3

import json
import os
import random
import time

import learn2learn as l2l
import numpy as np
import torch
import tqdm
from torch import nn, optim

from datasets import get_tasksets
from models import CNN4
from parser_utils import get_args

from copy import deepcopy

def create_json_experiment_log(args):
    json_experiment_log_file_name = os.path.join(
        'results/test/', args.test_dataset, args.name) + '.json'
    experiment_summary_dict = {'val_error': [], 'val_ber': [], 'val_bler': [],
                               'train_error': [], 'train_ber': [], 'train_bler': [],
                               'total_time': [], 'total_val_time': [], 'iter': [],
                               'train_ber_list': [], 'train_bler_list': [],
                               'val_ber_list': [], 'val_bler_list': [],
                               'train_ber_std': [], 'train_bler_std': [],
                               'val_ber_std': [], 'val_bler_std': [],
                               '0shot_error': [], '0shot_ber': [],
                               '0shot_bler': [], '0shot_ber_list': [],
                                '0shot_bler_list': [], '0shot_ber_std':[], '0shot_bler_std': [],
                               }

    with open(json_experiment_log_file_name, 'w') as f:
        json.dump(experiment_summary_dict, fp=f)


def update_json_experiment_log_dict(experiment_update_dict, args):
    json_experiment_log_file_name = os.path.join(
        'results/test/', args.test_dataset, args.name) + '.json'
    with open(json_experiment_log_file_name, 'r') as f:
        summary_dict = json.load(fp=f)

    for key in experiment_update_dict:
        summary_dict[key].append(experiment_update_dict[key])

    with open(json_experiment_log_file_name, 'w') as f:
        json.dump(summary_dict, fp=f)


def comms_ber(y_targ, y_pred):
    y_targ = y_targ.cpu().detach().numpy()
    y_pred = y_pred.cpu().detach().numpy()

    num_unequal = np.not_equal(
        np.round(y_targ), np.round(y_pred)).astype('float64')
    ber = sum(sum(num_unequal)) * 1.0 / (np.size(y_targ))

    return ber


def comms_bler(y_targ, y_pred):
    y_targ = y_targ.cpu().detach().numpy()
    y_pred = y_pred.cpu().detach().numpy()
    y_pred = np.round(y_pred)

    tp0 = abs(y_targ - y_pred)
    bler = sum(np.sum(tp0, axis=1).astype('float') > 0) * \
        1.0 / (y_pred.shape[0])

    return bler


def fast_adapt(batch, learner, loss, adaptation_steps, shots, ways, opt, device):
    adaptation_data, adaptation_labels, evaluation_data, evaluation_labels = batch

    # Adapt the model
    for step in range(adaptation_steps):
        opt.zero_grad()
        adaptation_error = loss(learner(adaptation_data), adaptation_labels)
        adaptation_error.backward()
        opt.step()

    # Evaluate the adapted model
    predictions = learner(evaluation_data)
    evaluation_error = loss(predictions, evaluation_labels)

    evaluation_ber = comms_ber(evaluation_labels, torch.sigmoid(predictions))
    evaluation_bler = comms_bler(evaluation_labels, torch.sigmoid(predictions))

    return evaluation_error, evaluation_ber, evaluation_bler

def eva_wo_adapt(batch, learner, loss,  device):
    _, _, evaluation_data, evaluation_labels = batch

    # Evaluate the adapted model
    predictions = learner(evaluation_data)
    evaluation_error = loss(predictions, evaluation_labels)

    evaluation_ber = comms_ber(evaluation_labels, torch.sigmoid(predictions))
    evaluation_bler = comms_bler(evaluation_labels, torch.sigmoid(predictions))

    return evaluation_error, evaluation_ber, evaluation_bler


def main(args, device):
    # process the args
    ways = args.num_classes_per_set
    shots = args.train_num_samples_per_class
    seed = args.train_seed
    meta_batch_size = args.batch_size

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    time_start = time.time()
    total_val_time = 0.0

    if str(device) != 'cpu':
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    # specify some details manually - based on L2L implementation
    meta_lr = 1.0
    fast_lr = 0.1
    adaptation_steps = 5
    num_iterations = 50000
    meta_valid_freq = 10000
    save_model_freq = 2000

    meta_mom=0.0
    # we could set num_tasks to something arbitrary like 20000
    num_tasks = num_iterations * meta_batch_size

    # Create a custom channel coding dataset
    tasksets = get_tasksets(num_tasks, args, device)

    num_val_tasks = tasksets.validation.images.shape[0] * \
        args.copies_of_vali_metrics
    # num_val_tasks = len(tasksets.validation.batch_arrange)

    model = CNN4(args.image_height)
    model = model.to(device)
    model_path = os.path.join('models_before_may/', args.name + '_49999.pt')
    print("model path loading from ", model_path)
    load_dict = torch.load(model_path)
    model.load_state_dict(load_dict)
    # opt = optim.Adam(model.parameters(), meta_lr)
    # opt = optim.Adam(model.parameters(), meta_lr, betas=(meta_mom, 0.999))
    opt = optim.SGD(model.parameters(), lr=meta_lr, momentum=meta_mom)

    loss = nn.BCEWithLogitsLoss()

    adapt_opt = optim.Adam(model.parameters(), lr=fast_lr)
    adapt_opt_state = adapt_opt.state_dict()

    create_json_experiment_log(args)

    
    
    val_time_start = time.time()
    meta_valid_error = 0.0
    meta_valid_ber = 0.0
    meta_valid_bler = 0.0
    meta_valid_ber_list = []
    meta_valid_bler_list = []

    for task in range(num_val_tasks):
        # Compute meta-validation loss
        # learner = maml.clone()
        learner = deepcopy(model)
        adapt_opt = optim.Adam(learner.parameters(),
                            lr=fast_lr,
                            betas=(0, 0.999))
        adapt_opt.load_state_dict(adapt_opt_state)
        batch = tasksets.validation.sample(task)
        evaluation_error, evaluation_ber, evaluation_bler = fast_adapt(batch,
                                                                        learner,
                                                                        loss,
                                                                        adaptation_steps,
                                                                        shots,
                                                                        ways,
                                                                        adapt_opt,
                                                                        device)
        meta_valid_error += evaluation_error.item()
        meta_valid_ber += evaluation_ber.item()
        meta_valid_bler += evaluation_bler.item()
        meta_valid_ber_list.append(evaluation_ber.item())
        meta_valid_bler_list.append(evaluation_bler.item())

    print('\n')
    print('Meta Val Error: {:.4f}'.format(
        meta_valid_error / num_val_tasks))
    print('Meta Val BER: {:.4f}'.format(
        meta_valid_ber / num_val_tasks))
    print('Meta Val BLER: {:.4f}'.format(
        meta_valid_bler / num_val_tasks))
    experiment_update_dict = {'val_error': meta_valid_error / num_val_tasks,
                                'val_ber': meta_valid_ber / num_val_tasks,
                                'val_bler': meta_valid_bler / num_val_tasks,
                                'val_ber_list': meta_valid_ber_list,
                                'val_bler_list': meta_valid_bler_list,
                                'val_ber_std': np.std(meta_valid_ber_list),
                                'val_bler_std': np.std(meta_valid_bler_list),
                                }
    update_json_experiment_log_dict(experiment_update_dict, args)
    total_val_time += time.time() - val_time_start

    # -------------------------zeroshot-----------------------------
    meta_0shot_error = 0.0
    meta_0shot_ber = 0.0
    meta_0shot_bler = 0.0
    meta_0shot_ber_list = []
    meta_0shot_bler_list = []
    for task in range(num_val_tasks):
        learner = deepcopy(model)
        batch = tasksets.validation.sample(task)
        eva_0shot_error, eva_0shot_ber, eva_0shot_bler = eva_wo_adapt(batch,
                                                                        learner,
                                                                        loss,
                                                                        device)
        meta_0shot_error += eva_0shot_error.item()
        meta_0shot_ber += eva_0shot_ber.item()
        meta_0shot_bler += eva_0shot_bler.item()
        meta_0shot_ber_list.append(eva_0shot_ber.item())
        meta_0shot_bler_list.append(eva_0shot_bler.item())

    print('\n')
    print('Meta 0shot Error: {:.4f}'.format(meta_0shot_error / num_val_tasks))
    print('Meta 0shot BER: {:.4f}'.format(meta_0shot_ber / num_val_tasks))
    print('Meta 0shot BLER: {:.4f}'.format(meta_0shot_bler / num_val_tasks))

    experiment_update_dict = {'0shot_error': meta_0shot_error / num_val_tasks,
                                '0shot_ber': meta_0shot_ber / num_val_tasks,
                                '0shot_bler': meta_0shot_bler / num_val_tasks,
                                '0shot_ber_list': meta_0shot_ber_list,
                                '0shot_bler_list': meta_0shot_bler_list,
                                '0shot_ber_std': np.std(meta_0shot_ber_list),
                                '0shot_bler_std': np.std(meta_0shot_bler_list),
                                }
                                
    update_json_experiment_log_dict(experiment_update_dict, args)
    

    total_time = time.time() - time_start
    experiment_update_dict = {
        'total_time': total_time, 'total_val_time': total_val_time}
    update_json_experiment_log_dict(experiment_update_dict, args)

    # validation is our testing for now


if __name__ == '__main__':
    args, device = get_args()
    main(args, device)
