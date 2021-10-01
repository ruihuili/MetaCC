#!/usr/bin/env python3


import os
import random
import time

import learn2learn as l2l
import numpy as np
import torch
import tqdm
from torch import nn, optim

from data_utils.datasets import get_tasksets
from models.models import CNN4
from utils.args_parser import get_args

from copy import deepcopy
from utils.utils import create_json_experiment_log, update_json_experiment_log_dict, comms_ber, comms_bler


def fast_adapt_eval(batch, learners, loss, device, max_iter=40):
    # need to do the sur adaptation here
    adaptation_data, adaptation_labels, evaluation_data, evaluation_labels = batch

    lambdas = torch.zeros(4).to(device)
    lambdas.requires_grad_(True)
    optimizer = torch.optim.Adadelta([lambdas], lr=(3e+3 / 5))

    for i in range(max_iter):
        optimizer.zero_grad()
        predictions_list = []
        lambdas_sm = torch.nn.Softmax(dim=0)(lambdas)
        for idx, learner in enumerate(learners):
            predictions_list.append(
                learner(adaptation_data) * lambdas_sm[idx])
        predictions = torch.stack(predictions_list).sum(dim=0)
        adaptation_error = loss(predictions, adaptation_labels)

        adaptation_error.backward()
        optimizer.step()

    # we need to combine the produced signals
    predictions_list = []
    lambdas_sm = torch.nn.Softmax(dim=0)(lambdas)
    for idx, learner in enumerate(learners):
        predictions_list.append(
            learner(evaluation_data) * lambdas_sm[idx])
    
    predictions = torch.stack(predictions_list).sum(dim=0)
    evaluation_error = loss(predictions, evaluation_labels)

    evaluation_ber = comms_ber(evaluation_labels, torch.sigmoid(predictions))
    evaluation_bler = comms_bler(evaluation_labels, torch.sigmoid(predictions))

    return evaluation_error, evaluation_ber, evaluation_bler


def main(args, device):
    # process the args
    ways =  args.ways #args.num_classes_per_set
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
    meta_lr = args.meta_lr #0.001  # 0.003
    fast_lr = args.task_lr #0.1 
    adaptation_steps = args.adapt_steps#5

    print("Meta LR ", meta_lr, " inner loop LR ", fast_lr, " adaptation steps ", adaptation_steps)
    num_iterations = args.num_iterations
    meta_valid_freq = args.meta_valid_freq
    save_model_freq = args.save_model_freq 

    # we could set num_tasks to something arbitrary like 20000
    num_tasks = num_iterations * meta_batch_size

    # Create a custom channel coding dataset
    tasksets = get_tasksets(num_tasks, args, device)

    num_val_tasks = tasksets.validation.images.shape[0] * \
        args.copies_of_vali_metrics
    # num_val_tasks = len(tasksets.validation.batch_arrange)

    noise_families = ["awgn", "memory", "multipath", "bursty"]

    models = []
    for noise_family in noise_families:
        model_path = os.path.join(
            'saved_models', args.name + '_' + str(noise_family) + '_80k.pt')
        print("evaluation only, loading model from ", model_path)
        model = CNN4(args.image_height, hidden_size=args.cnn_filter, layers=args.cnn_layers)
        model = model.to(device)
        model.load_state_dict(torch.load(model_path))
        models.append(model)
    
    loss = nn.BCEWithLogitsLoss()
    f_name = os.path.join('results', 'test', args.test_dataset, args.name) + '_erm_80k.json'

    create_json_experiment_log(f_name)

    val_time_start = time.time()
    meta_valid_error = 0.0
    meta_valid_ber = 0.0
    meta_valid_bler = 0.0
    meta_valid_ber_list = []
    meta_valid_bler_list = []

    print(num_val_tasks)
    for task in range(num_val_tasks):
        print(task)
        # Compute meta-validation loss
        batch = tasksets.validation.sample(task)
        evaluation_error, evaluation_ber, evaluation_bler = fast_adapt_eval(batch,
                                                                            models,
                                                                            loss,
                                                                            device)
        meta_valid_error += evaluation_error.item()
        meta_valid_ber += evaluation_ber.item()
        meta_valid_bler += evaluation_bler.item()
        meta_valid_ber_list.append(evaluation_ber.item())
        meta_valid_bler_list.append(evaluation_bler.item())

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
                              'val_bler_std': np.std(meta_valid_bler_list)}
    update_json_experiment_log_dict(experiment_update_dict, f_name)
    total_val_time += time.time() - val_time_start
 
    total_time = time.time() - time_start
    experiment_update_dict = {
        'total_time': total_time, 'total_val_time': total_val_time}
    update_json_experiment_log_dict(experiment_update_dict, f_name)
    


if __name__ == '__main__':
    args, device = get_args()
    main(args, device)
