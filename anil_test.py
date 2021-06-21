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

import collections
from functools import partial
import pickle


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

def append_activation(acts, name, mod, inp, out):
    acts[name].append(out.cpu())

def fast_adapt(batch, learner, features, loss, adaptation_steps, shots, ways, device):
    adaptation_data, adaptation_labels, evaluation_data, evaluation_labels = batch

    adaptation_data = features(adaptation_data)
    evaluation_data = features(evaluation_data)

    # Adapt the model
    for step in range(adaptation_steps):
        adaptation_error = loss(learner(adaptation_data), adaptation_labels)
        learner.adapt(adaptation_error)

    # Evaluate the adapted model
    predictions = learner(evaluation_data)
    evaluation_error = loss(predictions, evaluation_labels)

    # print("input", evaluation_data[0], evaluation_labels[0], predictions[0], torch.sigmoid(predictions)[0])

    # print(evaluation_labels)
    evaluation_ber = comms_ber(evaluation_labels, torch.sigmoid(predictions))
    # print("ber", evaluation_ber)
    evaluation_bler = comms_bler(evaluation_labels, torch.sigmoid(predictions))

    return evaluation_error, evaluation_ber, evaluation_bler


def eva_wo_adapt(batch, learner, features, loss, device):
    _, _, evaluation_data, evaluation_labels = batch
# adaptation_data = features(adaptation_data)
    evaluation_data = features(evaluation_data)
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
    # num_val_tasks = len(tasksets.validation.batch_arrange)

    model = CNN4(args.image_height)
    features = model.features
    features.to(device)

    classifier = model.classifier
    classifier = l2l.algorithms.MAML(classifier, lr=fast_lr)
    classifier.to(device)


    from copy import deepcopy
    fix_feature = deepcopy(features)
    all_parameters = list(features.parameters()) + list(classifier.parameters())
    opt = torch.optim.Adam(all_parameters, meta_lr)

    # model = model.to(device)

    model_path = os.path.join('models_before_may/', args.name + '_49999.pt')
    print("model path loading from ", model_path)
    # maml = l2l.algorithms.MAML(model, lr=fast_lr, first_order=False)
    load_dict = torch.load(model_path)
    classifier.load_state_dict(load_dict['classifier'])
    features.load_state_dict(load_dict['features'])
    loss = nn.BCEWithLogitsLoss()

    create_json_experiment_log(args)

    val_time_start = time.time()
    meta_valid_error = 0.0
    meta_valid_ber = 0.0
    meta_valid_bler = 0.0
    meta_valid_ber_list = []
    meta_valid_bler_list = []

    acts_pre_adapt = collections.defaultdict(list)
    acts_post_adapt = collections.defaultdict(list)

    print("num_val_tasks", num_val_tasks)
    for task in range(num_val_tasks): #400 go oom for maml
        # Compute meta-validation loss
        learner = classifier.clone()
        batch = tasksets.validation.sample(task)

        # print(" save what", args.save_adapt_acts, type(args.save_adapt_acts), args.save_adapt_acts=="before")
        evaluation_error, evaluation_ber, evaluation_bler = eva_wo_adapt(batch,
                                                                            learner,
                                                                            features,
                                                                            loss,
                                                                            # adaptation_steps,
                                                                            # shots,
                                                                            # ways,
                                                                            device,
                                                                    )
        evaluation_error, evaluation_ber, evaluation_bler = fast_adapt(batch,
                                                                            learner,
                                                                            features,
                                                                            loss,
                                                                            adaptation_steps,
                                                                            shots,
                                                                            ways,
                                                                            device)

        meta_valid_error += evaluation_error.item()
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
    update_json_experiment_log_dict(experiment_update_dict, args)
    total_val_time += time.time() - val_time_start

    experiment_update_dict = {
        'total_time': 0, 'total_val_time': total_val_time}
    update_json_experiment_log_dict(experiment_update_dict, args)


if __name__ == '__main__':
    args, device = get_args()
    main(args, device)
