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


def pairwise_distances_logits(a, b):
    n = a.shape[0]
    m = b.shape[0]
    logits = -((a.unsqueeze(1).expand(n, m, -1) -
                b.unsqueeze(0).expand(n, m, -1))**2).sum(dim=2)
    return logits


def apply_selection(learners, lambdas, data):
    sigmoid = nn.Sigmoid()
    lambdas_01 = sigmoid(lambdas)
    features_list = []

    for learner in learners:
        feature = learner(data)
        features_list.append(
            feature / (feature ** 2).sum(-1, keepdim=True).sqrt())
    n_cont = features_list[0].shape[0]
    concat_feat = torch.stack(features_list, -1)
    selected_features = (concat_feat * lambdas_01).reshape([n_cont, -1])
    
    return selected_features


def fast_adapt_eval(batch, learners, loss, device, max_iter=40):
    adaptation_data, adaptation_labels, evaluation_data, evaluation_labels = batch
    for learner in learners:
        learner.features_only = True

    lambdas = torch.zeros([1, 1, 4]).to(device)
    lambdas.requires_grad_(True)
    optimizer = torch.optim.Adadelta([lambdas], lr=(3e+3 / 5))

    for i in range(max_iter):
        selected_features_support = apply_selection(learners, lambdas, adaptation_data)
        predicted_bits_list = []
        support = selected_features_support
        # there are 10 bits in a message
        for i in range(adaptation_labels.shape[1]):
            # check if the bit has the same value across all examples
            if torch.sum(adaptation_labels[:, i] < 1).item() == 0 or torch.sum(adaptation_labels[:, i] > 0).item() == 0:
                predicted_bits_list.append(
                    adaptation_labels[:, i][0].repeat(support.shape[0]))
            else:
                # create the prototypes for 0 and 1
                prototype_zero = support[adaptation_labels[:, i] < 1].mean(dim=0)
                prototype_one = support[adaptation_labels[:, i] > 0].mean(dim=0)
                # calculate distance between support examples and the prototypes for 0 and 1 for the current bit
                logits = pairwise_distances_logits(
                    support, torch.stack([prototype_zero, prototype_one]))
                # get the probability the current bit should be 1
                predicted_bit_i = nn.Softmax(dim=1)(logits)[:, 1]
                # construct the predicted message
                predicted_bits_list.append(predicted_bit_i)

        predictions = torch.stack(predicted_bits_list).T
        optimizer.zero_grad()
        adaptation_error = loss(predictions, adaptation_labels)
        adaptation_error.backward()
        optimizer.step()

    selected_features_support = apply_selection(learners, lambdas, adaptation_data)
    selected_features_query = apply_selection(
        learners, lambdas, evaluation_data)

    predicted_bits_list = []
    evaluation_error = 0
    support = selected_features_support
    query = selected_features_query
    # there are 10 bits in a message
    for i in range(evaluation_labels.shape[1]):
        # check if the bit has the same value across all examples
        if torch.sum(adaptation_labels[:, i] < 1).item() == 0 or torch.sum(adaptation_labels[:, i] > 0).item() == 0:
            predicted_bits_list.append(
                adaptation_labels[:, i][0].repeat(query.shape[0]))
        else:
            # create the prototypes for 0 and 1
            prototype_zero = support[adaptation_labels[:, i] < 1].mean(dim=0)
            prototype_one = support[adaptation_labels[:, i] > 0].mean(dim=0)
            # calculate distance between query examples and the prototypes for 0 and 1 for the current bit
            logits = pairwise_distances_logits(
                query, torch.stack([prototype_zero, prototype_one]))
            # get the probability the current bit should be 1
            predicted_bit_i = nn.Softmax(dim=1)(logits)[:, 1]
            # construct the predicted message
            predicted_bits_list.append(predicted_bit_i)
            # clamp the predicted value for BCE loss
            predicted_bit_i = torch.clamp(predicted_bit_i, 0.00001, 0.99999)
            # update the loss - all bits contribute separately
            evaluation_error += nn.BCELoss()(predicted_bit_i,
                                             evaluation_labels[:, i])

    # combine the individual bit predictions into one message
    predictions = torch.stack(predicted_bits_list).T
    evaluation_ber = comms_ber(evaluation_labels, predictions)
    evaluation_bler = comms_bler(evaluation_labels, predictions)

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
    f_name = os.path.join('results', 'test', args.test_dataset, args.name) + '_proto_80k.json'

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
