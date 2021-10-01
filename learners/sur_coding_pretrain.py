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


def fast_adapt(batch, learner, loss, device):
    adaptation_data, adaptation_labels, evaluation_data, evaluation_labels = batch
    # print(adaptation_labels, evaluation_labels)
    inputs = torch.cat((adaptation_data, evaluation_data))
    labels = torch.cat((adaptation_labels, evaluation_labels))

    predictions = learner(inputs)
    adaptation_error = loss(predictions, labels)

    adaptation_ber = comms_ber(labels, torch.sigmoid(predictions))
    adaptation_bler = comms_bler(labels, torch.sigmoid(predictions))
    return adaptation_error, adaptation_ber, adaptation_bler


def fast_adapt_eval(batch, learner, loss, device):
    adaptation_data, adaptation_labels, evaluation_data, evaluation_labels = batch

    predictions = learner(evaluation_data)
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
    noise_family = args.noise_family

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
    tasksets = get_tasksets(num_tasks, args, device, noise_family)

    num_val_tasks = tasksets.validation.images.shape[0] * \
        args.copies_of_vali_metrics

    model = CNN4(args.image_height, hidden_size=args.cnn_filter, layers=args.cnn_layers)
    model = model.to(device)

    if args.resume:
        print("resuming run and loading model from ",  os.path.join('saved_models/', args.name + "_" + str(args.start_iter) + '.pt'))
        model_path = os.path.join('saved_models/', args.name + "_" + str(args.start_iter) + '.pt')#('edin_models_final', args.name) + '_49999.pt'
        print("model path loading from ", model_path)

        model.load_state_dict(torch.load(model_path))
    elif args.eval_only:
        model_path = os.path.join('saved_models/', args.name + '_49999.pt')
        print("evaluation only, loading model from ", model_path)
        model.load_state_dict(torch.load(model_path))
    
    opt = optim.Adam(model.parameters(), meta_lr)
    loss = nn.BCEWithLogitsLoss()

    adapt_opt = optim.Adam(model.parameters(), lr=fast_lr)
    adapt_opt_state = adapt_opt.state_dict()

    if args.eval_only:
        f_name = os.path.join('results/test/', args.test_dataset, args.name) + '.json'
    else:
        f_name = os.path.join('results/', args.name) + '.json'
    
    # we will train these models for only a fourth of the standard number of iterations
    # because it has four time less data
    print("starting iteration: ", args.start_iter, " total iteration ", num_iterations)

    with tqdm.tqdm(total=num_iterations, disable=args.disable_tqdm) as pbar_epochs:
        for iteration in range(args.start_iter, num_iterations):
            opt.zero_grad()
            meta_train_error = 0.0
            meta_train_ber = 0.0
            meta_train_bler = 0.0
            meta_train_ber_list = []
            meta_train_bler_list = []
            if not args.eval_only:

                for task in range(meta_batch_size):
                    # Compute meta-training loss

                    batch = tasksets.train.sample(task_aug=args.task_aug)
                    evaluation_error, evaluation_ber, evaluation_bler = fast_adapt(batch,
                                                                                model,
                                                                                loss,
                                                                                device)

                    evaluation_error.backward()
                    meta_train_error += evaluation_error.item()
                    meta_train_ber += evaluation_ber.item()
                    meta_train_bler += evaluation_bler.item()
                    meta_train_ber_list.append(evaluation_ber.item())
                    meta_train_bler_list.append(evaluation_bler.item())

                # Average the accumulated gradients and optimize
                for p in model.parameters():
                    p.grad.data.mul_(1.0 / meta_batch_size)
                opt.step()
                # Print the metrics to tqdm panel
                pbar_epochs.set_description(
                    "Iteration {}: Error {:.4f} BER {:.4f} BLER {:.4f}".format(iteration,
                                                                            meta_train_error / meta_batch_size,
                                                                            meta_train_ber / meta_batch_size,
                                                                            meta_train_bler / meta_batch_size))

            pbar_epochs.update(1)

    torch.save(model.state_dict(), f=os.path.join(
        'saved_models', args.name + "_" + str(noise_family) + "_80k.pt"))

if __name__ == '__main__':
    args, device = get_args()
    main(args, device)
