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
        'results', args.name) + '.json'
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
        'results', args.name) + '.json'
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


def fast_adapt(batch, learner, loss, device):
    adaptation_data, adaptation_labels, evaluation_data, evaluation_labels = batch

    predictions = learner(adaptation_data)
    adaptation_error = loss(predictions, adaptation_labels)

    adaptation_ber = comms_ber(adaptation_labels, torch.sigmoid(predictions))
    adaptation_bler = comms_bler(adaptation_labels, torch.sigmoid(predictions))

    return adaptation_error, adaptation_ber, adaptation_bler


def fast_adapt_eval(batch, learner, loss, device):
    adaptation_data, adaptation_labels, evaluation_data, evaluation_labels = batch

    predictions = learner(evaluation_data)
    evaluation_error = loss(predictions, evaluation_labels)

    evaluation_ber = comms_ber(evaluation_labels, torch.sigmoid(predictions))
    evaluation_bler = comms_bler(evaluation_labels, torch.sigmoid(predictions))

    return evaluation_error, evaluation_ber, evaluation_bler


# def fast_adapt(batch, learner, loss, adaptation_steps, shots, ways, opt, device):
#     adaptation_data, adaptation_labels, evaluation_data, evaluation_labels = batch

#     # Adapt the model

#     # Evaluate the adapted model
#     predictions = learner(evaluation_data)
#     evaluation_error = loss(predictions, evaluation_labels)

#     evaluation_ber = comms_ber(evaluation_labels, torch.sigmoid(predictions))
#     evaluation_bler = comms_bler(evaluation_labels, torch.sigmoid(predictions))

#     return evaluation_error, evaluation_ber, evaluation_bler


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
    meta_lr = 0.001
    fast_lr = 0.1
    adaptation_steps = 5
    num_iterations = 50000
    meta_valid_freq = 10000
    save_model_freq = 2000

    # we could set num_tasks to something arbitrary like 20000
    num_tasks = num_iterations * meta_batch_size

    # Create a custom channel coding dataset
    tasksets = get_tasksets(num_tasks, args, device)

    num_val_tasks = tasksets.validation.images.shape[0] * \
        args.copies_of_vali_metrics
    # num_val_tasks = len(tasksets.validation.batch_arrange)

    model = CNN4(args.image_height, hidden_size=args.cnn_filter, layers=args.cnn_layers)
    model = model.to(device)
    opt = optim.Adam(model.parameters(), meta_lr)
    loss = nn.BCEWithLogitsLoss()

    adapt_opt = optim.Adam(model.parameters(), lr=fast_lr)
    adapt_opt_state = adapt_opt.state_dict()

    create_json_experiment_log(args)

    with tqdm.tqdm(total=num_iterations) as pbar_epochs:
        for iteration in range(num_iterations):
            opt.zero_grad()
            # for p in model.parameters():
            #     p.grad = torch.zeros_like(p.data)
            meta_train_error = 0.0
            meta_train_ber = 0.0
            meta_train_bler = 0.0
            meta_train_ber_list = []
            meta_train_bler_list = []

            for task in range(meta_batch_size):
                # Compute meta-training loss
                # learner = deepcopy(model)
                # learner = model
                # adapt_opt = optim.Adam(learner.parameters(),
                #                    lr=fast_lr,
                #                    betas=(0, 0.999))
                # adapt_opt.load_state_dict(adapt_opt_state)

                batch = tasksets.train.sample(task_aug=args.task_aug)
                evaluation_error, evaluation_ber, evaluation_bler = fast_adapt(batch,
                                                                               model,
                                                                               loss,
                                                                               device)
                # evaluation_error, evaluation_ber, evaluation_bler = fast_adapt(batch,
                #                                                                learner,
                #                                                                loss,
                #                                                                adaptation_steps,
                #                                                                shots,
                #                                                                ways,
                #                                                                adapt_opt,
                #                                                                device)
                # adapt_opt_state = adapt_opt.state_dict()
                # for p, l in zip(model.parameters(), learner.parameters()):
                #     p.grad.data.add_(-1.0,  l.data)
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

            if iteration % meta_valid_freq == (meta_valid_freq - 1):
                val_time_start = time.time()
                meta_valid_error = 0.0
                meta_valid_ber = 0.0
                meta_valid_bler = 0.0
                meta_valid_ber_list = []
                meta_valid_bler_list = []

                for task in range(num_val_tasks):
                    # Compute meta-validation loss
                    # learner = maml.clone()
                    # learner = deepcopy(model)
                    # adapt_opt = optim.Adam(learner.parameters(),
                    #                    lr=fast_lr,
                    #                    betas=(0, 0.999))
                    # adapt_opt.load_state_dict(adapt_opt_state)
                    batch = tasksets.validation.sample(task)
                    evaluation_error, evaluation_ber, evaluation_bler = fast_adapt_eval(batch,
                                                                                        model,
                                                                                        loss,
                                                                                        device)
                    # evaluation_error, evaluation_ber, evaluation_bler = fast_adapt(batch,
                    #                                                                learner,
                    #                                                                loss,
                    #                                                                adaptation_steps,
                    #                                                                shots,
                    #                                                                ways,
                    #                                                                adapt_opt,
                    #                                                                device)
                    meta_valid_error += evaluation_error.item()
                    meta_valid_ber += evaluation_ber.item()
                    meta_valid_bler += evaluation_bler.item()
                    meta_valid_ber_list.append(evaluation_ber.item())
                    meta_valid_bler_list.append(evaluation_bler.item())

                print('\n')
                print('Iteration {}'.format(iteration))
                print('Meta Val Error: {:.4f}'.format(
                    meta_valid_error / num_val_tasks))
                print('Meta Val BER: {:.4f}'.format(
                    meta_valid_ber / num_val_tasks))
                print('Meta Val BLER: {:.4f}'.format(
                    meta_valid_bler / num_val_tasks))
                experiment_update_dict = {'val_error': meta_valid_error / num_val_tasks,
                                          'val_ber': meta_valid_ber / num_val_tasks,
                                          'val_bler': meta_valid_bler / num_val_tasks,
                                          'train_error': meta_train_error / meta_batch_size,
                                          'train_ber': meta_train_ber / meta_batch_size,
                                          'train_bler': meta_train_bler / meta_batch_size,
                                          'train_ber_list': meta_train_ber_list,
                                          'train_bler_list': meta_train_bler_list,
                                          'val_ber_list': meta_valid_ber_list,
                                          'val_bler_list': meta_valid_bler_list,
                                          'train_ber_std': np.std(meta_train_ber_list),
                                          'train_bler_std': np.std(meta_train_bler_list),
                                          'val_ber_std': np.std(meta_valid_ber_list),
                                          'val_bler_std': np.std(meta_valid_bler_list),
                                          'iter': iteration + 1}
                update_json_experiment_log_dict(experiment_update_dict, args)
                total_val_time += time.time() - val_time_start

            if iteration % save_model_freq == (save_model_freq - 1):
                torch.save(model.state_dict(), f=os.path.join('models', args.name +"_" +str(iteration)+ ".pt"))
            pbar_epochs.update(1)

    total_time = time.time() - time_start
    experiment_update_dict = {
        'total_time': total_time, 'total_val_time': total_val_time}
    update_json_experiment_log_dict(experiment_update_dict, args)
    
    # validation is our testing for now


if __name__ == '__main__':
    args, device = get_args()
    main(args, device)
