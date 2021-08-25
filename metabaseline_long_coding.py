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
from utils.utils import create_json_experiment_log, update_json_experiment_log_dict, comms_ber, comms_bler


def normalize(x, epsilon=1e-8):
    x = x / (x.norm(p=2, dim=1, keepdim=True) + epsilon)
    return x


def cosine_distances_logits(a, b):
    # normalize both prototypes and queries
    a = normalize(a)
    b = normalize(b)
    # to get the logits we have minus and minus, so plus
    return torch.mm(a.reshape(a.shape[0], -1), b.reshape(b.shape[0], -1).t())


def fast_adapt_pretrain(batch, learner, loss, device):
    adaptation_data, adaptation_labels, evaluation_data, evaluation_labels = batch

    predictions = learner(adaptation_data)
    adaptation_error = loss(predictions, adaptation_labels)

    adaptation_ber = comms_ber(adaptation_labels, torch.sigmoid(predictions))
    adaptation_bler = comms_bler(adaptation_labels, torch.sigmoid(predictions))

    return adaptation_error, adaptation_ber, adaptation_bler


def fast_adapt(batch, model, scaling_param, loss, adaptation_steps, shots, ways, device):
    adaptation_data, adaptation_labels, evaluation_data, evaluation_labels = batch

    data = torch.cat([adaptation_data, evaluation_data])
    labels = torch.cat([adaptation_labels, evaluation_labels])
    embeddings = model(data)
    query_indices = adaptation_data.shape[0]  # 5*5
    support = embeddings[:query_indices]
    query = embeddings[query_indices:]

    # sequential approach
    predicted_bits_list = []
    evaluation_error = 0
    # there are 10 bits in a message
    for i in range(labels.shape[1]):
        # check if the bit has the same value across all examples
        if torch.sum(adaptation_labels[:, i] < 1).item() == 0 or torch.sum(adaptation_labels[:, i] > 0).item() == 0:
            predicted_bits_list.append(
                adaptation_labels[:, i][0].repeat(query.shape[0]))
        else:
            # create the prototypes for 0 and 1
            prototype_zero = support[adaptation_labels[:, i] < 1].mean(dim=0)
            prototype_one = support[adaptation_labels[:, i] > 0].mean(dim=0)
            # calculate distance between query examples and the prototypes for 0 and 1 for the current bit
            logits = cosine_distances_logits(
                query, torch.stack([prototype_zero, prototype_one]))
            logits = logits * scaling_param
            # get the probability the current bit should be 1
            predicted_bit_i = nn.Softmax(dim=1)(logits)[:, 1]
            # construct the predicted message
            predicted_bits_list.append(predicted_bit_i)
            # to make the signal cleaner we could round it
            # predicted_bits_list.append(torch.round(predicted_bit_i))
            # clamp the predicted value for BCE loss
            predicted_bit_i = torch.clamp(predicted_bit_i, 0.00001, 0.99999)
            # update the loss - all bits contribute separately
            evaluation_error += nn.BCELoss()(predicted_bit_i,
                                             evaluation_labels[:, i])
    predictions = torch.stack(predicted_bits_list).T

    # # we use approach similar to v2 from protonets
    # support = support.reshape(ways, shots, -1).mean(dim=1)
    # # the true messages are the codes corresponding to the prototypes
    # ordered_labels = adaptation_labels.reshape(ways, shots, -1).mean(dim=1)
    # logits = cosine_distances_logits(query, support)
    # # scale the logits by some learnable parameter
    # logits = logits * scaling_param

    # # only take the top prototype
    # # create simple labels corresponding to the different classes / messages
    # transformed_labels = torch.tensor(
    #     [[e for i in range(evaluation_data.shape[0] // ways)] for e in range(ways)], device=device).view(-1)
    # # this is directly like predicting one of the N classes / messages, so use CE
    # evaluation_error = nn.CrossEntropyLoss()(logits, transformed_labels)
    # predictions = ordered_labels[torch.argmax(logits, dim=1)]

    # if we have directly the predicted label, we should not use sigmoid to transform it
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
    num_iterations = 80000
    meta_valid_freq = 10000
    save_model_freq = 2000

    # we could set num_tasks to something arbitrary like 20000
    num_tasks = num_iterations * meta_batch_size

    # Create a custom channel coding dataset
    tasksets = get_tasksets(num_tasks, args, device)

    num_val_tasks = tasksets.validation.images.shape[0] * \
        args.copies_of_vali_metrics

    model = CNN4(args.image_height, hidden_size=args.cnn_filter, layers=args.cnn_layers )
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

    if args.eval_only:
        f_name = os.path.join('results/test/', args.test_dataset, args.name) + '.json'
    else:
        f_name = os.path.join('results/', args.name) + '.json'

    if not args.resume:
        create_json_experiment_log(f_name)
    print("starting iteration: ", args.start_iter, " total iteration ", num_iterations)


    # Stage 1: pre-training
    if not args.eval_only:
        with tqdm.tqdm(total=num_iterations, disable=args.disable_tqdm) as pbar_epochs:
            for iteration in range(num_iterations):
                opt.zero_grad()
                meta_train_error = 0.0
                meta_train_ber = 0.0
                meta_train_bler = 0.0
                meta_train_ber_list = []
                meta_train_bler_list = []

                for task in range(meta_batch_size):
                    # Compute meta-training loss
                    batch = tasksets.train.sample(task_aug=args.task_aug)
                    evaluation_error, evaluation_ber, evaluation_bler = fast_adapt_pretrain(batch,
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

                if iteration % meta_valid_freq == (meta_valid_freq - 1):
                    val_time_start = time.time()
                    meta_valid_error = 0.0
                    meta_valid_ber = 0.0
                    meta_valid_bler = 0.0
                    meta_valid_ber_list = []
                    meta_valid_bler_list = []

                    for task in range(num_val_tasks):
                        # Compute meta-validation loss
                        batch = tasksets.validation.sample(task)
                        evaluation_error, evaluation_ber, evaluation_bler = fast_adapt_pretrain(batch,
                                                                                                model,
                                                                                                loss,
                                                                                                device)
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
                    # experiment_update_dict = {'val_error': meta_valid_error / num_val_tasks,
                    #                           'val_ber': meta_valid_ber / num_val_tasks,
                    #                           'val_bler': meta_valid_bler / num_val_tasks,
                    #                           'train_error': meta_train_error / meta_batch_size,
                    #                           'train_ber': meta_train_ber / meta_batch_size,
                    #                           'train_bler': meta_train_bler / meta_batch_size,
                    #                           'train_ber_list': meta_train_ber_list,
                    #                           'train_bler_list': meta_train_bler_list,
                    #                           'val_ber_list': meta_valid_ber_list,
                    #                           'val_bler_list': meta_valid_bler_list,
                    #                           'train_ber_std': np.std(meta_train_ber_list),
                    #                           'train_bler_std': np.std(meta_train_bler_list),
                    #                           'val_ber_std': np.std(meta_valid_ber_list),
                    #                           'val_bler_std': np.std(meta_valid_bler_list),
                    #                           'iter': iteration + 1}
                    # update_json_experiment_log_dict(experiment_update_dict, f_name)
                    total_val_time += time.time() - val_time_start

                # if iteration % save_model_freq == (save_model_freq - 1):
                #     torch.save(model.state_dict(), f=os.path.join(
                #         'models', args.name + "_" + str(iteration) + ".pt"))
                pbar_epochs.update(1)

    # Stage 2: meta-learning with cosine distance - like protonets
    # prepare the model
    model.features_only = True
    opt = optim.Adam(model.features.parameters(), meta_lr)
    scaling_param = nn.Parameter(torch.tensor(10.0, device=device))
    scaling_opt = optim.Adam([scaling_param], meta_lr)

    with tqdm.tqdm(total=2*num_iterations, initial=num_iterations) as pbar_epochs:
        for iteration in range(num_iterations):
            opt.zero_grad()
            scaling_opt.zero_grad()
            meta_train_error = 0.0
            meta_train_ber = 0.0
            meta_train_bler = 0.0
            meta_train_ber_list = []
            meta_train_bler_list = []
            if not args.eval_only:

                for task in range(meta_batch_size):
                    # Compute meta-training loss
                    batch = tasksets.train.sample()
                    evaluation_error, evaluation_ber, evaluation_bler = fast_adapt(batch,
                                                                                model,
                                                                                scaling_param,
                                                                                loss,
                                                                                adaptation_steps,
                                                                                shots,
                                                                                ways,
                                                                                device)
                    if type(evaluation_error) != int:
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
                scaling_opt.step()
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
                    batch = tasksets.validation.sample(task)
                    evaluation_error, evaluation_ber, evaluation_bler = fast_adapt(batch,
                                                                                   model,
                                                                                   scaling_param,
                                                                                   loss,
                                                                                   adaptation_steps,
                                                                                   shots,
                                                                                   ways,
                                                                                   device)

                    if type(evaluation_error) != int:
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
                update_json_experiment_log_dict(experiment_update_dict, f_name)
                total_val_time += time.time() - val_time_start
                if args.eval_only: exit()

            if iteration % save_model_freq == (save_model_freq - 1):
                torch.save(model.state_dict(), f=os.path.join('models', args.name + "_" + str(iteration) + ".pt"))
            pbar_epochs.update(1)

    total_time = time.time() - time_start
    experiment_update_dict = {
        'total_time': total_time, 'total_val_time': total_val_time}
    update_json_experiment_log_dict(experiment_update_dict, f_name)


if __name__ == '__main__':
    args, device = get_args()
    main(args, device)
