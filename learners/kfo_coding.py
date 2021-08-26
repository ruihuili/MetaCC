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


def fast_adapt(batch, features, classifier, update, diff_sgd, loss, adaptation_steps,
               shots, ways, device):
    adaptation_data, adaptation_labels, evaluation_data, evaluation_labels = batch
    adaptation_data = features(adaptation_data)
    evaluation_data = features(evaluation_data)

    adaptation_data = torch.squeeze(adaptation_data).permute(0, 2, 1)
    unflatten_adapt = torch.nn.Unflatten(0, (adaptation_data.shape[0:2]))
    adaptation_data = torch.flatten(adaptation_data, 0, 1)

    evaluation_data = torch.squeeze(evaluation_data).permute(0, 2, 1)
    unflatten_eva = torch.nn.Unflatten(0, (evaluation_data.shape[0:2]))
    evaluation_data = torch.flatten(evaluation_data, 0, 1)

    # Adapt the model & learned update
    for step in range(adaptation_steps):
        output = classifier(adaptation_data)
        # print("after classifier",  adaptation_data.shape)
        output = unflatten_adapt(output).squeeze()

        adaptation_error = loss(output, adaptation_labels)
        if step > 0:  # Update the learnable update function
            update_grad = torch.autograd.grad(adaptation_error,
                                              update.parameters(),
                                              create_graph=True,
                                              retain_graph=True)
            diff_sgd(update, update_grad)
        classifier_updates = update(adaptation_error,
                                    classifier.parameters(),
                                    create_graph=True,
                                    retain_graph=True)
        diff_sgd(classifier, classifier_updates)

    # Evaluate the adapted model
    predictions = unflatten_eva(classifier(evaluation_data)).squeeze()
    evaluation_error = loss(predictions, evaluation_labels)

    evaluation_ber = comms_ber(evaluation_labels, torch.sigmoid(predictions))
    evaluation_bler = comms_bler(evaluation_labels, torch.sigmoid(predictions))

    return evaluation_error, evaluation_ber, evaluation_bler

def eva_wo_adapt(batch, features, classifier, update, diff_sgd, loss, device):
    _, _, evaluation_data, evaluation_labels = batch
    
    evaluation_data = features(evaluation_data)

    evaluation_data = torch.squeeze(evaluation_data).permute(0, 2, 1)
    unflatten_eva = torch.nn.Unflatten(0, (evaluation_data.shape[0:2]))
    evaluation_data = torch.flatten(evaluation_data, 0, 1)

    # Evaluate the adapted model
    predictions = unflatten_eva(classifier(evaluation_data)).squeeze()
    evaluation_error = loss(predictions, evaluation_labels)

    evaluation_ber = comms_ber(evaluation_labels, torch.sigmoid(predictions))
    evaluation_bler = comms_bler(evaluation_labels, torch.sigmoid(predictions))

    return evaluation_error, evaluation_ber, evaluation_bler

def main(args, device):
    # process the args
    ways = args.ways #args.num_classes_per_set
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

    model = CNN4(args.image_height)
    model.to(device)
    features = model.features
    classifier = model.classifier
    kfo_transform = l2l.optim.transforms.KroneckerTransform(l2l.nn.KroneckerLinear)
    fast_update = l2l.optim.ParameterUpdate(
        parameters=classifier.parameters(),
        transform=kfo_transform,
    )
    fast_update.to(device)

    if args.resume:
        print("resuming run and loading model from ",  os.path.join('saved_models/', args.name + "_" + str(args.start_iter) + '.pt'))
        model_path = os.path.join('saved_models/', args.name + "_" + str(args.start_iter) + '.pt')#('edin_models_final', args.name) + '_49999.pt'
        print("model path loading from ", model_path)

        load_dict = torch.load(model_path)
        model.load_state_dict(load_dict['model'])
        fast_update.load_state_dict(load_dict['fast_update'])

    elif args.eval_only:
        model_path = os.path.join('saved_models/', args.name + '_49999.pt')
        print("evaluation only, loading model from ", model_path)
        load_dict = torch.load(model_path)
        model.load_state_dict(load_dict['model'])
        fast_update.load_state_dict(load_dict['fast_update'])
    
    diff_sgd = l2l.optim.DifferentiableSGD(lr=fast_lr)

    all_parameters = list(model.parameters()) + list(fast_update.parameters())
    opt = torch.optim.Adam(all_parameters, meta_lr)
    loss = nn.BCEWithLogitsLoss()

    if args.eval_only:
        f_name = os.path.join('results/test/', args.test_dataset, args.name) + '.json'
    else:
        f_name = os.path.join('results/', args.name) + '.json'

    if not args.resume:
        create_json_experiment_log(f_name)
    print("starting iteration: ", args.start_iter, " total iteration ", num_iterations)


    with tqdm.tqdm(total=num_iterations, disable=args.disable_tqdm) as pbar_epochs:
        for iteration in range(num_iterations):
            opt.zero_grad()
            meta_train_error = 0.0
            meta_train_ber = 0.0
            meta_train_bler = 0.0
            meta_train_ber_list = []
            meta_train_bler_list = []

            if not args.eval_only:

                for task in range(meta_batch_size):
                    # Compute meta-training loss
                    task_features = l2l.clone_module(features)
                    task_classifier = l2l.clone_module(classifier)
                    task_update = l2l.clone_module(fast_update)
                    batch = tasksets.train.sample(task_aug=args.task_aug)

                    eva_0shot_error, eva_0shot_ber, eva_0shot_bler = eva_wo_adapt(batch,
                                                                                    task_features,
                                                                                    task_classifier,
                                                                                    task_update,
                                                                                    diff_sgd,
                                                                                    loss,
                                                                                    device)


                    evaluation_error, evaluation_ber, evaluation_bler = fast_adapt(batch,
                                                                                task_features,
                                                                                task_classifier,
                                                                                task_update,
                                                                                diff_sgd,
                                                                                loss,
                                                                                adaptation_steps,
                                                                                shots,
                                                                                ways,
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
                for p in fast_update.parameters():
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
                    task_features = l2l.clone_module(features)
                    task_classifier = l2l.clone_module(classifier)
                    task_update = l2l.clone_module(fast_update)
                    batch = tasksets.validation.sample(task)
                    evaluation_error, evaluation_ber, evaluation_bler = fast_adapt(batch,
                                                                                   task_features,
                                                                                   task_classifier,
                                                                                   task_update,
                                                                                   diff_sgd,
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

# -------------------------zeroshot-----------------------------
                meta_0shot_error = 0.0
                meta_0shot_ber = 0.0
                meta_0shot_bler = 0.0
                meta_0shot_ber_list = []
                meta_0shot_bler_list = []
                for task in range(num_val_tasks):
                    task_features = l2l.clone_module(features)
                    task_classifier = l2l.clone_module(classifier)
                    task_update = l2l.clone_module(fast_update)
                    batch = tasksets.validation.sample(task)
                    eva_0shot_error, eva_0shot_ber, eva_0shot_bler = eva_wo_adapt(batch,
                                                                                   task_features,
                                                                                   task_classifier,
                                                                                   task_update,
                                                                                   diff_sgd,
                                                                                   loss,
                                                                                   device)
                    meta_0shot_error += eva_0shot_error.item()
                    meta_0shot_ber += eva_0shot_ber.item()
                    meta_0shot_bler += eva_0shot_bler.item()
                    meta_0shot_ber_list.append(eva_0shot_ber.item())
                    meta_0shot_bler_list.append(eva_0shot_bler.item())

                print('\n')
                print('Iteration {}'.format(iteration))
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
                                          
                update_json_experiment_log_dict(experiment_update_dict, f_name)
                # -------------------------zeroshot-----------------------------
                if args.eval_only: exit()
                
            if iteration % save_model_freq == (save_model_freq - 1):
                torch.save({'model': model.state_dict(), 'fast_update': fast_update.state_dict()}, f=os.path.join(
                    'models', args.name + "_" + str(iteration) + ".pt"))
            pbar_epochs.update(1)

    total_time = time.time() - time_start
    experiment_update_dict = {
        'total_time': total_time, 'total_val_time': total_val_time}
    update_json_experiment_log_dict(experiment_update_dict, f_name)

    # validation is our testing for now


if __name__ == '__main__':
    args, device = get_args()
    main(args, device)
