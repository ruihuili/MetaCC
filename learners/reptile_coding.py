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

    if args.resume:
        print("resuming run and loading model from ",  os.path.join('saved_models/', args.name + "_" + str(args.start_iter) + '.pt'))
        model_path = os.path.join('saved_models/', args.name + "_" + str(args.start_iter) + '.pt')#('edin_models_final', args.name) + '_49999.pt'
        print("model path loading from ", model_path)

        [loaded_model_states, loaded_opt_state] = torch.load(model_path)
        model.load_state_dict(loaded_model_states)
    elif args.eval_only:
        model_path = os.path.join('saved_models/', args.name + '_49999.pt')
        print("evaluation only, loading model from ", model_path)
        [loaded_model_states, adapt_opt_state] = torch.load(model_path)
        model.load_state_dict(loaded_model_states)


    # opt = optim.Adam(model.parameters(), meta_lr)
    opt = optim.Adam(model.parameters(), meta_lr, betas=(meta_mom, 0.999))
    # opt = optim.SGD(model.parameters(), lr=meta_lr, momentum=meta_mom)

    loss = nn.BCEWithLogitsLoss()

    adapt_opt = optim.Adam(model.parameters(), lr=fast_lr)

    if not args.resume or args.eval_only:
        adapt_opt_state = adapt_opt.state_dict()
    
    if args.eval_only:
        f_name = os.path.join('results/test/', args.test_dataset, args.name) + '.json'
    else:
        f_name = os.path.join('results/', args.name) + '.json'

    if not args.resume:
        create_json_experiment_log(f_name)
    print("starting iteration: ", args.start_iter, " total iteration ", num_iterations)

    with tqdm.tqdm(total=num_iterations, disable=args.disable_tqdm) as pbar_epochs:
        for iteration in range(num_iterations):
            # opt.zero_grad()
            frac_done = float(iteration) / num_iterations
            new_lr = frac_done * meta_lr + (1 - frac_done) * meta_lr
            for pg in opt.param_groups:
                pg['lr'] = new_lr

            for p in model.parameters():
                p.grad = torch.zeros_like(p.data)
            meta_train_error = 0.0
            meta_train_ber = 0.0
            meta_train_bler = 0.0
            meta_train_ber_list = []
            meta_train_bler_list = []

            if not args.eval_only:
                for task in range(meta_batch_size):
                    # Compute meta-training loss
                    # learner = maml.clone()
                    learner = deepcopy(model)
                    adapt_opt = optim.Adam(learner.parameters(),
                                    lr=fast_lr,
                                    betas=(0, 0.999))
                    adapt_opt.load_state_dict(adapt_opt_state)
                    batch = tasksets.train.sample(task_aug=args.task_aug)
                    evaluation_error, evaluation_ber, evaluation_bler = fast_adapt(batch,
                                                                                learner,
                                                                                loss,
                                                                                adaptation_steps,
                                                                                shots,
                                                                                ways,
                                                                                adapt_opt,
                                                                                device)
                    adapt_opt_state = adapt_opt.state_dict()
                    for p, l in zip(model.parameters(), learner.parameters()):
                        p.grad.data.add_(-1.0,  l.data)
                    # evaluation_error.backward()
                    meta_train_error += evaluation_error.item()
                    meta_train_ber += evaluation_ber.item()
                    meta_train_bler += evaluation_bler.item()
                    meta_train_ber_list.append(evaluation_ber.item())
                    meta_train_bler_list.append(evaluation_bler.item())

                # Average the accumulated gradients and optimize
                for p in model.parameters():
                    p.grad.data.mul_(1.0 / meta_batch_size).add_(p.data)
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
                if args.eval_only: exit()

            if iteration % save_model_freq == (save_model_freq - 1):
                torch.save([model.state_dict(), adapt_opt_state], f=os.path.join('saved_models', args.name +"_" +str(iteration)+ ".pt"))
            pbar_epochs.update(1)

    total_time = time.time() - time_start
    experiment_update_dict = {
        'total_time': total_time, 'total_val_time': total_val_time}
    update_json_experiment_log_dict(experiment_update_dict, f_name)
    

    # validation is our testing for now


if __name__ == '__main__':
    args, device = get_args()
    main(args, device)