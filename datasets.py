import random
from collections import namedtuple

import numpy as np
import torch

from gen_channel_data import *


class ChannelCodingTaskDataset:
    """
    Channel Coding Task Dataset New loads examples from associated dataset files
    and stores them into memory.
    """

    def __init__(self, dataset_name, num_tasks, args, device):
        self.dataset_name = dataset_name
        self.device = device

        if self.dataset_name == 'train':
            self.support_samples_per_class = args.train_num_samples_per_class
            self.rng = np.random.RandomState(args.train_seed)
        else:
            self.support_samples_per_class = args.val_num_samples_per_class
            self.copies_of_vali_metrics = args.copies_of_vali_metrics

        self.num_classes_per_set = args.num_classes_per_set
        self.target_samples_per_class = args.num_target_samples
        # config file is e.g. set_nd_2/awgn_narrow.json
        # we need to parse it to get the name
        self.config = args.name_of_args_json_file.split('/')[-1].split('.')[0]

        # load the data
        print('Loading data for: ' + self.config, " real data for val? ", args.test_dataset)
        if self.dataset_name == 'train':
            data = np.load('dataset_v2/train/' + self.config + '_data.npz')
            self.images = data['train_images']
            self.labels = data['train_labels']
            print("train shapes", np.shape(self.images), np.shape(self.labels))
            # sample data to get smaller sets of tasks
            num_setting, num_classes, _, _, _ = list(np.shape(self.images))
            print("num_setting, num_classes", num_setting, num_classes)
            ids_settings = np.random.randint(num_setting, size = args.n_setting_lim)
            images_select = self.images[ids_settings]
            labels_select = self.labels[ids_settings]

            images = []#np.zeros((args.n_setting_lim, args.n_classes_lim, num_samples, h, w))
            labels = [] #np.zeros((args.n_setting_lim, args.n_classes_lim, num_samples, x))
            print(np.shape(self.images))

            for i in range(len(images_select)): 
                
                ids_classes = np.random.randint(num_classes, size = args.n_classes_lim)

                images.append(images_select[i][ids_classes])
                labels.append(labels_select[i][ids_classes])

            self.images = np.asarray(images)
            self.labels = np.asarray(labels)
            print(np.shape(self.images), np.all((self.images==0)), np.shape(self.labels))

        elif args.test_dataset[:8] == "realdata":
            print("loading real data for val/test")

            import scipy.io as sio
            if args.test_dataset == "realdata64_v2":
                print("loading real data for val/test from datasets/DataIn_64_v2.mat")
                self.labels = sio.loadmat("datasets/DataIn_64_v2.mat")['DataIn']
                self.images = sio.loadmat("datasets/DataOut_64_v2.mat")['DataOut']
            elif args.test_dataset == "realdata64_v1":
                print("loading real data for val/test from datasets/DataIn_64_v1.mat")
                self.labels = sio.loadmat("datasets/DataIn_64_v1.mat")['DataIn']
                self.images = sio.loadmat("datasets/DataOut_64_v1.mat")['DataOut']
            else:
                print("real data set undefined. EXIT")
                import sys
                sys.exit()
            # print(type(out_mat), np.shape(out_mat))

            # data = np.load('datasets/test/test_data.npz')
            # self.images = data['test_images']
            # self.labels = data['test_labels']
            # print("val shapes", np.shape(self.images), np.shape(self.labels))
            selected = np.random.choice(np.shape(self.images)[0], 250) # this should also shuffle the msgs
            self.images = np.reshape(self.images[selected, :500], (-1, 50, 50, 10, 2))
            self.labels = np.reshape(self.labels[selected, :500], (-1, 50, 50, 10))
            # print("after", np.shape(self.images), np.shape(self.labels))
            self.images = np.float32(self.images/6)
            self.labels = np.float32(self.labels)
            print(type(self.images), type(self.labels), type(self.images[0][0][0][0][0]))
            # sys.exit()

        else:
            print("load synth val data ")            
            data = np.load('dataset_v2/test/test_data.npz')
            self.images = data['test_images']
            self.labels = data['test_labels']
            print("val shapes", np.shape(self.images), np.shape(self.labels))
            print(type(self.images), type(self.labels), type(self.images[0][0][0][0][0]))
            # sys.exit()


        # put the data to the device for better speed
        self.images = torch.from_numpy(self.images).to(device=self.device)
        self.labels = torch.from_numpy(self.labels).to(device=self.device)
        self.dims = (self.images.shape[0],
                     self.images.shape[1],
                     self.images.shape[2])

    def sample(self, idx=None, task_aug="None"):
        """
        Possible values for task_aug:
            mixup1, mixup2, mixup3, dropout1, dropout2, dropout3, dropout4
            mixup1: some classes come from noise type 1 and other from noise type 2
            mixup2: a class may contain examples from two noise types
            mixup3: an example is a convex combination of two noise types
            mixup4: an example is a convex combination of two classes from the same noise type

            dropout1: number of classes is smaller
            dropout2: number of support examples / shots is smaller
            dropout3: some of the example inputs are zeroed out
            dropout4: some of the example targets are 0.5 messages
        """
        prob_aug = 0.5
        dropout2_now = False
        # print("dataset name ", self.dataset_name)
        # we only use task augmentation during training
        if task_aug == "None" or self.dataset_name != 'train' or np.random.uniform() < prob_aug:
            # we sample one task at a time
            if self.dataset_name == 'train':
                # print("seed", self.rng.randint(1, 999999))
                random.seed(self.rng.randint(1, 999999))
                # pick a random noise setup
                selected_noise_setup = random.randint(0, self.dims[0] - 1)
            else:
                # make sure the testing has the same order of tasks across runs
                random.seed(idx)
                # print("seed", idx, random.seed(idx))
                # use the previous ordering of copies
                selected_noise_setup = idx // self.copies_of_vali_metrics

            selected_support_classes = random.sample(
                range(self.dims[1]), self.num_classes_per_set)
            selected_query_classes = random.sample(
                range(self.dims[1]), self.num_classes_per_set)


            task_images = []
            task_labels = []
            for idx in range(self.num_classes_per_set):

                selected_supp_examples = random.sample(range(self.dims[2]), self.support_samples_per_class)
                selected_query_examples = random.sample(range(self.dims[2]), self.target_samples_per_class)
                class_support = selected_support_classes[idx]
                class_target = selected_query_classes[idx]
                # print(type(self.images[selected_noise_setup][class_support][selected_supp_examples]))
                # print(np.shape(self.images[selected_noise_setup][class_support][selected_query_examples]))
                combined_images = torch.cat((self.images[selected_noise_setup][class_support][selected_supp_examples], 
                self.images[selected_noise_setup][class_target][selected_query_examples]), 0)
                # print(type(combined_images))
                # print(np.shape(combined_images))

                combined_labels = torch.cat((self.labels[selected_noise_setup][class_support][selected_supp_examples], 
                self.labels[selected_noise_setup][class_target][selected_query_examples]), 0)
                task_images.append(combined_images)
                task_labels.append(combined_labels)

        else:
            if task_aug == 'mixup1':
                # generate two random noise types
                random.seed(self.rng.randint(1, 999999))
                noise_setup_1 = random.randint(0, self.dims[0] - 1)
                random.seed(self.rng.randint(1, 999999))
                noise_setup_2 = random.randint(0, self.dims[0] - 1)

                selected_classes = random.sample(
                    range(self.dims[1]), self.num_classes_per_set)
                # we can select the classes and then pick the given class from one of the two noise types
                # randomly decide on how many classes will come from noise type 1 and noise type 2
                # k = self.rng.randint(0, self.num_classes_per_set + 1)
                # at least 1 and at most 4 classes should come from noise type 1
                # to make sure it is always mixed between the two noise types
                k = self.rng.randint(1, self.num_classes_per_set)
                selected_noise_setups = [
                    noise_setup_1 if e < k else noise_setup_2 for e in range(self.num_classes_per_set)]

                task_images = []
                task_labels = []
                for selected_noise_setup, current_class in zip(selected_noise_setups, selected_classes):
                    selected_examples = random.sample(
                        range(self.dims[2]), self.support_samples_per_class + self.target_samples_per_class)
                    task_images.append(
                        self.images[selected_noise_setup][current_class][selected_examples])
                    task_labels.append(
                        self.labels[selected_noise_setup][current_class][selected_examples])
            
            elif task_aug == 'mixup2':
                # generate two random noise types
                random.seed(self.rng.randint(1, 999999))
                noise_setup_1 = random.randint(0, self.dims[0] - 1)
                random.seed(self.rng.randint(1, 999999))
                noise_setup_2 = random.randint(0, self.dims[0] - 1)

                selected_classes = random.sample(
                    range(self.dims[1]), self.num_classes_per_set)
                # we can select the classes and then pick the given class from one of the two noise types
                # randomly decide on how many classes will come from noise type 1 and noise type 2
                # k = self.rng.randint(0, self.num_classes_per_set + 1)
                # at least 1 and at most 4 classes should come from noise type 1
                # to make sure it is always mixed between the two noise types
                k = self.rng.randint(1, self.num_classes_per_set)
                # k is the number of mixed classes
                mixed_classes = [True if e < k else False for e in range(self.num_classes_per_set)]

                task_images = []
                task_labels = []
                for mixed_class, current_class in zip(mixed_classes, selected_classes):
                    if mixed_class:
                        # decide on the proportion how it is mixed - at least 1 support sample and at most (n-1)
                        num_noise_type_1_samples_support = self.rng.randint(1, self.support_samples_per_class)
                        num_noise_type_2_samples_support = self.support_samples_per_class - num_noise_type_1_samples_support
                        num_noise_type_1_samples_target = int(self.target_samples_per_class * num_noise_type_1_samples_support / self.support_samples_per_class)
                        num_noise_type_2_samples_target = self.target_samples_per_class - \
                            num_noise_type_1_samples_target
                        all_selected_examples = random.sample(range(self.dims[2]), self.support_samples_per_class + self.target_samples_per_class)
                        # now generate the samples and append them to the list
                        noise_setups_order = [
                            noise_setup_1,
                            noise_setup_2,
                            noise_setup_1,
                            noise_setup_2]

                        num_samples_list = [
                            num_noise_type_1_samples_support,
                            num_noise_type_2_samples_support,
                            num_noise_type_1_samples_target,
                            num_noise_type_2_samples_target]

                        task_images_l = []
                        task_labels_l = []
                        end = 0
                        for current_noise_setup, current_num_samples in zip(noise_setups_order, num_samples_list):
                            start = end
                            end += current_num_samples
                            current_examples = all_selected_examples[start:end]
                            task_images_l.append(
                                self.images[current_noise_setup][current_class][current_examples])
                            task_labels_l.append(
                                self.labels[current_noise_setup][current_class][current_examples])
                        task_images.append(torch.cat(task_images_l))
                        task_labels.append(torch.cat(task_labels_l))
                    else:
                        selected_examples = random.sample(
                            range(self.dims[2]), self.support_samples_per_class + self.target_samples_per_class)
                        # use the first noise type if not mixed
                        task_images.append(
                            self.images[noise_setup_1][current_class][selected_examples])
                        task_labels.append(
                            self.labels[noise_setup_1][current_class][selected_examples])

            elif task_aug == 'mixup3':
                # generate two random noise types
                random.seed(self.rng.randint(1, 999999))
                noise_setup_1 = random.randint(0, self.dims[0] - 1)
                random.seed(self.rng.randint(1, 999999))
                noise_setup_2 = random.randint(0, self.dims[0] - 1)

                selected_classes = random.sample(
                    range(self.dims[1]), self.num_classes_per_set)
                # randomly decide on how many classes will be mixed up in the task
                # at least 1 and at most all
                # the ordering does not matter so get the first classes as mixed
                k = self.rng.randint(1, self.num_classes_per_set + 1)
                mixed_classes = [True if e < k else False for e in range(
                    self.num_classes_per_set)]

                task_images = []
                task_labels = []
                for mixed_class, current_class in zip(mixed_classes, selected_classes):
                    if mixed_class:
                        selected_examples = random.sample(
                            range(self.dims[2]), self.support_samples_per_class + self.target_samples_per_class)
                        lamda = np.random.uniform()
                        mixed_images = lamda * \
                            self.images[noise_setup_1][current_class][selected_examples] + (
                                1.0 - lamda) * self.images[noise_setup_2][current_class][selected_examples]
                        mixed_labels = lamda * \
                            self.labels[noise_setup_1][current_class][selected_examples] + (
                                1.0 - lamda) * self.labels[noise_setup_2][current_class][selected_examples]
                        task_images.append(mixed_images)
                        task_labels.append(mixed_labels)
                    else:
                        selected_examples = random.sample(
                            range(self.dims[2]), self.support_samples_per_class + self.target_samples_per_class)
                        # use the first noise type if not mixed
                        task_images.append(
                            self.images[noise_setup_1][current_class][selected_examples])
                        task_labels.append(
                            self.labels[noise_setup_1][current_class][selected_examples])

            elif task_aug == 'mixup4':
                # generate two random noise types
                random.seed(self.rng.randint(1, 999999))
                noise_setup_1 = random.randint(0, self.dims[0] - 1)

                selected_classes = random.sample(
                    range(self.dims[1]), self.num_classes_per_set)
                selected_classes_2 = random.sample(
                    range(self.dims[1]), self.num_classes_per_set)
                # randomly decide on how many classes will be mixed up in the task
                # at least 1 and at most all
                # the ordering does not matter so get the first classes as mixed
                k = self.rng.randint(1, self.num_classes_per_set + 1)
                mixed_classes = [True if e < k else False for e in range(
                    self.num_classes_per_set)]
                task_images = []
                task_labels = []
                for mixed_class, current_class, current_class_2 in zip(mixed_classes, selected_classes, selected_classes_2):
                    if mixed_class:
                        selected_examples = random.sample(
                            range(self.dims[2]), self.support_samples_per_class + self.target_samples_per_class)
                        selected_examples_2 = random.sample(
                            range(self.dims[2]), self.support_samples_per_class + self.target_samples_per_class)
                        lamda = np.random.uniform()
                        mixed_images = lamda * \
                            self.images[noise_setup_1][current_class][selected_examples] + (
                                1.0 - lamda) * self.images[noise_setup_1][current_class_2][selected_examples_2]
                        mixed_labels = lamda * \
                            self.labels[noise_setup_1][current_class][selected_examples] + (
                                1.0 - lamda) * self.labels[noise_setup_1][current_class_2][selected_examples_2]
                        task_images.append(mixed_images)
                        task_labels.append(mixed_labels)
                    else:
                        selected_examples = random.sample(
                            range(self.dims[2]), self.support_samples_per_class + self.target_samples_per_class)
                        # use the first noise type if not mixed
                        task_images.append(
                            self.images[noise_setup_1][current_class][selected_examples])
                        task_labels.append(
                            self.labels[noise_setup_1][current_class][selected_examples])

            elif task_aug == 'dropout1':
                # generate two random noise types
                random.seed(self.rng.randint(1, 999999))
                noise_setup_1 = random.randint(0, self.dims[0] - 1)

                selected_classes = random.sample(
                    range(self.dims[1]), self.num_classes_per_set)
                # we can select the classes and then pick the given class from one of the two noise types
                # randomly decide on how many classes will come from noise type 1 and noise type 2
                # k = self.rng.randint(0, self.num_classes_per_set + 1)
                # at least 1 and at most 4 classes should come from noise type 1
                # to make sure it is always mixed between the two noise types
                k = self.rng.randint(1, self.num_classes_per_set)
                dropped_classes = [True if e < k else False for e in range(
                    self.num_classes_per_set)]
                task_images = []
                task_labels = []
                for dropped_class, current_class in zip(dropped_classes, selected_classes):
                    if not dropped_class:
                        selected_examples = random.sample(
                            range(self.dims[2]), self.support_samples_per_class + self.target_samples_per_class)
                        task_images.append(
                            self.images[noise_setup_1][current_class][selected_examples])
                        task_labels.append(
                            self.labels[noise_setup_1][current_class][selected_examples])

            elif task_aug == 'dropout2':
                random.seed(self.rng.randint(1, 999999))
                # pick a random noise setup
                selected_noise_setup = random.randint(0, self.dims[0] - 1)
                selected_classes = random.sample(
                    range(self.dims[1]), self.num_classes_per_set)

                # randomly generate number of support samples per class to use
                # randomly generate number of target samples per class to use
                # this needs to be the same across all classes
                # so that we keep constant shape
                # prob_aug_2 = 0.5
                # if np.random.uniform() < prob_aug_2:
                #     support_samples_per_class_l = self.rng.randint(
                #         1, self.support_samples_per_class)
                # else:
                #     support_samples_per_class_l = self.support_samples_per_class

                # if np.random.uniform() < prob_aug_2:
                #     # keep at least 10 samples
                #     # unlikely to help if there are too few of these
                #     target_samples_per_class_l = self.rng.randint(
                #         10, self.target_samples_per_class)
                # else:
                #     target_samples_per_class_l = self.target_samples_per_class
                support_samples_per_class_l = self.rng.randint(
                    1, self.support_samples_per_class)

                task_images = []
                task_labels = []
                for current_class in selected_classes:
                    selected_examples = random.sample(
                        range(self.dims[2]), support_samples_per_class_l + self.target_samples_per_class)
                    task_images.append(
                        self.images[selected_noise_setup][current_class][selected_examples])
                    task_labels.append(
                        self.labels[selected_noise_setup][current_class][selected_examples])
                dropout2_now = True
            
            elif task_aug == 'dropout3':
                random.seed(self.rng.randint(1, 999999))
                selected_noise_setup = random.randint(0, self.dims[0] - 1)
                selected_classes = random.sample(
                    range(self.dims[1]), self.num_classes_per_set)

                task_images = []
                task_labels = []
                # usually have only one class with augmentation
                prob_aug_2 = 1.0 / self.num_classes_per_set
                for current_class in selected_classes:
                    selected_examples = random.sample(
                        range(self.dims[2]), self.support_samples_per_class + self.target_samples_per_class)
                    if np.random.uniform() < prob_aug_2:
                        # keep at least 1 clean example, otherwise random
                        num_samples_masked = self.rng.randint(1, self.support_samples_per_class)
                        task_images_l = []
                        task_images_l.append(
                            self.images[selected_noise_setup][current_class][selected_examples][:num_samples_masked] * 0.0)
                        task_images_l.append(
                            self.images[selected_noise_setup][current_class][selected_examples][num_samples_masked:])
                        task_images.append(torch.cat(task_images_l))
                    else:
                        task_images.append(
                            self.images[selected_noise_setup][current_class][selected_examples])
                    task_labels.append(
                        self.labels[selected_noise_setup][current_class][selected_examples])

            elif task_aug == 'dropout4':
                random.seed(self.rng.randint(1, 999999))
                selected_noise_setup = random.randint(0, self.dims[0] - 1)
                selected_classes = random.sample(
                    range(self.dims[1]), self.num_classes_per_set)

                task_images = []
                task_labels = []
                # usually have only one class with augmentation
                prob_aug_2 = 1.0 / self.num_classes_per_set
                for current_class in selected_classes:
                    selected_examples = random.sample(
                        range(self.dims[2]), self.support_samples_per_class + self.target_samples_per_class)
                    if np.random.uniform() < prob_aug_2:
                        # keep at least 1 clean example, otherwise random
                        num_samples_masked = self.rng.randint(1, self.support_samples_per_class)
                        task_labels_l = []
                        task_labels_l.append(
                            self.labels[selected_noise_setup][current_class][selected_examples][:num_samples_masked] * 0.0 + 0.5)
                        task_labels_l.append(
                            self.labels[selected_noise_setup][current_class][selected_examples][num_samples_masked:])
                        task_labels.append(torch.cat(task_labels_l))
                    else:
                        task_labels.append(
                            self.labels[selected_noise_setup][current_class][selected_examples])
                    task_images.append(
                        self.images[selected_noise_setup][current_class][selected_examples])

            else:
                raise ValueError('The specified task augmentation is not implemented')

        task_images = torch.stack(task_images)
        task_labels = torch.stack(task_labels)

        if dropout2_now:
            support_set_images = task_images[:,
                                             :support_samples_per_class_l].contiguous()
            support_set_labels = task_labels[:,
                                            :support_samples_per_class_l].contiguous()
            target_set_images = task_images[:,
                                            support_samples_per_class_l:].contiguous()
            target_set_labels = task_labels[:,
                                            support_samples_per_class_l:].contiguous()
        else:
            support_set_images = task_images[:,
                                            :self.support_samples_per_class].contiguous()
            support_set_labels = task_labels[:,
                                            :self.support_samples_per_class].contiguous()
            target_set_images = task_images[:,
                                            self.support_samples_per_class:].contiguous()
            target_set_labels = task_labels[:,
                                            self.support_samples_per_class:].contiguous()

        # reshape
        h = support_set_images.shape[2]
        w = support_set_images.shape[3]
        c = 1  # 1 channel is required
        support_set_images = support_set_images.view(-1, c, h, w)
        support_set_labels = support_set_labels.view(-1, h)
        target_set_images = target_set_images.view(-1, c, h, w)
        target_set_labels = target_set_labels.view(-1, h)

        return support_set_images, support_set_labels, target_set_images, target_set_labels


class ChannelCodingTaskDatasetOld:
    """
    Channel Coding Task Dataset

    We will have one for each of training, validation and testing.

    """

    def __init__(self, dataset_name, num_tasks, args, device):
        self.dataset_name = dataset_name
        self.num_tasks = num_tasks
        self.device = device

        # save parameters from args
        self.seed = {"train": args.train_seed,
                     "val": args.val_seed, 'test': args.val_seed}

        # there are some other parameters that we need to set
        self.train_num_samples_per_class = args.train_num_samples_per_class
        self.num_classes_per_set = args.num_classes_per_set
        self.val_num_samples_per_class = args.val_num_samples_per_class
        self.sample_metric = args.sample_metric
        self.train_range_dict = args.train_range_dict
        self.num_target_samples = args.num_target_samples
        self.image_height = args.image_height

        # in our case we do not really do testing, so use the same as for validation
        if self.dataset_name == "train":
            self.batch_arrange = self.get_metric_arrangement(
                args.train_metrics, num_copies=args.copies_of_train_metrics)
        else:
            self.batch_arrange = self.get_metric_arrangement(
                args.val_metrics, num_copies=args.copies_of_vali_metrics)

    def __len__(self):
        return self.num_tasks

    def sample(self, idx=None):
        i = random.randint(0, len(self) - 1)
        if idx is None:
            idx = i

        x_support_set, x_target_set, y_support_set, y_target_set, seed = self.create_comms_set(
            self.dataset_name, seed=self.seed[self.dataset_name] + i, augment_images=False, idx=idx)

        # process the batch so that it has the right shape and details
        z_support_set, z_target_set = y_support_set[1], y_target_set[1]

        x_support_set = torch.Tensor(
            x_support_set).float().contiguous().to(device=self.device)
        x_target_set = torch.Tensor(
            x_target_set).float().contiguous().to(device=self.device)
        z_support_set = torch.Tensor(
            z_support_set).float().contiguous().to(device=self.device)
        z_target_set = torch.Tensor(
            z_target_set).float().contiguous().to(device=self.device)

        h = x_support_set.shape[2]
        w = x_support_set.shape[3]
        c = 1  # we can only use 1 channel
        x_support_set = x_support_set.view(-1, c, h, w)
        z_support_set = z_support_set.view(-1, h)
        x_target_set = x_target_set.view(-1, c, h, w)
        z_target_set = z_target_set.view(-1, h)

        return x_support_set, z_support_set, x_target_set, z_target_set

    def get_metric_arrangement(self, input_metrics, num_copies=1):

        org_metrics = []

        for noise_type, snr_param in input_metrics.items():
            if len(snr_param["snr"]) == 3:
                snr_low, snr_hi, snr_stp = snr_param["snr"]
                snr_list = np.arange(snr_low, snr_hi, snr_stp)
            else:
                snr_list = snr_param["snr"]
            if not snr_param["param"]:
                for snr in snr_list:
                    for j in range(num_copies):
                        org_metrics.append(
                            {"noise_type": noise_type, "snr": float(snr)})
                continue

            para_len = len(snr_param["param"][np.random.choice(
                list(snr_param["param"].keys()), size=1, replace=False)[0]])

            for snr in snr_list:
                for i in range(para_len):
                    met_dict = {"noise_type": noise_type, "snr": float(snr)}
                    for para in snr_param["param"].keys():
                        if noise_type == "t" and para == "vv":
                            assert snr_param["param"][para][i] >= 2, "T channel requires parameter vv to be no less than 2"
                        met_dict[para] = snr_param["param"][para][i]

                    for j in range(num_copies):
                        org_metrics.append(met_dict)
                    met_dict = {}
        return org_metrics

    def sample_a_metric(self, seed, train_range_dict, debug=False):
        """
        Given ranges of each channel type sample a criteria for data generation
        """
        if debug:
            print("sample a metric with seed ", seed, " dict: ",
                  train_range_dict, " noise types ", list(train_range_dict.keys()))
        rng = np.random.RandomState(seed)

        # channel_type_len = len(train_trange_dict)
        selected_channel = rng.choice(list(train_range_dict.keys()), size=1)[0]
        selected_channel_range = train_range_dict[selected_channel]
        if debug:
            print("selected type ", selected_channel,
                  " range ", selected_channel_range)

        met_dict = {"noise_type": selected_channel}

        for param, range_list in selected_channel_range.items():
            min_, max_ = range_list[0], range_list[1]
            val = np.random.uniform(min_, max_)
            met_dict[param] = val
        if debug:
            print("RETURNING sampled dict ", met_dict)
        return met_dict

    def create_comms_set(self, dataset_name, seed, augment_images=False, idx=0):
        """
        [from get set] Generates a task-set to be used for training or evaluation
        :param set_name: The name of the set to use, e.g. "train", "val" etc.
        :return: A task-set containing an image and label support set, and an image and label target set.
        """

        rng = np.random.RandomState(seed)

        x_images = []
        z_labels = []

        if dataset_name == "train":
            support_sample_per_class = self.train_num_samples_per_class
            num_classes = self.num_classes_per_set
        elif dataset_name == "val" or dataset_name == "test":
            support_sample_per_class = self.val_num_samples_per_class
            num_classes = 100

        id_c = int(idx % len(self.batch_arrange))
        if self.sample_metric and dataset_name == "train":
            batch_criteria = self.sample_a_metric(seed=rng.randint(
                1, 999999), train_range_dict=self.train_range_dict)
        else:
            batch_criteria = self.batch_arrange[id_c]

        # print("batch c", batch_criteria, dataset_name, id_c)
        for i in range(num_classes):
            k_seed = rng.randint(1, 999999)
            x_class_data, true_msgs = generate_viterbi_batch(batch_size=support_sample_per_class + self.num_target_samples,
                                                             block_len=self.image_height,
                                                             batch_criteria=batch_criteria,
                                                             seed=k_seed)
            x_class_data = np.array(x_class_data, dtype=np.float32)
            x_class_data = torch.from_numpy(x_class_data)

            x_images.append(x_class_data)
            z_labels.append(true_msgs)

        x_images = torch.stack(x_images)
        z_labels = np.array(z_labels, dtype=np.float32)

        support_set_images = x_images[:, :support_sample_per_class]
        support_set_labels = [
            z_labels[:, :support_sample_per_class], z_labels[:, :support_sample_per_class]]
        target_set_images = x_images[:, support_sample_per_class:]
        target_set_labels = [
            z_labels[:, support_sample_per_class:], z_labels[:, support_sample_per_class:]]

        return support_set_images, target_set_images, support_set_labels, target_set_labels, seed


BenchmarkTasksets = namedtuple(
    'BenchmarkTasksets', ('train', 'validation', 'test'))


def get_tasksets(num_tasks, args, device):
    train_tasks = ChannelCodingTaskDataset("train", num_tasks, args, device)
    val_tasks = ChannelCodingTaskDataset("val", num_tasks, args, device)
    test_tasks = ChannelCodingTaskDataset("test", num_tasks, args, device)

    return BenchmarkTasksets(train_tasks, val_tasks, test_tasks)
