import torch

def get_args():
    import argparse
    
    parser = argparse.ArgumentParser(description='Welcome to L2L Channel Coding!')

    # Training related arguments
    parser.add_argument("--meta_learner", type=str, help='learner name', default='maml')

    parser.add_argument('--meta_lr', type=float, default=0.001, help='Learning rate of overall MAML system')
    parser.add_argument('--task_lr', type=float, default=0.1, help='Learning rate per task gradient step')
    parser.add_argument('--adapt_steps', type=int, default=2, help='Number of inner loop adaptation steps')
    parser.add_argument('--first_order', action='store_true', default=False, help='First order MAML')

    parser.add_argument('--batch_size', nargs="?", type=int, default=10, help='Batch_size for experiment')
    parser.add_argument('--image_height', nargs="?", type=int, default=10, help='Block length')
    parser.add_argument('--image_width', nargs="?", type=int, default=2, help='Code rate')

    parser.add_argument('--train_seed', type=int, default=2)
    parser.add_argument('--val_seed', type=int, default=0)
    parser.add_argument('--seed', type=int, default=99)

    parser.add_argument('--name', type=str, default="tmp", help="Name of the experiment")
    parser.add_argument('--name_of_args_json_file', type=str, default="None")
    parser.add_argument('--disable_pbar', type=str, default="True")
    parser.add_argument('--start_iter', type=int, default=0, help='starting an experiment at step')
    parser.add_argument('--resume', action='store_true', default=False, help='resuming an experiment, use in conjunction with start_iter')

    parser.add_argument('--disable_tqdm', action='store_true', default=True, help='Show progress bar')
    
    # Testing related arguments
    parser.add_argument('--eval_only', action='store_true', default=False, help='Eval only')
    parser.add_argument('--save_adapt_acts', type=str, default="None")
    parser.add_argument('--test_dataset', type=str, default="synth")
    parser.add_argument('--tb_depth', type=int, default=2, help="viterbi param. indicating memory length")

    # Dataset related arguments
    parser.add_argument('--dataset_name', type=str, default="omniglot_dataset")    

    parser.add_argument('--num_classes_per_set', type=int, default=5, help='Number of classes to sample per set')
    parser.add_argument('--train_num_samples_per_class', type=int, default=5, help='Number of samples per set to sample')
    parser.add_argument('--val_num_samples_per_class', type=int, default=5, help='Number of samples per set to sample')
    parser.add_argument('--num_target_samples', type=int, default=15, help='number of samples in the target set')

    parser.add_argument('--ways', type=int, default=20, help='Number of ways to replace the above')
    parser.add_argument('--copies_of_train_metrics', type=int, default=1, help='Copies of the validation metrics in train')
    parser.add_argument('--copies_of_vali_metrics', type=int, default=50, help='Copies of the validation metrics in one validation')

    parser.add_argument('--train_metrics', type=dict, default={"awgn": {"snr":[-1, 2, 1], "param":{}}})
    parser.add_argument('--sample_metric', type=str, default="True", help='Sampling metrics from continuous space for comms data gen')

    # Network Architecture related arguments
    parser.add_argument('--cnn_layers', type=int, default=4, help='Number of classes to sample per set')

    parser.add_argument('--cnn_filter', type=int, default=64, help='Number of classes to sample per set')

    # Learner specific -- CAVIA
    parser.add_argument('--num_film_hidden_layers', type=int, default=1, help='[cavia] number of hidden layers used for FiLM')
    parser.add_argument('--context_layer_id', type=int, default=2, help='[cavia] layer id to plug in the context vector (single value for now)')
    parser.add_argument('--num_context_params', type=int, default=100, help='[cavia] number of context params')
    
    # Experiment specific -- impact of number of domains
    parser.add_argument('--n_setting_lim', type=int, default=500, help='Number of settings to sample for the study of #task vs accu')
    parser.add_argument('--n_classes_lim', type=int, default=50, help='Number of classes in each setting to sample for the study of #task vs accu')

    # Experiment specific -- task augmentation
    parser.add_argument('--task_aug', type=str, default="None")


    args = parser.parse_args()
    args_dict = vars(args)
    if args.name_of_args_json_file is not "None":
        args_dict = extract_args_from_json(args.name_of_args_json_file, args_dict)

    for key in list(args_dict.keys()):

        if str(args_dict[key]).lower() == "true":
            args_dict[key] = True
        elif str(args_dict[key]).lower() == "false":
            args_dict[key] = False

        print(key, args_dict[key], type(args_dict[key]))

    args = Bunch(args_dict)


    args.use_cuda = torch.cuda.is_available()
    if torch.cuda.is_available():  # checks whether a cuda gpu is available and whether the gpu flag is True
        device = torch.cuda.current_device()

        print("use GPU", device)
        print("GPU ID {}".format(torch.cuda.current_device()))

    else:
        print("use CPU")
        device = torch.device('cpu')  # sets the device to be CPU


    return args, device



class Bunch(object):
  def __init__(self, adict):
    self.__dict__.update(adict)

def extract_args_from_json(json_file_path, args_dict):
    import json
    summary_filename = json_file_path
    with open(summary_filename) as f:
        summary_dict = json.load(fp=f)

    for key in summary_dict.keys():
        if "continue_from" in key:
            pass
        elif "gpu_to_use" in key:
            pass
        else:
            args_dict[key] = summary_dict[key]

    return args_dict





