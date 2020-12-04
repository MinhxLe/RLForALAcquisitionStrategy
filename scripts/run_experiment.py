import json
import argparse
import os
from src.constants import (
    EXPERIMENTS_RESULT_DIR,
)
from src.experiment.experiment_manager import (
    RandomExperimentManager,
    LCExperimentManager,
    UCBBanditExperimentManager
)

parser = argparse.ArgumentParser()
# mnist model args
parser.add_argument('--model_num_filters', type=int, default=8)
parser.add_argument('--model_filter_size', type=int, default=3)
parser.add_argument('--model_pool_size', type=int, default=2)
parser.add_argument('--model_num_classes', type=int, default=10)

# train args
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--train_epochs', type=int, default=5, help="number of train epochs per al step")
parser.add_argument('--retrain_model_from_scratch', type=bool, default=True, help="retrain model from scratch per AL step")
# parser.add_argument('--learning_rate', type=int, default=0.01)

# log args
parser.add_argument('--experiment_name', type=str, default="", help="experiment id")
parser.add_argument('--train_log_interval', type=int, default=0, help="log interval for internal train interval(<0 means not log)")
parser.add_argument('--debug', type=bool, default=False, help="debug mode")
parser.add_argument("--save_model_interval", type=int, default=10, help="save model frequency")
parser.add_argument("--stdout", type=bool, default=True, help="print logs to stdout")

# experiment args
parser.add_argument('--dataset', type=str, help='dataset selector', default='cifar10')
parser.add_argument('--al_sampler', type=str)
parser.add_argument('--al_epochs', type=int, default=100)
parser.add_argument("--seed", type=int, default=42, help="random seed")
parser.add_argument("--n_experiment_runs", type=int, default=1, help="number of times to run experiment")
parser.add_argument('--al_step_percentage', type=float, default=0.005, help="percentage of data to be labelled per timestep")
parser.add_argument("--rare_class", type=int, default=None, help="which class is rare")
parser.add_argument("--rare_class_percentage", type=float, default=None, help="percentage of rare class data to keep to simulate a class imbalance scenario")

# only relevent to bandit/RL type samplers
parser.add_argument('--reward_metric_name', type=str, default=None, help='only relevent for UCB bandit sampler')
parser.add_argument('--validation_percentage', type=float, default=0.2, help='percentage of validation data')
args = parser.parse_args()

# derived arguments
# TODO add other arguments
if not args.experiment_name:
    args.experiment_name = f"{args.al_sampler}"
    if args.reward_metric_name is not None:
        args.experiment_name += f"_{args.reward_metric_name}"
    args.experiment_name += "_sampler"

if args.debug:
    args.train_epochs = 1
    args.batch_size = 5
    args.al_epochs = 5
    args.al_step_percentage = (1.0/args.al_epochs)
    args.experiment_name += "_DEBUG"
    args.experiment_dir = os.path.join(EXPERIMENTS_RESULT_DIR, "DEBUG")
else:
    args.experiment_root_dir = os.path.join(EXPERIMENTS_RESULT_DIR, args.experiment_name)
    common_experiment_args_attr_ls = [
        "al_step_percentage",
        "al_epochs",
        "rare_class_percentage",
        "train_epochs",
        "retrain_model_from_scratch",
    ]
    experiment_dir = f"{args.dataset}_classification"
    for arg_name in common_experiment_args_attr_ls:
        if vars(args)[arg_name] is not None:
            experiment_dir += f"_{arg_name}={vars(args)[arg_name]}"

    args.experiment_dir = os.path.join(
            EXPERIMENTS_RESULT_DIR,
            experiment_dir,
            args.experiment_name)

if not os.path.exists(args.experiment_dir):
    os.makedirs(args.experiment_dir)

with open(os.path.join(args.experiment_dir, "experiment_args.txt"), 'w') as f:
    json.dump(vars(args), f, indent=2)


if args.al_sampler == "random":
    manager = RandomExperimentManager(args)
elif args.al_sampler == "lc":
    manager = LCExperimentManager(args)
elif args.al_sampler == 'ucb_bandit':
    manager = UCBBanditExperimentManager(args)
else:
    raise NotImplementedError("sampler not implemented")

manager.run_experiment()
