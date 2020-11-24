import json
import argparse
import os
from src.constants import (
    EXPERIMENTS_RESULT_DIR,
)
from src.experiment.mnist_experiment_manager import (
    ALRandomExperimentManager,
)
from datetime import datetime

parser = argparse.ArgumentParser()
# model args
parser.add_argument('--model_num_filters', type=int, default=8)
parser.add_argument('--model_filter_size', type=int, default=3)
parser.add_argument('--model_pool_size', type=int, default=2)

# AL args
# TODO add initial seed data percentage
parser.add_argument('--al_sampler', type=str)
parser.add_argument('--al_epochs', type=int, default=50)

# train args
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--train_epochs', type=int, default=1, help="number of train epochs per al step")
# parser.add_argument('--learning_rate', type=int, default=0.01)

# experiment args
parser.add_argument('--experiment_name', type=str, default="", help="experiment id")
parser.add_argument('--train_log_interval', type=int, default=0, help="log interval for internal train interval(<0 means not log)")
parser.add_argument('--debug', type=bool, default=False, help="debug mode")
parser.add_argument("--save_model_interval", type=int, default=10, help="save model frequency")
parser.add_argument("--stdout", type=bool, default=True, help="print logs to stdout")
parser.add_argument("--seed", type=int, default=42, help="random seed")
args = parser.parse_args()

# derived arguments
timestamp_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
# TODO add other arguments
if not args.experiment_name:
    args.experiment_name = f"mnist_{args.al_sampler}_sampler_{timestamp_str}"

if args.debug:
    args.experiment_name += "_DEBUG"
    args.train_epochs = 1
    args.batch_size = 5
    args.al_epochs = 5

args.experiment_dir = os.path.join(EXPERIMENTS_RESULT_DIR, args.experiment_name)

if args.al_sampler == "random":
    manager = ALRandomExperimentManager(args)
else:
    raise NotImplementedError("sampler not implemented")

with open(os.path.join(args.experiment_dir, "experiment_args.txt"), 'w') as f:
    json.dump(args.__dict__, f, indent=2)
manager.run_experiment()


