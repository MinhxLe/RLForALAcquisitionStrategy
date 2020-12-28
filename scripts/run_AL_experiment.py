import os
import json
import tensorflow as tf

from src.model.cifar10_model import Cifar10Model
from src.model.resnet_50 import ResNetModel

from src.al_manager import Cifar10ALManager
from src.environment import (
    BaseClassiferALEnvironment,
)
from src.model_manager import (
    ClassifierModelManager,
)
from src.al_agent import (
    RandomALAgent,
    LeastConfidentALAgent,
)
from src.al_session import (
    ClassiferALSessionManager,
)
from src.al_session import (
    ClassiferALSessionManager,
)
from src.utils.class_utils import get_class_kwargs

# don't just consume all the memory
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

ARGS = dict(
    # al manager/data manager
    classes=[0, 1],
    class_ratio=[0.9, 0.1],
    validation_split=0.1,

    # ml model manager
    batch_size=32,
    n_train_epochs=10,

    # al_agent
    agent_type="random",

    # session manager
    al_epochs=30,
    al_step_percentage=0.01,
    warm_start_percentage=0.01,
    retrain_model=True,

    random_seed=42,  # TODO use this
    n_session_runs=5,

    # session dir log args
    save_model_interval=10,
    stdout=True,

    # shared parameters
    is_debug=False,
)
# some derived arguments
ARGS["n_classes"] = len(ARGS["classes"])

session_args = []
args_for_name=[
    "classes",
    "class_ratio",
    "al_epochs",
    "al_step_percentage",
    "warm_start_percentage",
    "retrain_model",
    "n_train_epochs"
]
for arg in args_for_name:
    session_args.append(f"{arg}={ARGS[arg]}")

ARGS["session_dir"] = os.path.join(
    "results", "Cifar10", "_".join(session_args), ARGS["agent_type"])


def get_model():
    return Cifar10Model(
        **get_class_kwargs(Cifar10Model, ARGS))

al_manager = Cifar10ALManager(
        **get_class_kwargs(Cifar10ALManager, ARGS))
model_manager = ClassifierModelManager(
        get_model_fn=get_model,
        **get_class_kwargs(ClassifierModelManager, ARGS))
al_env = BaseClassiferALEnvironment(
        al_manager,
        model_manager,
        )
if ARGS["agent_type"] == "random":
    al_agent = RandomALAgent(
        env=al_env, **get_class_kwargs(RandomALAgent, ARGS))
elif ARGS["agent_type"] == "LC":
    al_agent = LeastConfidentALAgent(
        env=al_env, **get_class_kwargs(RandomALAgent, ARGS))
else:
    raise Exception("agent not implemented")
al_session = ClassiferALSessionManager(
    al_env=al_env,
    al_agent=al_agent,
    al_manager=al_manager,
    **get_class_kwargs(ClassiferALSessionManager, ARGS)
)

if not os.path.exists(ARGS["session_dir"]):
    os.makedirs(ARGS["session_dir"])
with open(os.path.join(ARGS["session_dir"], "session_args.txt"), 'w') as f:
    json.dump(ARGS, f, indent=2)

for i in range(ARGS["n_session_runs"]):
    al_session.reset_session()
    al_session.run_session()
