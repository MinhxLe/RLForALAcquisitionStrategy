from src.environment import (
    ClassiferALEnvironmentT,
)
from src.al_manager import Cifar10ALManager
from src.model_manager import ClassifierModelManager
from src.model.cifar10_model import Cifar10Model
from src.al_agent import (
    RandomALAgent,
    LeastConfidentALAgent,
)
import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
def get_model():
    return Cifar10Model(2)

model_manager = ClassifierModelManager(get_model, 1)
al_manager = Cifar10ALManager([0,1], [0.5, 0.5], 0.1)

class TestClassifierALEnv(ClassiferALEnvironmentT):
    def get_reward(self):
        return None
    def get_observation(self):
        return None

env = TestClassifierALEnv(al_manager, model_manager)
agent = RandomALAgent(env)

# env.warm_start(100)
# env.train_step()
# indices_to_label = agent.select_data_to_label(5)
# env.label_step(indices_to_label)



agent = LeastConfidentALAgent(env)
env.warm_start(100)
env.train_step()
indices_to_label = agent.select_data_to_label(5)
env.label_step(indices_to_label)
print(env.evaluate_model("test"))

# env.train_step()
# indices_to_label = agent.select_data_to_label(5)
# env.label_step(indices_to_label)

