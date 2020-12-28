from src.environment import (
    ClassiferALEnvironmentT,
)
from src.al_manager import Cifar10ALManager
from src.model_manager import ClassifierModelManager
from src.model.cifar10_model import Cifar10Model
import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
def get_model():
    return Cifar10Model(2)

model_manager = ClassifierModelManager(1)
al_manager = Cifar10ALManager([0,1], [0.5, 0.5], 0.1)

class TestClassifierALEnv(ClassiferALEnvironmentT):
    def get_reward(self):
        return None
    def get_observation(self):
        return None

env = TestClassifierALEnv(100, get_model, model_manager, al_manager)
env.reset()
env.label_step(0)
env.train_step()
