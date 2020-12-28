"""
Active Learning Environment interface
"""
import abc
import numpy as np
import tensorflow as tf

from attr import attrs, attrib

from src.al_manager import Cifar10ALManager
from src.model_manager import ClassifierModelManager
from tf_agents.specs import array_spec

from sklearn.metrics import f1_score
from typing import Callable

@attrs
class ClassiferALEnvironmentT(abc.ABC):
    al_manager: Cifar10ALManager = attrib()
    model_manager: ClassifierModelManager = attrib()

    @property
    def model(self):
        return self.model_manager.model

    @property
    def n_step(self):
        """
        current "time" step (# of datapoint labelled)
        """
        return self.al_manager.num_labelled

    def reset(self):
        self.al_manager.reset()
        self.model_manager.reset_model()

    def warm_start(self, n_to_label):
        """
        warm start of environment (initial sample that is random)
        """
        self.reset()
        unlabelled_idx, unlabelled_x = self.al_manager.unlabelled_train_data
        label_indices = np.random.choice(
            unlabelled_idx, n_to_label, replace=False)
        self.label_step(label_indices)


    def train_step(self, retrain=False):
        """
        a pseudostep to train the model, not an action
        """
        if retrain:
            self.model_manager.reset_model()

        _, x, y = self.al_manager.labelled_train_data
        self.model_manager.train_model(x, y)


    def label_step(self, indices_to_label):
        """
        a label step by the AL agent
        """
        if self.al_manager.data_is_labelled(indices_to_label):
            raise Exception(
                "invalid action, data is already labelled")
        self.al_manager.label_data(indices_to_label)

    # def evaluate_model(self, data_type: str):
    #     x, y = self.al_manager.get_dataset(data_type)
    #     model_manager = self.model_manager
    #     return model_manager.evaluate_model(x, y)

    @abc.abstractmethod
    def get_reward(self):
        """
        gets reward bassed on current state of environment.
        we do not return the reward right after a step b/c we might
        call pseudo steps such as train_step before generating reward
        """
        ...

    @abc.abstractmethod
    def get_observation(self):
        """
        gets current state of environment
        """
        # get state of environment
        # can be model/data agnostistic or not
        ...

class BaseClassiferALEnvironment(ClassiferALEnvironmentT):

    def get_reward(self):
        return None

    def get_observation(self):
        return None
