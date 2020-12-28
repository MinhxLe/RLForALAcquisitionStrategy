import abc
import numpy as np
import tensorflow as tf
from src.environment import ClassiferALEnvironmentT
from src.utils.utils import (
    batch_sample_indices,
)
from src.utils.fixed_heap import (
    FixedHeap,
)
from typing import List


class ClassifierALAgentT(abc.ABC):
    def __init__(self, env: ClassiferALEnvironmentT):
        self.al_environment = env

    @abc.abstractmethod
    def select_data_to_label(
            self,
            n_labels: int,
            ) -> List[int]:
        """
        choose up to n points to label
        """
        ...

class RandomALAgent(ClassifierALAgentT):
    def select_data_to_label(
            self,
            n_labels: int,
            ) -> List[int]:
        """
        choose up to n points to label
        """
        al_manager = self.al_environment.al_manager
        n_to_label = min(n_labels, al_manager.num_unlabelled)

        unlabelled_idx, _ = al_manager.unlabelled_train_data
        return np.random.choice(
            unlabelled_idx, n_to_label, replace=False)


class LeastConfidentALAgent(ClassifierALAgentT):
    def select_data_to_label(
            self,
            n_labels: int,
            ) -> List[int]:
        """
        choose up to n points to label
        """
        model = self.al_environment.model
        al_manager = self.al_environment.al_manager
        n_to_label = min(n_labels, al_manager.num_unlabelled)
        heap = FixedHeap(key=lambda x : x[0])

        unlabelled_indices, unlabelled_x = (
            al_manager.unlabelled_train_data)

        # we need to keep the original indices b/c that is the actual action we are taking
        for batch_indices in batch_sample_indices(unlabelled_indices.shape[0], shuffle=False):
            batch_original_indices = unlabelled_indices[batch_indices]
            batch_x = unlabelled_x[batch_indices]
            prediction = model(batch_x, training=False)

            # we get absolute value of prediction logit which is how confident
            # confidences = tf.math.abs(prediction)
            # multiclassifier confidence
            prediction = tf.nn.softmax(prediction)  # normalize to softmax
            most_confident_prediction = tf.math.reduce_max(prediction, axis=1)
            confidences = tf.math.abs(most_confident_prediction - 0.5)

            for confidence, index in zip(confidences, batch_original_indices):
                if len(heap) < n_to_label:
                    # push - confidnece since we want to pop most confident
                    heap.push((-confidence, index))
                else:
                    top_confidence, _ = heap.top()
                    if confidence < -top_confidence:
                        heap.pop()
                        heap.push((-confidence, index))
        label_selection = []
        while len(heap) > 0:
            _, idx = heap.pop()
            label_selection.append(idx)
        del heap
        return np.array(label_selection)
