"""
Active Learning Manager classes manages the AL environment
"""
import abc
from typing import List
import tensorflow as tf
import numpy as np

from src.utils.utils import to_onehot


class Cifar10ALManager:
    def __init__(
            self,
            classes: List[int],
            class_ratio: List[float]=None,  # defaults to uniform split
            validation_split: float=None,
            ):
        assert len(classes) == len(class_ratio)

        self.classes = classes  # mapping from class to actual class
        self.num_classes = len(self.classes)
        self.class_ratio = class_ratio
        self.validation_split = validation_split
        self._init_dataset()

        self.pool_size = self.train_data[0].shape[0]
        self.is_labelled = np.repeat(False, self.pool_size)

    def _init_dataset(self):
        """
        constructs the train (pool) dataset
        , validation (used to compute reward if reward if relevent
        , and test environment
        """
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

        # Normalize pixel values to be between 0 and 1
        x_train, x_test = x_train / 255.0, x_test / 255.0
        y_train = y_train.flatten()
        y_test = y_test.flatten()


        def augment_dataset(x, y, classes, class_ratio):
            n_classes = len(classes)
            if class_ratio:
                # assumes equal class distribution from beginning
                class_ratio = np.array(class_ratio)/np.sum(class_ratio)
            else:
                class_ratio = np.ones(n_classes)/n_classes

            final_x, final_y = None, None
            for i, c, c_ratio in zip(np.arange(n_classes), classes, class_ratio):
                class_x = x[y==c]
                n_to_keep = int(class_x.shape[0] * c_ratio)
                # TODO shuffle?

                # instead of reusuing the class label, we start at 0,1,2...
                class_x = class_x[:n_to_keep]
                if final_x is not None:
                    final_x = np.concatenate((final_x, class_x))
                    final_y = np.concatenate((final_y, np.repeat(i, n_to_keep)))

                else:
                    final_x = class_x
                    final_y = np.repeat(i, n_to_keep)

            # 1 hot
            final_y = to_onehot(final_y, n_classes)

            final_n_points = final_y.shape[0]
            shuffle = np.random.permutation(final_n_points)
            return final_x[shuffle], final_y[shuffle]

        x_train, y_train = augment_dataset(x_train, y_train, self.classes, self.class_ratio)
        x_test, y_test = augment_dataset(x_test, y_test, self.classes, self.class_ratio)

        # build validation dataset
        if self.validation_split:
            num_validation = int(len(x_train) * self.validation_split)
            idx = np.random.choice(len(x_train), num_validation)
            mask = np.ones(len(x_train), np.bool)
            mask[idx] = 0
            x_val = x_train[~mask]
            y_val = y_train[~mask]
            x_train = x_train[mask]
            y_train = y_train[mask]


        # we keep as raw numpy as it's easier to index only the labelled set
        self.train_data = (x_train, y_train)
        self.test_data = (x_test, y_test)
        if self.validation_split:
            self.validation_data = (x_val, y_val)
        else:
            self.validation_split = None

    def reset(self):
        self.is_labelled = np.repeat(False, self.pool_size)

    def label_data(self, data_indices):
        self.is_labelled[data_indices] = True

    def data_is_labelled(self, data_indices):
        """
        returns if any data is labelled
        """
        return np.any(self.is_labelled[data_indices])

    @property
    def labelled_train_data(self):
        x, y = self.train_data
        return np.where(self.is_labelled)[0], x[self.is_labelled], y[self.is_labelled]

    @property
    def unlabelled_train_data(self):
        x, _ = self.train_data
        return np.where(~self.is_labelled)[0], x[~self.is_labelled]

    @property
    def num_labelled(self):
        return self.is_labelled[self.is_labelled].shape[0]

    @property
    def num_unlabelled(self):
        return self.is_labelled[~self.is_labelled].shape[0]

    def get_dataset(self, data_type: str):
        if data_type == "train":
            return self.train_data
        elif data_type == "test":
            return self.test_data
        elif data_type == "validation":
            data = self.validation_data
            if data is None:
                raise Exception("Validation data is not available")
            return data
