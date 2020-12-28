"""
f.learning model manager that trains and evaluates model
"""
import numpy as np
import tensorflow as tf

from src.utils.utils import (
    batch_sample_indices,
)
from typing import Callable


class ClassifierModelManager():
    def __init__(self,
            get_model_fn: Callable,
            n_train_epochs: int,
            batch_size: int=32,
            is_debug: bool=False,
            ):
        self.get_model_fn = get_model_fn

        self.n_train_epochs = n_train_epochs
        self.batch_size= batch_size
        self.is_debug = is_debug,
        self.optimizer = self._get_optimizer()
        self.loss_fn = self._get_loss()
        self.model = get_model_fn()

    def reset_model(self, clear_backend=False):
        if clear_backend:
            tf.keras.backend.clear_session()
        del self.model
        self.model = self.get_model_fn()


    def _get_optimizer(self):
        # TODO add optimizer args
        return tf.keras.optimizers.Adam()

    def _get_loss(self):
        # TODO weight a class loss
        return tf.keras.losses.CategoricalCrossentropy(
            from_logits=True,
            reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)

    def train_model(
            self,
            train_x,
            train_y,
            n_train_epochs=None,
            batch_size=None) -> None:
        """
        trains model for n epochs
        """
        model = self.model
        if n_train_epochs is None:
            n_train_epochs = self.n_train_epochs
        if batch_size is None:
            batch_size = self.batch_size

        optimizer = self.optimizer
        loss_fn = self.loss_fn
        data_size = train_x.shape[0]
        for i in range(n_train_epochs):
            for idx, batch in enumerate(
                    batch_sample_indices(data_size, batch_size=batch_size)):
                batch_x, batch_y = train_x[batch], train_y[batch]
                with tf.GradientTape() as tape:
                    prediction = model(batch_x)
                    loss = loss_fn(batch_y, prediction)
                    grads = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(grads, model.trainable_variables))
        # TODO debug logging

    def evaluate_model(
            self,
            test_x,
            test_y,
            batch_size=None) -> dict:
        """
        generator for evaluating model and input, prediction, and true label
        """
        model = self.model
        if batch_size is None:
            batch_size = self.batch_size
        data_size = test_x.shape[0]
        for idx, test_batch in enumerate(
                batch_sample_indices(data_size, batch_size=batch_size)):
            batch_x, batch_y = test_x[test_batch], test_y[test_batch]
            raw_prediction = model(batch_x, training=False)
            batch_loss = self.loss_fn(batch_y, raw_prediction)

            yield batch_x, batch_y, raw_prediction, batch_loss

    def save_model(self, model_fpath: str):
        self.model.save_weights(model_fpath)
