import numpy as np
import tensorflow as tf


def batch_sample_slices(tensor_slices, batch_size=32, shuffle=True):
    # TODO DO NOT USE for tuples of Tensors
    dataset = tf.data.Dataset.from_tensor_slices(tensor_slices)
    if shuffle:
        dataset = dataset.shuffle(len(tensor_slices), reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size)
    for batch in dataset.as_numpy_iterator():
        yield batch


def batch_sample_indices(n_elements, batch_size=32, shuffle=True):
    dataset = tf.data.Dataset.range(n_elements)
    if shuffle:
        dataset = dataset.shuffle(n_elements, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size)
    for batch in dataset.as_numpy_iterator():
        yield batch


def to_onehot(y, n_classes):
    return np.eye(n_classes)[y]
