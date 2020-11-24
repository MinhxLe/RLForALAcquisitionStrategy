import tensorflow as tf


def batch_sample_data(data, batch_size=32, shuffle=True):
    # TODO DO NOT USE, this is broken
    dataset = tf.data.Dataset.from_tensors(data)
    if shuffle:
        dataset = dataset.shuffle(len(data), reshuffle_each_iteration=True)
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

