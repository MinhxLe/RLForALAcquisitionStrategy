import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Model


class Cifar10Model(Model):
# model from https://www.tensorflow.org/tutorials/images/cnn
    def __init__(self, n_classes=10):
        super().__init__()
        # TODO move hyperparemters of model out to training script
        model = tf.keras.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(n_classes))
        self.model = model

    def call(self, images):
        return self.model(images)
