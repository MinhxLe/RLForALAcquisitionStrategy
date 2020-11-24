import tensorflow as tf


class MNISTModel(tf.keras.Model):
    def __init__(self,
            num_filters=8,
            filter_size=3,
            pool_size=2,
            ):
        super().__init__()
        # TODO move hyperparemters of model out to training script
        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(num_filters, filter_size, input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D(pool_size=pool_size),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1),
            ]
        )

    def call(self, images):
        return self.model(images)
