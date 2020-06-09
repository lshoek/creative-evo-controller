import tensorflow as tf
import numpy as np

class ConvModel(tf.keras.Model):

    def __init__(self):
        super(ConvModel, self).__init__()

        self.input = tf.keras.layers.Input(shape=(16, 16, 1))

        x = tf.keras.layers.Conv2D(64, kernel_size=5, strides=2, activation='relu', padding="same")(input)
        x = tf.keras.layers.MaxPooling2D((2, 2), strides=2)(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Conv2D(128, kernel_size=3, activation="relu", padding="same")(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Conv2D(256, kernel_size=3, activation="relu", padding="same")(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.Dropout(.5)(x)
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        self.output = tf.keras.layers.Dense(16, activation="softmax", kernel_regularizer=tf.keras.regularizers.l2(0.1), name="hours")(x)

        self.model = tf.keras.Model(inputs=i, outputs=[x])
        model.compile(loss=["sparse_categorical_crossentropy", "sparse_categorical_crossentropy"], optimizer="sgd", metrics=["accuracy"])
        model.summary()

    def forward(self, inputs, train=False):
        x = self.input(inputs)
        return self.input(inputs)

model = ConvModel()

