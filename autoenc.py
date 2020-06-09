import tensorflow as tf
import numpy as np

class SketchAutoEncoder(tf.keras.Model):
    def __init__(self):
        super(SketchAutoEncoder, self).__init__()
        
        self.latent_dim = 8

        self.encoder = tf.keras.models.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(16, 16, 1)),
                tf.keras.layers.Conv2D(
                    16, (3, 3), activation='relu', padding='same'),
                tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
                tf.keras.layers.Conv2D(
                    8, (3, 3), activation='relu', padding='same'),
                tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
                tf.keras.layers.Conv2D(
                    8, (3, 3), activation='relu', padding='same'),
                tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(
                    self.latent_dim + self.latent_dim, activation='tanh'),
            ]
        )

        # self.decoder = tf.keras.Sequential(
        #     [
        #         tf.keras.layers.InputLayer(input_shape=(16, 16, 1)),
        #     ]
        # )

    def summary(self):
        self.autoencoder.summary()
    
    def train(self, x):
        print('unimplemented')
        #self.autoencoder.compile(optimizer='adadelta', loss=['binary_crossentropy'])

    def encode(self, x):
        x = np.reshape(x, (1, 16, 16, 1))
        return self.encoder(x)

    def decode(self, x):
        print('unimplemented')
