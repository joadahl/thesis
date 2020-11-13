from tensorflow import keras
from sampling import Sampling
from keras.layers import Dense, BatchNormalization, Flatten, Conv2D
import tensorflow as tf

class encoder(keras.layers.Layer):

    def __init__(self, latent_dim):
        super(encoder, self).__init__()
        self.layer_1 = Conv2D(filters=32, kernel_size=(4, 4), activation="relu",
                              strides=2, padding="same", kernel_initializer=tf.keras.initializers.HeNormal())
        self.layer_2 = Conv2D(filters=32, kernel_size=(4, 4), activation="relu",
                              strides=2, padding="same", kernel_initializer=tf.keras.initializers.HeNormal())
        self.layer_3 = Conv2D(filters=64, kernel_size=(4, 4), activation="relu",
                              strides=2, padding="same", kernel_initializer=tf.keras.initializers.HeNormal())
        self.layer_4 = Conv2D(filters=64, kernel_size=(4, 4), activation="relu",
                              strides=2, padding="same", kernel_initializer=tf.keras.initializers.HeNormal())
        self.layer_5 = Dense(units=128, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())
        self.dense_log_var = Dense(units=latent_dim, kernel_initializer=tf.keras.initializers.HeNormal())
        self.dense_mean = Dense(units=latent_dim, kernel_initializer=tf.keras.initializers.HeNormal())
        self.sampling = Sampling()
        self.batch_norm_1 = BatchNormalization()
        self.batch_norm_2 = BatchNormalization()
        self.batch_norm_3 = BatchNormalization()
        self.batch_norm_4 = BatchNormalization()
        self.flatten = Flatten()

    def call(self, inputs, training=False):
        inputs = tf.cast(inputs, dtype=tf.float32)
        z = self.layer_1(inputs)
        z = self.batch_norm_1(z)
        z = self.layer_2(z)
        z = self.batch_norm_2(z)
        z = self.layer_3(z)
        z = self.batch_norm_3(z)
        z = self.layer_4(z)
        z = self.batch_norm_4(z)
        z = self.flatten(z)
        z = self.layer_5(z)
        z_mean = self.dense_mean(z)
        z_log_var = self.dense_log_var(z)
        z = self.sampling([z_mean, z_log_var])
        return z, z_mean, z_log_var
