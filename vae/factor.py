from tensorflow import keras
from encoder import encoder
from decoder import decoder
import tensorflow as tf

class factor(keras.Model):
    def __init__(self, latent_dims):
        super(factor, self).__init__()
        self.latent_dims = latent_dims
        self.encoder = encoder(latent_dims)
        self.decoder = decoder()

    def call(self, inputs, training=False):
        inputs = tf.cast(inputs, dtype=tf.float64)
        z, z_mean, z_log_var = self.encoder(inputs)
        x_rec = self.decoder(z)
        return x_rec