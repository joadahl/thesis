from tensorflow import keras
from factor.factorencoder import encoder
from factor.factordecoder import decoder
from factor.discriminator import discriminator
import tensorflow as tf
import numpy as np
from factor.data import dsprite

class vae(keras.Model):
    def __init__(self, latent_dim):
        super(vae, self).__init__()
        self.encoder = encoder(latent_dim)
        self.decoder = decoder()
        self.latent_dim = latent_dim



    def call(self, inputs):
        inputs = tf.cast(inputs, dtype=tf.float32)
        z, z_mean, z_log_var = self.encoder(inputs)
        x_rec = self.decoder(z)
        return x_rec
