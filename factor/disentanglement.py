from tensorflow import keras
from factor.factorvae import factorvae
from factor.data import dsprite
import tensorflow as tf


class disentanglement:
    def __init__(self, latent_dim):
        self.factorvae = factorvae(latent_dim)


dis = disentanglement(10)
data = dsprite()
data_test = data.x[0:150]
data_val = data.x[150:300]
opt1 = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
opt2 = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9, epsilon=1e-08)
dis.factorvae.compile(optimizer=(opt1, opt2))
dis.factorvae.fit(data_test, data_test, validation_data =(data_val, data_val), batch_size = 50, epochs = 25)
