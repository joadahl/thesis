from keras import layers
import tensorflow as tf
import keras.backend as K


class Sampling(layers.Layer):


    def call(self, inputs, training=False):
        z_mean = inputs[0]
        z_log_var = inputs[1]
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim), mean=0., stddev=1.)
        return z_mean + tf.exp(z_log_var / 2) * epsilon
