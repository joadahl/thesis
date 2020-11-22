from tensorflow import keras
from keras.layers import Dense, Reshape, Conv2DTranspose, BatchNormalization
import tensorflow as tf


class decoder(keras.layers.Layer):

    def __init__(self):
        super(decoder, self).__init__()
        self.layer_1 = Dense(units=128, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())
        self.layer_2 = Dense(units=1024, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())
        self.layer_3 = Conv2DTranspose(filters=64, kernel_size=(4, 4), activation='relu',
                                       strides=2, padding='same', kernel_initializer=tf.keras.initializers.HeNormal())
        self.layer_4 = Conv2DTranspose(filters=32, kernel_size=(4, 4), activation='relu',
                                       strides=2, padding='same', kernel_initializer=tf.keras.initializers.HeNormal())
        self.layer_5 = Conv2DTranspose(filters=32, kernel_size=(4, 4), activation='relu',
                                       strides=2, padding='same', kernel_initializer=tf.keras.initializers.HeNormal())
        self.layer_6 = Conv2DTranspose(filters=1, kernel_size=(4, 4), activation='sigmoid',
                                       strides=2, padding='same', kernel_initializer=tf.keras.initializers.HeNormal())
        self.reshape = Reshape((4, 4, 64))
        self.batch_norm_1 = BatchNormalization()
        self.batch_norm_2 = BatchNormalization()
        self.batch_norm_3 = BatchNormalization()
        self.batch_norm_4 = BatchNormalization()
        self.batch_norm_5 = BatchNormalization()
        self.batch_norm_6 = BatchNormalization()

    def call(self, inputs, training=False):
        inputs = tf.cast(inputs, dtype=tf.float64)
        x_rec = self.batch_norm_1(inputs) #tillagd
        x_rec = self.layer_1(x_rec) #här stod inputs
        x_rec = self.batch_norm_2(x_rec)
        x_rec = self.layer_2(x_rec)
        x_rec = self.batch_norm_3(x_rec)
        x_rec = self.reshape(x_rec)
        x_rec = self.layer_3(x_rec)
        x_rec = self.batch_norm_4(x_rec)
        x_rec = self.layer_4(x_rec)
        x_rec = self.batch_norm_5(x_rec)
        x_rec = self.layer_5(x_rec)
        x_rec = self.batch_norm_6(x_rec)
        x_rec = self.layer_6(x_rec)
        return x_rec

