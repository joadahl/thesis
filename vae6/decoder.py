from tensorflow import keras
from keras.layers import Dense, Reshape, Conv2DTranspose, BatchNormalization
import tensorflow as tf
import numpy as np

#from data import dsprite
#from encoder import encoder
#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)


class decoder(keras.layers.Layer):

    def __init__(self, seed):
        super(decoder, self).__init__()
        np.random.seed(seed)
        self.layer_1 = Dense(units=128, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal(seed))
        self.layer_2 = Dense(units=1024, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal(seed))
        self.layer_3 = Conv2DTranspose(filters=64, kernel_size=(4, 4), activation='relu',
                                       strides=2, padding='same', kernel_initializer=tf.keras.initializers.HeNormal(seed))
        self.layer_4 = Conv2DTranspose(filters=32, kernel_size=(4, 4), activation='relu',
                                       strides=2, padding='same', kernel_initializer=tf.keras.initializers.HeNormal(seed))
        self.layer_5 = Conv2DTranspose(filters=32, kernel_size=(4, 4), activation='relu',
                                       strides=2, padding='same', kernel_initializer=tf.keras.initializers.HeNormal(seed))
        self.layer_6 = Conv2DTranspose(filters=1, kernel_size=(4, 4), activation='sigmoid',
                                       strides=2, padding='same', kernel_initializer=tf.keras.initializers.HeNormal(seed))
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

#data = dsprite(0)
#enc = encoder(5, 0)
#dec = decoder(0)
#z, z_mean, z_log_var = enc(data.x_train[0:5])
#x_rec = dec(z)
#print(x_rec)

