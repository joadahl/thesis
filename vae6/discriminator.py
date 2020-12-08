from tensorflow import keras
from keras.layers import Dense, LeakyReLU, BatchNormalization
import tensorflow as tf

class discriminator(keras.Model): #keras.layers.Layer, har ändrat

    def __init__(self, seed):
        super(discriminator, self).__init__()
        self.layer_1 = Dense(units=1000, kernel_initializer=tf.keras.initializers.HeNormal(seed), activation=LeakyReLU(alpha=0.2)) #denna är ej klar här
        self.layer_2 = Dense(units=1000, kernel_initializer=tf.keras.initializers.HeNormal(seed), activation=LeakyReLU(alpha=0.2))
        self.layer_3 = Dense(units=1000, kernel_initializer=tf.keras.initializers.HeNormal(seed), activation=LeakyReLU(alpha=0.2))
        self.layer_4 = Dense(units=1000, kernel_initializer=tf.keras.initializers.HeNormal(seed), activation=LeakyReLU(alpha=0.2))
        self.layer_5 = Dense(units=1000, kernel_initializer=tf.keras.initializers.HeNormal(seed), activation=LeakyReLU(alpha=0.2))
        self.layer_6 = Dense(units=2, kernel_initializer=tf.keras.initializers.HeNormal(seed), activation='softmax')
        self.batch_norm_1 = BatchNormalization()
        self.batch_norm_2 = BatchNormalization()
        self.batch_norm_3 = BatchNormalization()
        self.batch_norm_4 = BatchNormalization()
        self.batch_norm_5 = BatchNormalization()
        self.batch_norm_6 = BatchNormalization()

    def call(self, inputs):
        inputs = tf.cast(inputs, dtype=tf.float64)
        x = self.batch_norm_1(inputs)
        x = self.layer_1(x)
        x = self.batch_norm_2(x)
        x = self.layer_2(x)
        x = self.batch_norm_3(x)
        x = self.layer_3(x)
        x = self.batch_norm_4(x)
        x = self.layer_4(x)
        x = self.batch_norm_5(x)
        x = self.layer_5(x)
        x = self.batch_norm_6(x)
        x_probs = self.layer_6(x)
        return x_probs



