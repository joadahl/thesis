from tensorflow import keras
from keras.layers import Dense, LeakyReLU, BatchNormalization
import tensorflow as tf
from factor.data import dsprite
from factor.factorencoder import encoder

class discriminator(keras.layers.Layer):

    def __init__(self):
        super(discriminator, self).__init__()
        self.layer_1 = Dense(units=1000, activation=LeakyReLU(alpha=0.2)) #denna är ej klar här
        self.layer_2 = Dense(units=1000, activation=LeakyReLU(alpha=0.2))
        self.layer_3 = Dense(units=1000, activation=LeakyReLU(alpha=0.2))
        self.layer_4 = Dense(units=1000, activation=LeakyReLU(alpha=0.2))
        self.layer_5 = Dense(units=1000, activation=LeakyReLU(alpha=0.2))
        self.layer_6 = Dense(units=2)
        self.batch_norm_1 = BatchNormalization()
        self.batch_norm_2 = BatchNormalization()
        self.batch_norm_3 = BatchNormalization()
        self.batch_norm_4 = BatchNormalization()
        self.batch_norm_5 = BatchNormalization()

    def call(self, inputs):
        x = self.layer_1(inputs)
        x = self.batch_norm_1(x)
        x = self.layer_2(x)
        x = self.batch_norm_2(x)
        x = self.layer_3(x)
        x = self.batch_norm_3(x)
        x = self.layer_4(x)
        x = self.batch_norm_4(x)
        x = self.layer_5(x)
        x = self.batch_norm_5(x)
        x_logits = self.layer_6(x)
        x_probs = tf.keras.activations.softmax(x_logits)
        return x_logits, x_probs


#data = dsprite()
#enc = encoder(10)
#data_test = data.x_train[0:50].astype("float32")
#z, z_mean, z_log_var = enc(data_test)
#print(z.shape)
#mlp = discriminator()
#x_rec = mlp(z)
#print(x_rec.shape)


