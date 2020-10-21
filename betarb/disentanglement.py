from vae import betavae
from data import dsprite
from callbackvae import callbackvae
import math
import tensorflow as tf
#from callbackvae import callbackvae
from keras import Input
import tensorflow.keras as keras
import os
import numpy as np
import pickle
#from matplotlib import pyplot as plt
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
#tf.keras.backend.set_floatx('float64')

class disentanglement:
    def __init__(self, latent_dims, beta, cluster):
        self.cluster = cluster
        self.data = dsprite()
        self.latent_dims = latent_dims
        self.beta = beta
        self.path = "modelstore/"
        if not self.cluster:
            os.makedirs("modelstore/", exist_ok=True)
            os.makedirs("history/", exist_ok=True)
        #self.beta_vae = self.make_or_restore_beta_vae()
        self.beta_vae = self.make_or_restore_beta_vae()
        #self.beta_vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08))
        #self.train_or_load_beta_vae()

    def train_vae(self):
        with tf.device('/gpu:0'):
            checkpoint = callbackvae()
            #self.beta_vae._set_inputs(Input(shape=(64, 64, 64, 1), dtype=tf.uint8))
            #x_test = self.data.x[0:150].astype("float32")
            #gen = tf.data.Dataset.from_tensor_slices((x_test, x_test)).batch(64)
            #self.beta_vae(tf.random.normal((64, 64, 64, 1)))
            #optimizer_beta_vae = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
            #self.beta_vae.compile(optimizer=optimizer_beta_vae)
            print("bajs")
            history = self.beta_vae.fit(self.data.x_train_gen, validation_data=self.data.x_test_gen,
                                   epochs=5, verbose=0, callbacks=[checkpoint])
            #history = self.beta_vae.fit(gen, validation_data=gen,
            #                       epochs=5, verbose=0, callbacks=[checkpoint])
            #self.beta_vae.checkpoint = checkpoint
            #print(checkpoint.epoch_loss)
            #self.store_last_checkpoint = checkpoint #denna kommer försvinna eftersom jag skapar ett nytt objekt

            #print(history.history.keys())
            #print(history.history["val_loss"])

    def make_or_restore_beta_vae(self):
        vae = betavae(self.latent_dims, self.beta)
        optimizer_beta_vae = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        vae.compile(optimizer=optimizer_beta_vae)
        if not os.listdir("modelstore/"):
            print("Creating a new model")
            return vae
        #print("loading weights")
        print("loading model")
        #vae = keras.models.load_model('modelstore/betavae')
        #vae = tf.keras.models.load_model('modelstore/betavae.hp5')
        #x_rec = vae(tf.random.normal((64, 64, 64, 1), dtype=tf.float32))
        print("tja")
        #print(x_rec.shape)
        latest = tf.train.latest_checkpoint("modelstore/")
        vae.fit(self.data.x[:1], self.data.x[:1], epochs=1) #det här verkar fungera, den fortsätter träna
        vae.load_weights(latest)
        with open('modelstore/optimizer.pkl', 'rb') as f:
            weight_values = pickle.load(f)
        vae.optimizer.set_weights(weight_values)
        return vae

    #def train_or_load_beta_vae(self):
    #    if os.listdir("modelstore/"):
    #        latest = tf.train.latest_checkpoint("modelstore/")
    #        self.beta_vae.fit(self.data.x[:1], self.data.x[:1], epochs=1)
    #        self.beta_vae.load_weights(latest)
    #    else:
    #        self.train_vae()


dis = disentanglement(4, 4, False)
#vae = dis.make_or_restore_beta_vae()
#x_rec = vae.call(dis.data.x_train[0:50].astype("float32"))
dis.train_vae()
#print(dis.store_last_checkpoint.epoch_loss)
#print(dis.beta_vae.checkpoint.epoch_loss)
#print(x_rec.shape)
