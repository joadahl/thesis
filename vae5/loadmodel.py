from data import dsprite
import os
import tensorflow as tf
from betavae import betavae
import numpy as np
from cbbetavae import cbbeta
import pandas as pd
from cbfactorvae import cbfactor
from factorvae import factorvae

class loadmodel:
    def __init__(self, latent_dims, model):
        self.data = dsprite()
        self.model = model
        self.latent_dims = latent_dims
        self.path = self.model + str(self.latent_dims)
        os.makedirs(self.path, exist_ok=True)
        os.makedirs(self.path + "/modelstore", exist_ok=True) #lägg till en till folder för sparandet
        os.makedirs(self.path + "/history", exist_ok=True)
        if not os.path.exists(self.path + "/history/scores.csv"):
            pd.DataFrame(columns=["betascore", "factorscore"]).to_csv(self.path + "/history/scores.csv", sep=",", index=False)
        if self.model == "betavae":
            self.network = self.make_or_restore_beta_vae()
            self.callback = cbbeta(self.path)
        if self.model == "factorvae":
            self.network = self.make_or_restore_factor_vae()
            self.callback = cbfactor(self.path)


    def make_or_restore_beta_vae(self):
        network = betavae(self.latent_dims)
        optimizer_beta_vae = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9,
                                                        beta_2=0.999, epsilon=1e-08)
        network.compile(optimizer=optimizer_beta_vae)
        if not os.listdir(self.path + "/modelstore"):
            print("Creating a new model")
            return network
        print("loading weights")
        latest = tf.train.latest_checkpoint(self.path + "/modelstore")
        network.load_weights(latest)
        return network

    def make_or_restore_factor_vae(self):
        network = factorvae(self.latent_dims)
        optimizer_factor_vae = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        optimizer_discriminator = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9, epsilon=1e-08)
        network.compile(optimizer_factor_vae, optimizer_discriminator)
        if not os.listdir(self.path + "/modelstore/"):
            print("Creating a new model")
            return network
        print("loading model")
        latest = tf.train.latest_checkpoint(self.path + "/modelstore")
        network.load_weights(latest)
        return network

    def train_vae(self, epochs_n):
        ini_epoch = 0
        if os.path.isfile(self.path + "/history/lossvae.csv"):
            ini_epoch = int(np.genfromtxt(self.path + "/history/lossvae.csv", delimiter=',')[-1][0])
        with tf.device('/gpu:0'):
            history = self.network.fit(self.data.x_train_gen, validation_data=self.data.x_test_gen,
                                          verbose=0, epochs=ini_epoch + epochs_n, initial_epoch=ini_epoch, callbacks=[self.callback])





t = loadmodel(4, "factorvae") #lower lr for this one
t.train_vae(100)
#t.latent_traversal_2()
#t.train_vae()
#t.train_vae()
#t.latent_traversal(1)
#t.plot_reconstruction_2()

#t.beta_score(15000, 64)
#t.factor_score(15000, 64)
#t.DCI_score(15000, "gradient")