from vae import betavae
from data import dsprite
from callbackvae import callback
from matplotlib import pyplot as plt
import tensorflow as tf
import os
import numpy as np
import pickle
import csv
import math

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
#tf.keras.backend.set_floatx('float64')

class disentanglement:
    def __init__(self, latent_dims, beta, cluster):
        self.cluster = cluster
        self.data = dsprite()
        self.latent_dims = latent_dims
        self.beta = beta
        self.path = os.getcwd() #denna har vi lagt till, måste kolla hur klustret hanterar
        if not self.cluster:
            self.path = "betavae" + str(self.latent_dims) + str(self.beta)
            os.makedirs(self.path, exist_ok=True)
            os.makedirs(self.path + "/modelstore", exist_ok=True) #lägg till en till folder för sparandet
            os.makedirs(self.path + "/history", exist_ok=True)
        self.beta_vae = self.make_or_restore_beta_vae()


    def train_vae(self):
        ini_epoch = 0
        if os.path.isfile(self.path + "/history/loss.csv"):
            ini_epoch = int(np.genfromtxt(self.path + "/history/loss.csv", delimiter=',')[-1][0])
        with tf.device('/gpu:0'):
            history = self.beta_vae.fit(self.data.x_train_gen, validation_data=self.data.x_test_gen,
                                   epochs=ini_epoch + 50, initial_epoch=ini_epoch, verbose=0, callbacks=[callback(self.path)])

    def make_or_restore_beta_vae(self):
        vae = betavae(self.latent_dims, self.beta)
        optimizer_beta_vae = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        vae.compile(optimizer=optimizer_beta_vae)
        if not os.listdir(self.path + "/modelstore"):
            print("Creating a new model")
            return vae
        print("loading weights")
        latest = tf.train.latest_checkpoint(self.path + "/modelstore")
        vae.fit(self.data.x[:1], self.data.x[:1], epochs=1) #det här verkar fungera, den fortsätter träna
        vae.load_weights(latest)
        print("loading optimizer")
        with open(self.path + '/modelstore/optimizer.pkl', 'rb') as f:
            weight_values = pickle.load(f)
        vae.optimizer.set_weights(weight_values)
        return vae

    def plot_loss(self):
        x = []
        y = []
        with open('history/loss.csv', 'r') as csvfile:
            plots = csv.reader(csvfile, delimiter=',')
            for row in plots:
                #print(row[0])
                #print(row[1])
                x.append(int(row[0]))
                y.append(float(row[1]))

        plt.plot(x, y, label='Loaded from file!')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Interesting Graph\nCheck it out')
        plt.legend()
        plt.show()




dis = disentanglement(4, 4, True)
dis.train_vae()
#dis.plot_loss()

