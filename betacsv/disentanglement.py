from vae import betavae
from data import dsprite
from callbackvae import callback
from betaclass import beta_score_classifier
from callbackbeta import callbackbeta
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
        self.path = os.getcwd() #denna har vi lagt till, måste kolla hur klustret hanterar, verkar gå bra
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
                                   epochs=ini_epoch + 10000, initial_epoch=ini_epoch, verbose=0, callbacks=[callback(self.path)])

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

    def beta_score(self, amount_n, L):
        z_diff_total = np.zeros((amount_n, self.latent_dims))
        y_total = np.zeros((amount_n, len(self.data.gen_factor_size)))
        for step, x_batch in enumerate(self.data.batch_generation_beta(L)): #self.data.batch_generation(L, "beta")
            if step == amount_n:
                break
            x_1, x_2, y = x_batch
            z_1, z_mean_1, z_log_var_1 = self.beta_vae.encoder(x_1)
            z_2, z_mean_2, z_log_var_2 = self.beta_vae.encoder(x_2)
            z_diff = z_1 - z_2
            z_batch_diff = np.mean(z_diff, axis=0)
            z_diff_total[step] = z_batch_diff
            y_total[step] = y
        y_total = y_total[:, 1:len(self.data.gen_factor_size)] #la till denna utifall att
        z_diff_train, z_diff_test = np.split(z_diff_total, [int(.7 * len(z_diff_total))])
        y_train, y_test = np.split(y_total, [int(.7 * len(y_total))])
        data_train = tf.data.Dataset.from_tensor_slices((z_diff_train, y_train)).batch(64)
        data_test = tf.data.Dataset.from_tensor_slices((z_diff_test, y_test)).batch(64)
        beta_classifier = beta_score_classifier()
        beta_classifier.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.5, beta_2=0.9,
                                                             epsilon=1e-8)) #verkar vara okej lr, 1e-8
        history = beta_classifier.fit(data_train, validation_data=data_test, verbose=0,
                                                                        epochs=50,
                                                                        callbacks=[callbackbeta(self.path)])

    """
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
    """




dis = disentanglement(10, 6, True)
dis.train_vae()
#dis.beta_score(20, 20)
#dis.plot_loss()

