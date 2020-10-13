from vae import betavae
from data import dsprite
import math
import tensorflow as tf
from callbackvae import callbackvae
import os
from betascoreclassifier import beta_score_classifier
import numpy as np
from sklearn.model_selection import train_test_split
from callbackbetascoreclass import callbackbetascoreclass
from keras.callbacks import ModelCheckpoint
import pickle
import csv

class disentanglement:
    def __init__(self, latent_dims, beta):
        self.data = dsprite()
        self.latent_dims = latent_dims
        self.vae = betavae(latent_dims, beta)
        self.optimizer_vae = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        self.optimizer_beta_score_classifier = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9,
                                                             epsilon=1e-08)
        self.batch_size_vae = 64
        self.epochs_vae = math.ceil(self.data.x_train.shape[0] / self.batch_size_vae)
        self.path = os.path.dirname(__file__) #ta bort om kluster
        #self.beta_score_classifier = beta_score_classifier()
        self.checkpoint_vae = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(self.path + "/modelstore/", "vae" + str(self.latent_dims)),
        monitor="val_loss",
        verbose=0,
        save_best_only=False,
        save_weights_only=True,
        mode="min",
        save_freq="epoch")

    def train_vae(self):
        self.vae.compile(optimizer=self.optimizer_vae)
        history = self.vae.fit(self.data.x_train, self.data.x_train, validation_data=(self.data.x_test, self.data.x_test),
                               batch_size=self.batch_size_vae,
                               epochs=self.epochs_vae, verbose=0, callbacks=[callbackvae(), self.checkpoint_vae])
        os.makedirs(self.path + "/history/", exist_ok=True)
        np.save(os.path.join(self.path + "/history/", "history" + "vae" + str(self.latent_dims)), history.history)

    def beta_score(self, amount_n, L):
        z_diff_total = np.zeros((amount_n, self.latent_dims))
        y_total = np.zeros((amount_n, 6))
        beta_score_list = []
        for step, x_batch in enumerate(self.data.batch_generation(L)):
            if step == amount_n:
                break
            x_1, x_2, y = x_batch
            z_1, z_mean_1, z_log_var_1 = self.vae.encoder(x_1)
            z_2, z_mean_2, z_log_var_2 = self.vae.encoder(x_2)
            z_diff = z_1 - z_2
            z_batch_diff = np.mean(z_diff, axis=0)
            z_diff_total[step] = z_batch_diff
            y_total[step] = y
        z_diff_train, z_diff_test, y_train, y_test = train_test_split(z_diff_total, y_total, test_size=0.33,
                                                                      random_state=42)
        for lin_class in range(10):
            beta_classifier = beta_score_classifier()
            beta_classifier.compile(optimizer=self.optimizer_beta_score_classifier)
            history = beta_classifier.fit(z_diff_train, y_train, validation_data=(z_diff_test, y_test), verbose=0,
                                          batch_size=64,
                                          epochs=math.ceil(z_diff_train.shape[0] / 64), callbacks=[callbackbetascoreclass()]) #måste ändra på batch_size och epochs innan cluster
            accuracy = tf.keras.metrics.CategoricalAccuracy(name="acc")
            reconstruction = beta_classifier.call(z_diff_test)
            score = accuracy(y_test, reconstruction)
            beta_score_list.append(score.numpy())
        beta_score_list = np.array(beta_score_list)
        #print(beta_score_list)
        tot_score = np.mean(beta_score_list[np.argsort(np.array(beta_score_list))[-5:]])
        with open('beta_score.csv', 'a') as f:
            f.write("beta_score:" + str(tot_score))



dis = disentanglement(10, 4)
dis.train_vae()
dis.beta_score(5000, 10)