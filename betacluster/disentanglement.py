from data import dsprite
from vae import betavae
from classifier import classifier
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import math
from callbackvae import callbackvae
from callbackclass import callbackclass
#import os


class disentanglement:
    def __init__(self, data, latent_dims):
        self.latent_dims = latent_dims
        self.vae = betavae(latent_dims)
        self.classifier = classifier()
        self.optimizer_vae = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        self.optimizer_classifier = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9, epsilon=1e-08)
        self.batch_size_vae = 64
        self.epochs_vae = math.ceil(data.x_train.shape[0] / self.batch_size_vae)
        self.batch_size_classifier = 10
        #self.path = os.path.dirname(__file__)


    def generate_z_diff(self, data, amount_n, L):
        z_diff_total = np.zeros((amount_n, self.latent_dims))
        y_total = np.zeros((amount_n, 6))
        for step, x_batch in enumerate(data.batch_generation(L)):
            if step == amount_n:
                break
            x_1, x_2, y = x_batch
            z_1, z_mean_1, z_log_var_1 = self.vae.encoder(x_1)
            z_2, z_mean_2, z_log_var_2 = self.vae.encoder(x_2)
            z_diff = z_1 - z_2
            z_batch_diff = np.mean(z_diff, axis=0)
            z_diff_total[step] = z_batch_diff
            y_total[step] = y
        z_diff_train, z_diff_test, y_train, y_test = train_test_split(z_diff_total, y_total, test_size=0.33, random_state=42)
        return z_diff_train, z_diff_test, y_train, y_test

    def train_vae(self, data):
        self.vae.compile(optimizer=self.optimizer_vae)
        history = self.vae.fit(data.x_train, data.x_train, validation_data=(data.x_test, data.x_test),
                               batch_size=self.batch_size_vae,
                               epochs=self.epochs_vae, verbose=0, callbacks=[callbackvae()])
        #self.vae.save_weights(self.path)
        #file_path = os.path.join(self.path + "/modelstore/", str(10000))
        self.vae.save_weights("/Midgard/home/joadahl/thesis/betacluster/modelstore/" + str(1500))


    # "Midgard/home/joadahl/Workplace/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"

    def train_classifier(self, data, amount_n, L):
        z_diff_train, z_diff_test, y_train, y_test = self.generate_z_diff(data, amount_n, L)
        batch_size_classifier = 10
        epochs_classifier = math.ceil(z_diff_train.shape[0] / batch_size_classifier)
        self.classifier.compile(optimizer='adam')
        history = self.classifier.fit(z_diff_train, y_train, validation_data=(z_diff_test, y_test), verbose=0,
                                      batch_size=batch_size_classifier,
                                      epochs=epochs_classifier, callbacks=[callbackclass()]) #en separat callback för dem båda är nog rimligt
        classifier.save_weights()


data = dsprite()
dis = disentanglement(data, 10)
dis.train_vae(data)
