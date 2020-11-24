from data import dsprite
import os
import tensorflow as tf
from betavae import betavae
import numpy as np
from cbbetavae import cbbeta
import pandas as pd
from cbfactorvae import cbfactor
from factorvae import factorvae
from betaclass import beta_score_classifier
from cbbetascore import cbbetascore
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import normalize
from scipy.stats import entropy
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler

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
        network.fit(self.data.x[:1], self.data.x[:1], epochs=1)
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
        network.fit(self.data.x[:1], self.data.x[:1], epochs=1)
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

    def beta_score(self, amount_n, L):
        z_diff_total = np.zeros((amount_n, self.latent_dims))
        y_total = np.zeros((amount_n, len(self.data.gen_factor_size)))
        for step, x_batch in enumerate(self.data.batch_generation_beta(L)): #self.data.batch_generation(L, "beta")
            if step == amount_n:
                break
            x_1, x_2, y = x_batch
            z_1, z_mean_1, z_log_var_1 = self.network.vae.encoder(x_1)
            z_2, z_mean_2, z_log_var_2 = self.network.vae.encoder(x_2)
            z_diff = np.abs(z_1 - z_2)
            z_batch_diff = np.mean(z_diff, axis=0)
            z_diff_total[step] = z_batch_diff
            y_total[step] = y
        y_total = y_total[:, 1:len(self.data.gen_factor_size)] #la till denna utifall att
        z_diff_train, z_diff_test = np.split(z_diff_total, [int(.7 * len(z_diff_total))])
        y_train, y_test = np.split(y_total, [int(.7 * len(y_total))])

        data_train = tf.data.Dataset.from_tensor_slices((z_diff_train, y_train)).batch(64)
        data_test = tf.data.Dataset.from_tensor_slices((z_diff_test, y_test)).batch(64)
        beta_classifier = beta_score_classifier()
        beta_classifier.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999,
                                                             epsilon=1e-8)) #verkar vara okej lr, 1e-8
        history = beta_classifier.fit(data_train, validation_data=data_test, verbose=0,
                                                                        epochs=1000,
                                                                        callbacks=[cbbetascore(self.path)])

    def factor_score(self, amount_n, L):
        #vi testar göra one-hot encodings här med, vi märker om det är skit
        delta = 0.05
        size_n = 500
        subset_n_std = 10000  # value from challenges
        iterations_n = int(subset_n_std / size_n)
        rand_idx = np.random.randint(low=self.data.x_train.shape[0])
        x_subset = self.data.x[rand_idx:subset_n_std + rand_idx]  # lägg till så att vi prunar bort std < 0.05
        z_var = np.zeros((self.latent_dims,))
        for l in range(iterations_n):
            z, z_mean, z_log_var = self.network.vae.encoder(x_subset[l * size_n:(l + 1) * size_n])
            z_sub_var = np.var(z, axis=0)  # this should be ginis
            # print(np.sqrt(z_sub_var))
            # print("hej")
            z_var += z_sub_var
        z_std = np.sqrt((1 / iterations_n) * z_var)
        """
        rand_idx = np.random.randint(low=0, high=self.data.x_train.shape[0], size=6000)
        x_subset = self.data.x_train[rand_idx]
        z_subset, z_mean_subset, z_log_var_subset = self.network.vae.encoder(x_subset)
        z_std = np.std(z_subset, axis=0)
        """
        idx_prune = np.where(z_std < delta)[0]
        d_star = np.zeros((amount_n, 1))
        k = np.zeros((amount_n, ))
        for step, x_batch in enumerate(self.data.batch_generation_factor(L)):
            if step == amount_n:
                break
            x_1, y = x_batch
            z_rep, z_mean, z_log_var = self.network.vae.encoder(x_1)
            var_norm_z = np.var(z_rep, axis=0) / z_std
            var_norm_z[idx_prune] = 10000
            idx = np.where(var_norm_z == np.amin(var_norm_z))[0][0]
            d_star[step] = idx
            k[step] = y
        d_star_train, d_star_test = np.split(d_star, [int(.7 * len(d_star))])
        k_train, k_test = np.split(k, [int(.7 * len(k))])
        estimators = []
        for i in range(800):
            clf = LogisticRegression(multi_class='multinomial')
            estimators.append((str(i), clf))
        eclf1 = VotingClassifier(estimators=estimators, voting='hard')
        eclf1 = eclf1.fit(d_star_train, k_train)
        pred = eclf1.predict(d_star_test)
        accuracy = (1 / k_test.shape[0]) * np.sum(pred == k_test)  # vi får testa detta med klar modell
        df = pd.read_csv(self.path + "/history/scores.csv", delimiter=",")
        df.loc[0, "factorscore"] = accuracy
        df.to_csv(self.path + "/history/scores.csv", sep=",", index=False)





t = loadmodel(4, "factorvae") #lower lr for this one
t.train_vae(100)
#t.beta_score(10, 10)
#t.factor_score(10, 10)
#t.latent_traversal_2()
#t.train_vae()
#t.train_vae()
#t.latent_traversal(1)
#t.plot_reconstruction_2()

#t.beta_score(15000, 64)
#t.factor_score(15000, 64)
#t.DCI_score(15000, "gradient")