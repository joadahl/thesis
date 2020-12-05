from data import dsprite
import os
import tensorflow as tf
from betavae import betavae
import numpy as np
from cbbetavae import cbbeta
import pandas as pd
from cbfactorvae import cbfactor
from factorvae import factorvae
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import normalize
from scipy.stats import entropy
from sklearn.ensemble import GradientBoostingRegressor
from matplotlib import gridspec
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

class loadmodel:
    def __init__(self, latent_dims, model, weight):
        self.data = dsprite()
        self.model = model
        self.weight = weight
        self.latent_dims = latent_dims
        self.path = self.model + "lat" + str(self.latent_dims) + "weight" + str(self.weight)
        os.makedirs(self.path, exist_ok=True)
        os.makedirs(self.path + "/modelstore", exist_ok=True) #lägg till en till folder för sparandet
        os.makedirs(self.path + "/history", exist_ok=True)
        os.makedirs(self.path + "/figures", exist_ok=True)
        if not os.path.exists(self.path + "/history/scores.csv"):
            pd.DataFrame(columns=["betascore", "factorscore"]).to_csv(self.path + "/history/scores.csv", sep=",", index=False)
        if self.model == "betavae":
            self.network = self.make_or_restore_beta_vae(self.weight)
            self.callback = cbbeta(self.path)
        if self.model == "factorvae":
            self.network = self.make_or_restore_factor_vae(self.weight)
            self.callback = cbfactor(self.path)


    def make_or_restore_beta_vae(self, weight):
        network = betavae(self.latent_dims, weight)
        optimizer_beta_vae = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9,
                                                        beta_2=0.999, epsilon=1e-08)
        network.compile(optimizer=optimizer_beta_vae)
        if not os.listdir(self.path + "/modelstore"):
            print("Creating a new model")
            return network
        print("loading weights")
        network.fit(self.data.x[:1], self.data.x[:1], epochs=1)
        latest = tf.train.latest_checkpoint(self.path + "/modelstore")
        #print(latest)
        #print(latest)
        network.load_weights(latest)
        #network.load_weights(self.path + "/modelstore/" + "ckpt" + str(5) + "-1")
        return network

    def make_or_restore_factor_vae(self, weight):
        network = factorvae(self.latent_dims, weight)
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
            ini_epoch = int(np.genfromtxt(self.path + "/history/lossvae.csv", delimiter=',')[-1][0]) + 1
        with tf.device('/gpu:0'):
            history = self.network.fit(self.data.x_train_gen, validation_data=self.data.x_test_gen,
                                          verbose=0, epochs=ini_epoch + epochs_n, initial_epoch=ini_epoch, callbacks=[self.callback])

    def beta_score(self, amount_n, L):
        z_diff_total = np.zeros((amount_n, self.latent_dims))
        y_total = np.zeros((amount_n, ))
        for step, x_batch in enumerate(self.data.batch_generation_beta_score(L)):
            if step == amount_n:
                break
            x_1, x_2, y = x_batch
            z_1, z_mean_1, z_log_var_1 = self.network.vae.encoder(x_1)
            z_2, z_mean_2, z_log_var_2 = self.network.vae.encoder(x_2)
            z_diff = np.abs(z_1 - z_2)
            z_batch_diff = np.mean(z_diff, axis=0)
            z_diff_total[step] = z_batch_diff
            y_total[step] = y
        scaler = MinMaxScaler()
        z_diff_total = scaler.fit_transform(z_diff_total)
        z_diff_train, z_diff_test = np.split(z_diff_total, [int(.7 * len(z_diff_total))])
        y_train, y_test = np.split(y_total, [int(.7 * len(y_total))])
        clf = LogisticRegression()
        clf.fit(z_diff_train, y_train)
        accuracy = clf.score(z_diff_test, y_test)
        df = pd.read_csv(self.path + "/history/scores.csv", delimiter=",")
        df.loc[0, "betascore"] = accuracy
        df.to_csv(self.path + "/history/scores.csv", sep=",", index=False) #måste testa denna ordentligt

    def factor_score(self, amount_n, L):
        z_factor = np.zeros((amount_n, ))
        y_factor = np.zeros((amount_n, ))
        delta = 0.05
        size_n = 500
        subset_n_std = 10000  # value from challenges
        iterations_n = int(subset_n_std / size_n)
        rand_idx = np.random.randint(low=self.data.x_train.shape[0])
        x_subset = self.data.x[rand_idx:subset_n_std + rand_idx]  # lägg till så att vi prunar bort std < 0.05
        z_var = np.zeros((self.latent_dims,))
        for l in range(iterations_n):
            z, z_mean, z_log_var = self.network.vae.encoder(x_subset[l * size_n:(l + 1) * size_n])
            z_sub_var = np.var(z, axis=0)
            z_var += z_sub_var
        z_std = np.sqrt((1 / iterations_n) * z_var)
        relevant_dims = np.where(z_std > delta)[0]
        for step, x_batch in enumerate(self.data.batch_generation_factor_score(L)):
            if step == amount_n:
                break
            x_1, y = x_batch
            z_1, z_mean_1, z_log_var_1 = self.network.vae.encoder(x_1)
            y_factor[step] = y
            var_norm_z = np.var(z_1, axis=0) / z_std
            relevant_var_norm_z = var_norm_z[relevant_dims]
            idx = np.where(relevant_var_norm_z == np.amin(relevant_var_norm_z))[0][0]
            z_factor[step] = idx
        z_factor = np.array(z_factor, dtype=np.int)
        v_ij = np.zeros((relevant_dims.shape[0], len(self.data.gen_factor_size)))
        for i in range(v_ij.shape[0]):
            for j in range(v_ij.shape[1]):
                v_ij[i, j] = np.sum((z_factor == i) & (y_factor == j))
        C = np.argmax(v_ij, axis=1)
        pred = C[z_factor]
        accuracy = (1 / amount_n) * np.sum(pred == y_factor)
        df = pd.read_csv(self.path + "/history/scores.csv", delimiter=",")
        df.loc[0, "factorscore"] = accuracy
        df.to_csv(self.path + "/history/scores.csv", sep=",", index=False)

    def DCI_score(self, amount_n, L):
        z_dci = np.zeros((amount_n, self.latent_dims))
        y_dci = np.zeros((amount_n, len(self.data.gen_factor_size)-1))
        for step, x_batch in enumerate(self.data.batch_generation_dci_score(L)):
            if step == int(amount_n / L):
                break
            x_1, y = x_batch
            z_1, z_mean_1, z_log_var_1 = self.network.vae.encoder(x_1)
            z_dci[step * L: (step + 1) * L] = z_1
            y_dci[step * L: (step + 1) * L] = y[:, 1:len(self.data.gen_factor_size)]
        scaler_data = StandardScaler()
        scaled_z = scaler_data.fit_transform(z_dci)
        R_ij = np.zeros((self.latent_dims, len(self.data.gen_factor_size) - 1))
        for j in range(len(self.data.gen_factor_size) - 1):
            clf = GradientBoostingClassifier(n_estimators=10)  # använder classifier
            clf.fit(scaled_z, y_dci[:, j])
            R_ij[:, j] = np.abs(clf.feature_importances_)
        D_i = 1 - entropy(R_ij, base=len(self.data.gen_factor_size) - 1, axis=1)
        rho_i = np.sum(R_ij, axis=1) / (np.sum(R_ij))
        scaled_disentanglement = np.sum(D_i * rho_i)
        df = pd.read_csv(self.path + "/history/scores.csv", delimiter=",")
        if 'DCIscore' in df.columns:
            return
        df1 = pd.DataFrame()
        df2 = pd.DataFrame()
        df1["DCIscore"] = pd.Series(scaled_disentanglement)
        df2["R"] = pd.Series(np.ndarray.flatten(R_ij))
        df = df.reset_index()
        df1 = df1.reset_index()
        df2 = df2.reset_index()
        df_final = pd.concat([df, df1, df2], axis=1)
        df_final.to_csv(self.path + "/history/scores.csv", sep=",", index=False)

    def plot_reconstruction(self):
        n_col = 10
        n_row = 2
        pad = 5
        z, z_mean, z_log_var = self.network.vae.encoder(self.data.x_test[0:n_col])
        x_rec = self.network.vae.decoder(z)
        x_rec = x_rec.numpy()
        x_rec = x_rec.reshape(z.shape[0], 64, 64)
        rows = [r"$\bf{x}$", r"$\bf{\hat{x}}$"]
        fig = plt.figure(figsize=(10, 4))
        spec = fig.add_gridspec(nrows=n_row, ncols=n_col,
                                wspace=0.0, hspace=0.0)
        for i in range(n_row):
            for j in range(n_col):
                f_ax1 = fig.add_subplot(spec[i, j])
                if j == 0:
                    f_ax1.annotate(rows[i], xy=(0, 0.5), xytext=(-f_ax1.yaxis.labelpad - pad, 0),
                                   xycoords=f_ax1.yaxis.label, textcoords='offset points',
                                   size='large', fontsize=20, ha='right', va='center')
                f_ax1.set_yticks([])
                f_ax1.set_xticks([])
                if i == 0:
                    plt.gray()
                    f_ax1.imshow(self.data.x_test[j].reshape(64, 64), aspect='auto')
                else:
                    plt.gray()
                    plt.imshow(x_rec[j].reshape(64, 64), aspect='auto')
        plt.savefig(self.path + "/figures/" + "recons" + self.path + ".png")
        plt.show()

    def latent_traversal(self):
        limit = 2
        pad = 0
        random_index = np.random.randint(low=0, high=self.data.x_test.shape[0], size=2)
        z, z_mean, z_log_var = self.network.vae.encoder(self.data.x_test[random_index])
        z = z.numpy()
        fig = plt.figure(figsize=(18, 4))
        ax = fig.add_subplot(111)
        ax.set_yticks([])
        ax.set_xticks([-100, 100])
        ax.set_xticklabels([r"$\mu_{z_{j}}-$" + str(limit), r"$\mu_{z_{j}}+$" + str(limit)], fontsize=16)
        n_col = 10
        n_row = z.shape[1]
        spec = gridspec.GridSpec(nrows=n_row, ncols=n_col, width_ratios=np.ones((n_col,)), height_ratios=np.ones((n_row,)),
         wspace=0, hspace=0)
        for i in range(n_row):
            start = z[0][i] - limit
            end = z[0][i] + limit
            int1 = np.linspace(start, z[0][i], num=5, endpoint=False)
            int2 = np.linspace(z[0][i], end, num=5, endpoint=False)  # detta måste ändras
            grid_z = np.concatenate((int1, int2), axis=0)
            for j in range(n_col):
                f_ax1 = fig.add_subplot(spec[i, j])
                if j == 0:
                    string_i = str(i)
                    string_list = [int(string_i) for string_i in str(string_i)]
                    if len(string_list) == 2:
                        f_ax1.annotate(r"$z_{}$".format(string_list[0]) + r"$_{}$".format(string_list[1]), xy=(0, 0.5), xytext=(-f_ax1.yaxis.labelpad - pad, 0),
                                   xycoords=f_ax1.yaxis.label, textcoords='offset points',
                                   size='large', fontsize=20, ha='right', va='center')
                    else:
                        f_ax1.annotate(r"$z_{}$".format(string_list[0]), xy=(0, 0.5),
                                       xytext=(-f_ax1.yaxis.labelpad - pad, 0),
                                       xycoords=f_ax1.yaxis.label, textcoords='offset points',
                                       size='large', fontsize=20, ha='right', va='center')

                f_ax1.set_yticks([])
                f_ax1.set_xticks([])
                f_ax1.set_aspect('equal')
                z[0][i] = grid_z[j]
                x_rec = self.network.vae.decoder(z)
                x_rec = x_rec.numpy()
                plt.gray()
                f_ax1.imshow(x_rec[0].reshape(64, 64), aspect='auto')
        plt.savefig(self.path + "/figures/" + "traversal" + self.path + ".png")
        plt.show()


t = loadmodel(10, "betavae", 4) #lower lr for this
t.beta_score(15000, 64)
t.factor_score(15000, 64)
t.DCI_score(15000, 10)
