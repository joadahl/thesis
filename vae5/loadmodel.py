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
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize
from scipy.stats import entropy
from sklearn.ensemble import GradientBoostingRegressor
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
        beta_classifier.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.01, initial_accumulator_value=0.1, epsilon=1e-07))
        history = beta_classifier.fit(data_train, validation_data=data_test, verbose=0,
                                                                        epochs=1000,
                                                                        callbacks=[cbbetascore(self.path)])
        acc_last_epoch = np.genfromtxt(self.path + "/history/lossbeta.csv", delimiter=',')[-1][-1]
        df = pd.read_csv(self.path + "/history/scores.csv", delimiter=",")
        df.loc[0, "betascore"] = acc_last_epoch
        df.to_csv(self.path + "/history/scores.csv", sep=",", index=False)

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
            z_sub_var = np.var(z, axis=0)
            z_var += z_sub_var
        z_std = np.sqrt((1 / iterations_n) * z_var)
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

    def DCI_score(self, amount_n):
        scaler_data = StandardScaler()
        scaler_targets = StandardScaler()
        z_total = np.zeros((amount_n, self.latent_dims))
        y_total = np.zeros((amount_n, len(self.data.gen_factor_size)))
        for step, x_batch in enumerate(self.data.batch_generation_dci()):
            if step == amount_n:
                break
            x_1, y = x_batch
            z_1, z_mean_1, z_log_var_1 = self.network.vae.encoder(x_1)
            z_total[step] = z_1
            y_total[step] = y
        y_total = y_total[:, 1:len(self.data.gen_factor_size)]
        z_train, z_test = np.split(z_total, [int(.7 * len(z_total))])
        y_train, y_test = np.split(y_total, [int(.7 * len(y_total))])
        scaled_z_train = scaler_data.fit_transform(z_train)
        scaled_y_train = scaler_targets.fit_transform(y_train)
        scaled_z_test = scaler_data.transform(z_test)
        scaled_y_test = scaler_targets.transform(y_test)
        W_ij = np.zeros((self.latent_dims, len(self.data.gen_factor_size) - 1))
        list_clf = []

        for j in range(len(self.data.gen_factor_size) - 1):
            clf = GradientBoostingRegressor(n_estimators=10, max_depth=20)
            clf.fit(scaled_z_train, scaled_y_train[:, j])
            list_clf.append(clf)
            coef = clf.feature_importances_
            W_ij[:, j] = coef
        eps = 1e-11
        R_ij = np.abs(W_ij) #denna borde nte behövas
        P_ij = normalize(R_ij, axis=1, norm='l1')
        D_i = (1 - entropy((P_ij + eps), base=len(self.data.gen_factor_size) - 1, axis=1))
        rho_i = np.sum(R_ij, axis=1) / np.sum(R_ij)
        scaled_disentanglement = D_i * rho_i
        C_j = np.array(1 - entropy((P_ij + eps), base=self.latent_dims, axis=0))
        informativeness = np.zeros((len(self.data.gen_factor_size) - 1,))
        for j in range(len(self.data.gen_factor_size) - 1):
            reg = list_clf[j]
            pred = reg.predict(scaled_z_test)
            informativeness[j] = np.sqrt(np.mean((pred - scaled_y_test[:, j]) ** 2))
        df = pd.read_csv(self.path + "/history/scores.csv", delimiter=",")
        if 'DCIscore' in df.columns:
            return
        df1 = pd.DataFrame()
        df2 = pd.DataFrame()
        df3 = pd.DataFrame()
        df4 = pd.DataFrame()
        df1["DCIscore"] = pd.Series(scaled_disentanglement)
        df2["Completeness"] = pd.Series(C_j)
        df3["Informativeness"] = pd.Series(informativeness)
        df4["R"] = pd.Series(np.ndarray.flatten(R_ij))
        df = df.reset_index()
        df1 = df1.reset_index()
        df2 = df2.reset_index()
        df3 = df3.reset_index()
        df4 = df4.reset_index()
        df_final = pd.concat([df, df1, df2, df3, df4], axis=1)
        df_final.to_csv(self.path + "/history/scores.csv", sep=",", index=False)

    def plot_reconstruction(self):
        n = 10
        pad = 5
        z, z_mean, z_log_var = self.network.vae.encoder(self.data.x_test[0:n])
        x_rec = self.network.vae.decoder(z)
        x_rec = x_rec.numpy()
        x_rec = x_rec.reshape(z.shape[0], 64, 64)
        rows = ["Ground truth "r"$\bf{x}$", "Reconstructions " r"$\bf{\hat{x}}$"]
        fig, axes = plt.subplots(nrows=2, figsize=(20, 4))
        for ax, row in zip(axes, rows):
            # ax.set_ylabel(row, rotation=0, size='medium')
            ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                        xycoords=ax.yaxis.label, textcoords='offset points',
                        size='large', fontsize=24, ha='right', va='center')
            ax.set_yticks([])
            ax.set_xticks([])
            ax._frameon = False
        for i in range(n):
            ax = fig.add_subplot(2, n, i + 1)
            plt.imshow(self.data.x_test[i].reshape(64, 64))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            ax = fig.add_subplot(2, n, i + 1 + n)
            plt.imshow(x_rec[i].reshape(64, 64))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        #plt.title(self.model + "," + r"$D=$" + str(self.latent_dims), loc='center')
        fig.set_facecolor('w')
        fig.tight_layout()
        plt.show()

    def latent_traversal(self):
        limit = 0.5
        random_index = np.random.randint(low=0, high=self.data.x_test.shape[0], size=2)
        z, z_mean, z_log_var = self.network.vae.encoder(self.data.x_test[random_index])
        grid_z = np.linspace(-limit, limit, num=10)
        z = z.numpy()
        fig, axes = plt.subplots(nrows=z.shape[1], ncols=1, figsize=(20, 4))
        pad = 0
        counter = 0
        rows = []
        for i in range(z.shape[1]):
            rows.append(r"$z_{:d}$".format(i))
        for ax, row in zip(axes, rows):
            #ax.set_ylabel(row, rotation=0, size='medium')
            ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                        xycoords=ax.yaxis.label, textcoords='offset points',
                        size='large', fontsize=18, ha='right', va='center')
            ax.set_yticks([])
            ax.set_xticks([])
            ax._frameon = False
            counter += 1
            if counter == (z.shape[1]):
                ax.set_xticks([-limit, limit])
                ax.set_xticklabels([r"$z_{j}=-1$", r"$z_{j}=1$"], fontsize=14)
            #    ax.tick_params(axis="x", direction="in", pad=500)
        for k in range(z.shape[1]):
            for j in range(len(grid_z)): #detta kanske är rätt, svårt att säga
                z[0][k] = grid_z[j]
                x_rec = self.network.vae.decoder(z)
                x_rec = x_rec.numpy()
                ax = fig.add_subplot(z.shape[1], len(grid_z), (k * len(grid_z)) + (j+1))
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                plt.gray()
                ax.imshow(x_rec[0].reshape(64, 64), aspect="auto")
        #fig.set_facecolor('w')
        #fig.tight_layout()
        plt.show()



t = loadmodel(4, "betavae", 4) #lower lr for this
t.train_vae(50)
#t.beta_score(15000, 64)
#t.factor_score(15000, 64)
#t.DCI_score(15000)
