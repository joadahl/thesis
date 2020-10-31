from data import dsprite
import os
import tensorflow as tf
from betavae import betavae
from factorvae import factorvae
from betaclass import beta_score_classifier
import pickle
import numpy as np
from cbbetavae import cbbeta
from cbfactorvae import cbfactor
from cbbetascore import cbbetascore
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import normalize
from scipy.stats import entropy
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
import pandas as pd


class trainvae:
    def __init__(self, latent_dims, model, cluster):
        self.cluster = cluster
        self.data = dsprite()
        self.model = model
        self.latent_dims = latent_dims
        self.path = os.getcwd() 
        if not self.cluster:
            self.path = self.model + str(self.latent_dims)
            os.makedirs(self.path, exist_ok=True)
            os.makedirs(self.path + "/modelstore", exist_ok=True) #lägg till en till folder för sparandet
            os.makedirs(self.path + "/history", exist_ok=True)
        if not os.path.exists(self.path + "/history/scores.csv"):
            pd.DataFrame(columns=["betascore", "factorscore"]).to_csv(self.path + "/history/scores.csv", sep=",", index=False)
        if self.model == "betavae":
            self.vae = self.make_or_restore_beta_vae()
            self.callback = cbbeta(self.path)
        if self.model == "factorvae":
            self.vae = self.make_or_restore_factor_vae()
            self.callback = cbfactor(self.path)

    def make_or_restore_beta_vae(self):
        vae = betavae(self.latent_dims)
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

    def make_or_restore_factor_vae(self):
        vae = factorvae(self.latent_dims)
        optimizer_factor_vae = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        optimizer_discriminator = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        vae.compile(optimizer_factor_vae, optimizer_discriminator)
        if not os.listdir(self.path + "/modelstore/"):
            print("Creating a new model")
            return vae
        print("loading model")
        vae.fit(self.data.x[:1], self.data.x[:1], epochs=1) #det här verkar fungera, den fortsätter träna
        vae.load_weights(self.path + "/modelstore/factorvae.ckpt") #self.vae_optimizer = vae_optimizer
        vae.disc.load_weights(self.path + "/modelstore/factordisc.ckpt")
        with open(self.path + '/modelstore/optimizer_vae.pkl', 'rb') as f: #detta är ej testat
            weight_values_vae = pickle.load(f)
        vae.vae_optimizer.set_weights(weight_values_vae)
        with open(self.path + '/modelstore/optimizer_disc.pkl', 'rb') as f:
            weight_values_discriminator = pickle.load(f)
        vae.disc_optimizer.set_weights(weight_values_discriminator)
        return vae

    def train_vae(self):
        ini_epoch = 0
        if os.path.isfile(self.path + "/history/lossvae.csv"):
            ini_epoch = int(np.genfromtxt(self.path + "/history/lossvae.csv", delimiter=',')[-1][0])
        with tf.device('/gpu:0'):
            history = self.vae.fit(self.data.x_train_gen, validation_data=self.data.x_test_gen,
                                          verbose=0, epochs=ini_epoch + 50, initial_epoch=ini_epoch, callbacks=[self.callback])

    def beta_score(self, amount_n, L):
        z_diff_total = np.zeros((amount_n, self.latent_dims))
        y_total = np.zeros((amount_n, len(self.data.gen_factor_size)))
        for step, x_batch in enumerate(self.data.batch_generation_beta(L)): #self.data.batch_generation(L, "beta")
            if step == amount_n:
                break
            x_1, x_2, y = x_batch
            z_1, z_mean_1, z_log_var_1 = self.vae.encoder(x_1)
            z_2, z_mean_2, z_log_var_2 = self.vae.encoder(x_2)
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
                                                                        epochs=100,
                                                                        callbacks=[cbbetascore(self.path)])

        acc_last_epoch = np.genfromtxt(self.path + "/history/lossbeta.csv", delimiter=',')[-1][-1]
        df = pd.read_csv(self.path + "/history/scores.csv", delimiter=",")
        df.loc[0, "betascore"] = acc_last_epoch
        df.to_csv(self.path + "/history/scores.csv", sep=",", index=False)

    def factor_score(self, amount_n, L):
        #vi testar göra one-hot encodings här med, vi märker om det är skit
        delta = 0.05
        rand_idx = np.random.randint(low=0, high=self.data.x_train.shape[0], size=10000)
        x_subset = self.data.x_train[rand_idx]
        z_subset, z_mean_subset, z_log_var_subset = self.vae.encoder(x_subset)
        z_std = np.std(z_subset, axis=0)
        idx_prune = np.where(z_std < delta)[0]
        d_star = np.zeros((amount_n, 1))
        k = np.zeros((amount_n, ))
        for step, x_batch in enumerate(self.data.batch_generation_factor(L)):
            if step == amount_n:
                break
            x_1, y = x_batch
            z_rep, z_mean, z_log_var = self.vae.encoder(x_1)
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

    def DCI_score(self, amount_n, classifier):
        scaler_data = StandardScaler()
        scaler_targets = StandardScaler()
        z_total = np.zeros((amount_n, self.latent_dims))
        y_total = np.zeros((amount_n, len(self.data.gen_factor_size)))
        for step, x_batch in enumerate(self.data.batch_generation_dci()):
            if step == amount_n:
                break
            x_1, y = x_batch  # vi använder bara x_1 och y till att börja med
            z_1, z_mean_1, z_log_var_1 = self.vae.encoder(x_1)
            z_total[step] = z_1
            y_total[step] = y
        y_total = y_total[:, 1:len(self.data.gen_factor_size)]
        z_train, z_test = np.split(z_total, [int(.7 * len(z_total))])
        y_train, y_test = np.split(y_total, [int(.7 * len(y_total))])
        scaled_z_train = scaler_data.fit_transform(z_train)
        scaled_y_train = scaler_targets.fit_transform(y_train)
        scaled_z_test = scaler_data.transform(z_test)
        scaled_y_test = scaler_targets.transform(y_test)
        W_ij = np.zeros((self.latent_dims, len(self.data.gen_factor_size) - 1))  # gen-factors - 1
        list_clf = []

        for j in range(len(self.data.gen_factor_size) - 1):
            if classifier == "lasso":
                clf = linear_model.Lasso(alpha=0.015, max_iter=1000000000,
                                         tol=0.1)  # verkar vara okej parameter inställning
            if classifier == "gradient":
                clf = GradientBoostingRegressor(n_estimators=10, max_depth=20)
            clf.fit(scaled_z_train, scaled_y_train[:, j])  # vi kommer behöva random forests också för att utvärdera
            list_clf.append(clf)
            try:
                coef = clf.coef_
            except:
                coef = clf.feature_importances_
            W_ij[:, j] = coef
        eps = 1e-11
        R_ij = np.abs(W_ij)
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


t = trainvae(10, "betavae", True)
t.beta_score(15000, 64)
t.factor_score(15000, 64)
t.DCI_score(15000, "gradient")