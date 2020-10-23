from vae import betavae
from data import dsprite
from callbackvae import callbackvae
from betascoreclassifier import beta_score_classifier
from callbackbetascoreclassifier import callbackbetascoreclass
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import os
import numpy as np
import math
import pickle
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import linear_model
from matplotlib import pyplot as plt
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
        self.beta_vae = self.make_or_restore_beta_vae()


    def train_vae(self):
        with tf.device('/gpu:0'):
            #checkpoint = callbackvae()
            #self.beta_vae._set_inputs(Input(shape=(64, 64, 64, 1), dtype=tf.uint8))
            #x_test = self.data.x[0:150].astype("float32")
            #gen = tf.data.Dataset.from_tensor_slices((x_test, x_test)).batch(64)
            #self.beta_vae(tf.random.normal((64, 64, 64, 1)))
            #optimizer_beta_vae = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
            #self.beta_vae.compile(optimizer=optimizer_beta_vae)
            print("bajs")
            history = self.beta_vae.fit(self.data.x_train_gen, validation_data=self.data.x_test_gen,
                                   epochs=50, verbose=0, callbacks=[callbackvae()])
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

    def beta_score(self, amount_n, L):
        z_diff_total = np.zeros((amount_n, self.latent_dims))
        y_total = np.zeros((amount_n, 6))
        for step, x_batch in enumerate(self.data.batch_generation(L, "beta")):
            if step == amount_n:
                break
            x_1, x_2, y = x_batch
            z_1, z_mean_1, z_log_var_1 = self.beta_vae.encoder(x_1)
            z_2, z_mean_2, z_log_var_2 = self.beta_vae.encoder(x_2)
            z_diff = z_1 - z_2
            z_batch_diff = np.mean(z_diff, axis=0)
            z_diff_total[step] = z_batch_diff
            y_total[step] = y

        z_diff_train, z_diff_test = np.split(z_diff_total, [int(.8 * len(z_diff_total))])
        y_train, y_test = np.split(y_total, [int(.8 * len(y_total))])
        data_train = tf.data.Dataset.from_tensor_slices((z_diff_train, y_train)).batch(1)
        data_test = tf.data.Dataset.from_tensor_slices((z_diff_test, y_test)).batch(1)
        beta_classifier = beta_score_classifier()
        beta_classifier.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9,
                                                             epsilon=1e-08)) #verkar vara okej lr
        history = beta_classifier.fit(data_train, validation_data=data_test, verbose=0,
                                                                        epochs=100,
                                                                        callbacks=[callbackbetascoreclass()])
        with open('history/beta_score.csv', 'a') as f:
            f.write("beta_score:" + str(history.history['val_acc'][-1]))

    def factor_score(self, amount_n, L):
        #vi testar göra one-hot encodings här med, vi märker om det är skit

        #compute std
        size_n = 500
        subset_n_std = 10000 #value from challenges
        iterations_n = int(subset_n_std/size_n)
        rand_idx = np.random.randint(low=self.data.x_train.shape[0])
        x_subset = self.data.x[rand_idx:subset_n_std + rand_idx] #lägg till så att vi prunar bort std < 0.05
        z_var = np.zeros((self.latent_dims,))
        for l in range(iterations_n):
            z, z_mean, z_log_var = self.beta_vae.encoder(x_subset[l*size_n:(l+1)*size_n])
            z_sub_var = np.var(z, axis=0) #this should be ginis
            #print(np.sqrt(z_sub_var))
            #print("hej")
            z_var += z_sub_var
        z_std = np.sqrt((1 / iterations_n) * z_var)
        delta = 0.05 #threshold according to challenging
        idx_prune = np.where(z_std < delta) #shuld be used to prune
        d_star = np.zeros((amount_n, 1))  # vet ej om dessa ska vara denna storleken
        k = np.zeros((amount_n, 6))  # vet ej om dessa ska vara denna storleken
        for step, x_batch in enumerate(self.data.batch_generation(L, "factor")):
            if step == amount_n:
                break
            x_1, y = x_batch
            z_rep, z_mean, z_log_var = self.beta_vae.encoder(x_1)
            var_norm_z = np.var(z_rep, axis=0) / z_std
            #print(var_norm_z.shape)
            var_norm_z[idx_prune[0]] = 10000
            idx = np.where(var_norm_z == np.amin(var_norm_z))
            d = idx[0][0]
            d_star[step] = d
            k[step] = y
        k_categorical = np.argmax(k, axis=1)
        d_star_train, d_star_test = np.split(d_star, [int(.6 * len(d_star))])
        k_train, k_test = np.split(k_categorical, [int(.6 * len(k_categorical))])
        estimators = []
        for i in range(800):
            clf = LogisticRegression(multi_class='multinomial')
            estimators.append((str(i), clf))
        eclf1 = VotingClassifier(estimators=estimators, voting='hard')
        eclf1 = eclf1.fit(d_star_train, k_train)
        pred = eclf1.predict(d_star_test)
        accuracy = (1 / k_test.shape[0]) * np.sum(pred == k_test) #vi får testa detta med klar modell
        print(accuracy)
        with open('history/factor_score.csv', 'a') as f:
            f.write("factor_score:" + str(accuracy))


    def DCI_score(self, amount_n, classifier):
        scaler_data = StandardScaler()
        scaler_targets = StandardScaler()
        z_total = np.zeros((amount_n, self.latent_dims))
        y_total = np.zeros((amount_n, 6))
        for step, x_batch in enumerate(self.data.batch_generation_dci()):
            if step == amount_n:
                break
            x_1, y = x_batch  # vi använder bara x_1 och y till att börja med
            z_1, z_mean_1, z_log_var_1 = self.beta_vae.encoder(x_1)
            z_total[step] = z_1
            y_total[step] = y
        z_train, z_test = np.split(z_total, [int(.6 * len(z_total))])
        y_train, y_test = np.split(y_total, [int(.6 * len(y_total))])
        scaled_z_train = scaler_data.fit_transform(z_train)
        scaled_y_train = scaler_targets.fit_transform(y_train)
        scaled_z_test = scaler_data.transform(z_test) #borde vara såå här
        scaled_y_test = scaler_targets.transform(y_test)
        W = np.zeros((self.latent_dims, 5))  # gen-factors - 1
        list_clf = []
        for j in range(1, 6):
            if classifier == "lasso":
                clf = linear_model.Lasso(alpha=0.015, max_iter=1000000000,
                                         tol=0.1)  # verkar vara okej parameter inställning
            if classifier == "gradient":
                clf = GradientBoostingRegressor(n_estimators=10, max_depth=20)
            clf.fit(scaled_z_train, scaled_y_train[:, j])  # vi kommer behöva random forests också för att utvärdera
            list_clf.append(clf)
            if classifier == "lasso":
                W[:, j - 1] = clf.coef_
            else:
                W[:, j - 1] = clf.feature_importances_
        R = np.abs(W)
        P = np.zeros((self.latent_dims, 5))  # 10,5
        H = np.zeros((self.latent_dims,))
        rho = np.zeros((self.latent_dims,))
        for i in range(R.shape[0]):
            P[i] = R[i] / (np.sum(R[i]) + 1e-11)
            H[i] = - np.sum(P[i] * (np.log(P[i] + 1e-11) / np.log(5)))
            rho[i] = np.sum(R[i]) / np.sum(R)
        disentanglement_mean = np.sum(rho * (1 - H))  # denna tror jag är korrekt
        with open('history/DCI_score.csv', 'a') as f:
            f.write("disentanglement_" + classifier + ":" + str(rho * (1 - H)))
            f.write("\n")
            f.write("\n")
            f.write("disentanglement_mean:" + str(disentanglement_mean))
        # completeness
        H_c = np.zeros((5,))
        for i in range(H_c.shape[0]):
            H_c[i] = - np.sum(
                P[:, i] * (np.log(P[:, i] + 1e-11) / np.log(5)))
        completeness_score = 1 - H_c
        mean_completeness = np.mean(completeness_score)
        with open('history/DCI_score.csv', 'a') as f:
            f.write("\n")
            f.write("completeness:" + str(completeness_score))
            f.write("\n")
            f.write("completeness_mean:" + str(mean_completeness))
        NRMSE_avg = np.zeros((5,))
        for j in range(1, 6):
            reg = list_clf[j-1]
            pred = reg.predict(scaled_z_test)
            NRMSE_avg[j-1] = np.sqrt(np.mean((pred - scaled_y_test[:, j]) ** 2))
        mean_informativeness = np.mean(NRMSE_avg)
        with open('history/DCI_score.csv', 'a') as f:
            f.write("\n")
            f.write("informativeness:" + str(NRMSE_avg))
            f.write("\n")
            f.write("informativeness_mean:" + str(mean_informativeness))


    def plot_reconstruction(self):
        n = 10
        z, z_mean, z_log_var = self.beta_vae.encoder(self.data.x_test[0:n])
        x_rec = self.beta_vae.decoder(z)
        x_rec = x_rec.numpy()
        x_rec = x_rec.reshape(z.shape[0], 64, 64)
        plt.figure(figsize=(20, 4))
        for i in range(n):
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(self.data.x_test[i].reshape(64, 64))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(x_rec[i].reshape(64, 64))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        #plt.savefig("C:\\Users\\andre\\Desktop\\images\\reconfactor.png")
        plt.show()

    def latent_traversal(self, dim):
        random_index = np.random.randint(low=0, high=self.data.x_test.shape[0], size=2)
        z, z_mean, z_log_var = self.beta_vae.encoder(self.data.x_test[random_index])
        grid_z = np.linspace(-3, 3, num=10)
        z = z.numpy()
        for i in range(len(grid_z)):
            z[0][dim] = grid_z[i] #ändra andra värdet
            x_rec = self.beta_vae.decoder(z)
            ax = plt.subplot(1, len(grid_z), i + 1)
            x_rec = x_rec.numpy()
            plt.imshow(x_rec[0].reshape(64, 64))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()

    def latent_traversal2(self, dim):
        random_index = np.random.randint(low=0, high=self.data.x_test.shape[0], size=2)
        z, z_mean, z_log_var = self.beta_vae.encoder(self.data.x_test[random_index])
        grid_z = np.linspace(-3, 3, num=10)
        z = z.numpy()
        x_rec = self.beta_vae.decoder(z)
        x_rec = x_rec.numpy()
        plot_fig = x_rec[0].reshape(64, 64)
        fig = plt.figure()
        ax1 = fig.add_subplot(2, 1, 1)
        plt.gray()
        ax1.imshow(plot_fig)
        ax1.get_xaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False)

        plt.show()

    def latent_traversal3(self, dim):
        random_index = np.random.randint(low=0, high=self.data.x_test.shape[0], size=2)
        z, z_mean, z_log_var = self.beta_vae.encoder(self.data.x_test[random_index])
        grid_z = np.linspace(-5, 5, num=10) #vi borde börja på mean, och röra oss mot kanterna, måste fixa gränserna
        z = z.numpy()
        x_rec_real = self.beta_vae.decoder(z)
        x_rec_real = x_rec_real.numpy()
        fig = plt.figure(figsize=(30, 15)) #denna kommer behöva ändras
        for i in range(len(grid_z)):
            z[0][dim] = grid_z[i]  # ändra andra värdet
            x_rec = self.beta_vae.decoder(z)
            x_rec = x_rec.numpy()
            ax1 = fig.add_subplot(1, len(grid_z) + 2, i + 1)
            ax1.get_xaxis().set_visible(False)
            ax1.get_yaxis().set_visible(False)
            plt.gray()
            ax1.imshow(x_rec[0].reshape(64, 64))
            if i == len(grid_z) - 1:
                ax1 = fig.add_subplot(1, len(grid_z) + 2, len(grid_z) + 2)
                ax1.get_xaxis().set_visible(False)
                ax1.get_yaxis().set_visible(False)
                plt.gray()
                ax1.imshow(x_rec_real[0].reshape(64, 64))
        plt.savefig("C:\\Users\\andre\\Desktop\\figures\\betaVAElat4beta4\\betaVAElat4beta4" + "dim" + str(dim) + ".png",
                    bbox_inches='tight', pad_inches=0, transparent=True)
        plt.show()



        #figure, axs = plt.subplots(nrows=1, ncols=len(grid_z) + 2)
        #figure.setp(axs.get_xticklabels(), visible=False)
        #plt.show()
        #for i in range(len(grid_z)):
        #    z[0][dim] = grid_z[i] #ändra andra värdet
        #    x_rec = self.beta_vae.decoder(z)
        #    ax = plt.subplot(1, len(grid_z), i + 1)
        #    x_rec = x_rec.numpy()
        #    plt.imshow(x_rec[0].reshape(64, 64))
        #    plt.gray()
        #    ax.get_xaxis().set_visible(False)
        #    ax.get_yaxis().set_visible(False)
        #plt.show()



dis = disentanglement(4, 4, False)
dis.beta_score(15000, 20)
#dis.plot_reconstruction()
#dis.latent_traversal3(0)
#vae = dis.make_or_restore_beta_vae()
#x_rec = vae.call(dis.data.x_train[0:50].astype("float32"))
#dis.train_vae()
#dis.beta_score(10, 10) #denna ska testas med vikter från slurm, 10000 träningspunkter, 5000 test punker enligt challenging
#dis.factor_score(10, 10)
#dis.DCI_score(10, "lasso")
#dis.DCI_score_2(10, "gradient")
#dis.gradient_boost(10)
#dis.gradient_boost(10)
#dis.DCI_score_2(100)

#print(dis.store_last_checkpoint.epoch_loss)
#print(dis.beta_vae.checkpoint.epoch_loss)
#print(x_rec.shape)
