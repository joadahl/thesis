from vae import betavae
from data import dsprite
import math
import tensorflow as tf
from callbackvae import callbackvae
import os
import numpy as np
from sklearn.model_selection import train_test_split
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class disentanglement:
    def __init__(self, latent_dims, beta, cluster):
        self.cluster = cluster
        self.data = dsprite()
        self.latent_dims = latent_dims
        self.beta = beta
        self.path = os.path.dirname(__file__) #ta bort om kluster, verkar inte gå med os.path, stod self.path förrst på checkpoint_vae
        if self.cluster:
            self.path = "modelstore"  # använd för klustret

    def train_vae(self):
        #with tf.device('/gpu:0'):
        strategy = tf.distribute.MirroredStrategy()
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        with strategy.scope():
            self.vae = betavae(self.latent_dims, self.beta)
            optimizer_vae = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
            self.vae.compile(optimizer=optimizer_vae)
            checkpoint_vae = tf.keras.callbacks.ModelCheckpoint(
                os.path.join(self.path + "/modelstore/", "vae" + str(self.latent_dims)),
                monitor="val_loss",
                verbose=0,
                save_best_only=True,
                save_weights_only=True,
                mode="min",
                save_freq="epoch")
            history = self.vae.fit(self.data.x_train, validation_data=self.data.x_test,
                                   epochs=2000, verbose=0, callbacks=[callbackvae(), checkpoint_vae])
            if self.cluster == False:
                os.makedirs(self.path + "/history/", exist_ok=True) #ta bort om kluster
            np.save(os.path.join("history/", "history" + "vae" + str(self.latent_dims)), history.history)


#print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
#tf.test.is_built_with_cuda()
dis = disentanglement(4, 4, False)
dis.train_vae()
