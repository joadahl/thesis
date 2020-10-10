from tensorflow import keras
from factor.discriminator import discriminator
import tensorflow as tf
import numpy as np
from factor.data import dsprite
from factor.vae import vae

class factorvae(keras.Model):
    def __init__(self, latent_dim):
        super(factorvae, self).__init__()
        self.vae = vae(latent_dim)
        self.discriminator = discriminator()
        self.batch_size = 50
        self.gamma = 1
        self.bce = tf.keras.losses.BinaryCrossentropy()
        self.loss_tracker = keras.metrics.Mean(name="loss")

    def call(self, inputs):
        inputs = tf.cast(inputs, dtype=tf.float32)
        z, z_mean, z_log_var = self.encoder(inputs)
        x_rec = self.decoder(z)
        return x_rec

    def permute_dims(self, z):
        z_cols = tf.unstack(z, axis=1)
        store = []
        for z_col in z_cols:
            rand = np.random.randint(self.batch_size)
            z_perm = tf.roll(z_col, shift=rand, axis=0)
            store.append(z_perm)
        z_permuted = tf.stack(store, axis=1)
        return z_permuted

    def train_step(self, data):
        data = data[0]
        with tf.GradientTape(persistent=True) as tape:
            loss_vae, loss_discriminator = self.generate_loss(data)
        grads_vae = tape.gradient(loss_vae, self.vae.trainable_weights)
        grads_discriminator = tape.gradient(loss_discriminator, self.discriminator.trainable_weights)
        self.optimizer[0].apply_gradients(zip(grads_vae, self.vae.trainable_weights))
        self.optimizer[1].apply_gradients(zip(grads_discriminator, self.discriminator.trainable_weights))
        self.loss_tracker.update_state(loss_vae)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        data = data[0]
        loss_vae, loss_discriminator = self.generate_loss(data)
        self.loss_tracker.update_state(loss_vae)
        return {"loss": self.loss_tracker.result()}

    def generate_loss(self, data):
        z, z_mean, z_log_var = self.vae.encoder(data)
        x_rec = self.vae.decoder(z)
        recon_error = tf.reduce_mean(self.bce(data, x_rec)) * 4096
        KL = 0.5 * tf.reduce_mean(tf.math.exp(z_log_var) + tf.math.square(z_mean) - 1. - z_log_var)
        z_permuted = self.permute_dims(z)
        logits_true, probs_true = self.discriminator(z)
        logits_permuted, probs_permuted = self.discriminator(z_permuted)
        tc_reg = self.gamma * tf.reduce_mean(logits_true[:, 0] - logits_true[:, 1], axis=0)
        loss_vae = recon_error + KL + tc_reg
        loss_discriminator = - tf.add(0.5 * tf.reduce_mean(tf.math.log(probs_true[:, 0])),
                             0.5 * tf.reduce_mean(tf.math.log(1 - probs_permuted[:, 1])))
        return loss_vae, loss_discriminator





#data = dsprite()
#data_test = data.x[0:50]
#facvae = factorvae(10)
#opt1 = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
#opt2 = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9, epsilon=1e-08)
#facvae.compile(optimizer=(opt1, opt2))
#facvae.fit(data_test, data_test, batch_size=15, epochs = 35)

#loss_vae, loss_discriminator = vae.generate_loss(data_test)
#print(loss_vae.shape)
#print(loss_vae)