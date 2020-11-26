from tensorflow import keras
from vae import vae
from discriminator import discriminator

import tensorflow as tf
import numpy as np
tf.keras.backend.set_floatx('float64')

class factorvae(keras.Model):
    def __init__(self, latent_dims):
        super(factorvae, self).__init__()
        self.latent_dims = latent_dims
        self.gamma = 40
        self.vae = vae(latent_dims)
        self.disc = discriminator()
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.disc_tracker = keras.metrics.Mean(name="disc_loss")
        self.recon_tracker = keras.metrics.Mean(name="recon_error")
        self.kl_tracker = keras.metrics.Mean(name="KL")
        self.tc_reg_tracker = keras.metrics.Mean(name="tc_reg")
        self.loss_tracker_val = keras.metrics.Mean(name="val_loss")
        self.disc_tracker_val = keras.metrics.Mean(name="val_disc_loss")
        self.recon_tracker_val = keras.metrics.Mean(name="val_recon_error")
        self.kl_tracker_val = keras.metrics.Mean(name="val_KL")
        self.tc_reg_tracker_val = keras.metrics.Mean(name="val_tc_reg")

    def compile(self, vae_optimizer, disc_optimizer):
        super(factorvae, self).compile()
        self.vae_optimizer = vae_optimizer
        self.disc_optimizer = disc_optimizer

    """
    def call(self, inputs, training=False):
        inputs = tf.cast(inputs, dtype=tf.float64)
        z, z_mean, z_log = self.vae.encoder(inputs)
        x_rec = self.vae.decoder(z)
        return x_rec
    """

    def permute_dims(self, z):
        z_cols = tf.unstack(z, axis=1)
        store = []
        for z_col in z_cols:
            rand = np.random.randint(1, 63)  # denna kanske inte borde vara hårdkodad
            z_perm = tf.roll(z_col, shift=rand, axis=0)
            store.append(z_perm)
        z_permuted = tf.stack(store, axis=1)
        return z_permuted

    def generate_loss(self, data):
        z, z_mean, z_log_var = self.vae.encoder(data)
        probs_true = self.disc(z)
        z_permuted = self.permute_dims(z)
        probs_permuted = self.disc(z_permuted)
        x_rec = self.vae.decoder(z)
        bce = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        recon_error = tf.reduce_mean(bce(data, x_rec)) * 4096
        KL = 0.5 * tf.reduce_mean(tf.exp(z_log_var) + tf.square(z_mean) - 1. - z_log_var)
        tc_reg = self.gamma * tf.math.reduce_mean((tf.math.subtract(tf.math.log(probs_true[:, 0]), tf.math.log(probs_true[:, 1]))))
        tot_reg = tf.add(KL, tc_reg)
        total_loss = tf.add(recon_error, tot_reg)#recon_error + KL + tc_reg
        disc_loss = -0.5 * tf.add(tf.reduce_mean(tf.math.log(probs_true[:, 0])),
                           tf.reduce_mean(tf.math.log(probs_permuted[:, 1])))
        return total_loss, disc_loss, recon_error, KL, tc_reg

    def train_step(self, data):
        data = data[0]
        #with tf.GradientTape(persistent=True) as tape:
        with tf.GradientTape() as vae_tape, tf.GradientTape() as disc_tape:
            total_loss, disc_loss, recon_error, KL, tc_reg = self.generate_loss(data)
        grads_vae = vae_tape.gradient(total_loss, self.vae.trainable_weights)
        self.vae_optimizer.apply_gradients(zip(grads_vae, self.vae.trainable_weights))
        grads_disc = disc_tape.gradient(disc_loss, self.disc.trainable_weights)
        self.disc_optimizer.apply_gradients(zip(grads_disc, self.disc.trainable_weights))
        self.loss_tracker.update_state(total_loss)
        self.disc_tracker.update_state(disc_loss)
        self.recon_tracker.update_state(recon_error)
        self.kl_tracker.update_state(KL)
        self.tc_reg_tracker.update_state(tc_reg)
        return {"loss": self.loss_tracker.result(), "disc_loss": self.disc_tracker.result(),
                "recon_error": self.recon_tracker.result(), "KL": self.kl_tracker.result(),
                "tc_reg": self.tc_reg_tracker.result()}

    def test_step(self, data):
        data = data[0]
        total_loss, disc_loss, recon_error, KL, tc_reg = self.generate_loss(data)
        self.loss_tracker_val.update_state(total_loss)
        self.disc_tracker_val.update_state(disc_loss)
        self.recon_tracker_val.update_state(recon_error)
        self.kl_tracker_val.update_state(KL)
        self.tc_reg_tracker_val.update_state(tc_reg)
        return {"loss": self.loss_tracker_val.result(), "disc_loss": self.disc_tracker_val.result(),
                "recon_error": self.recon_tracker_val.result(), "KL": self.kl_tracker_val.result(),
                "tc_reg": self.tc_reg_tracker_val.result()}

    @property
    def metrics(self):
        return [self.loss_tracker, self.loss_tracker_val, self.disc_tracker, self.disc_tracker_val,
                self.recon_tracker, self.recon_tracker_val, self.kl_tracker, self.kl_tracker_val,
                self.tc_reg_tracker, self.tc_reg_tracker_val]
