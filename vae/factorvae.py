from tensorflow import keras
#from encoder import encoder
#from decoder import decoder
from MLP import MLP
import tensorflow as tf
import numpy as np
tf.keras.backend.set_floatx('float64')
from factor import factor

class factorvae(keras.Model):
    def __init__(self, latent_dims):
        super(factorvae, self).__init__()
        self.latent_dims = latent_dims
        self.gamma = 40
        """
        self.encoder = encoder(latent_dims)
        self.decoder = decoder()
        """
        self.factor = factor(latent_dims)
        self.disc = MLP()
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.disc_tracker = keras.metrics.Mean(name="disc_loss")
        self.recon_tracker = keras.metrics.Mean(name="recon_error")
        self.kl_tracker = keras.metrics.Mean(name="KL")
        self.tc_reg_tracker = keras.metrics.Mean(name="tc_reg")
        #här börjar val
        self.loss_tracker_val = keras.metrics.Mean(name="val_loss")
        self.disc_tracker_val = keras.metrics.Mean(name="val_disc_loss")
        self.recon_tracker_val = keras.metrics.Mean(name="val_recon_error")
        self.kl_tracker_val = keras.metrics.Mean(name="val_KL")
        self.tc_reg_tracker_val = keras.metrics.Mean(name="val_tc_reg")

    def compile(self, vae_optimizer, disc_optimizer):
        super(factorvae, self).compile()
        self.vae_optimizer = vae_optimizer
        self.disc_optimizer = disc_optimizer


    #def call(self, inputs, training=False):
    #    inputs = tf.cast(inputs, dtype=tf.float64)
    #    z, z_mean, z_log_var = self.encoder(inputs)
    #    x_rec = self.decoder(z)
    #    return x_rec

    #här måste vi fylla på
    def permute_dims(self, z):
        z_cols = tf.unstack(z, axis=1)
        store = []
        for z_col in z_cols:
            rand = np.random.randint(64)  # denna kanske inte borde vara hårdkodad
            z_perm = tf.roll(z_col, shift=rand, axis=0)
            store.append(z_perm)
        z_permuted = tf.stack(store, axis=1)
        return z_permuted

    """
    def generate_loss(self, data):
        z, z_mean, z_log_var = self.encoder(data)
        z_permuted = self.permute_dims(z)
        probs_true, logits_true = self.disc(z)
        probs_permuted, logits_permuted = self.disc(z_permuted)
        x_rec = self.decoder(z)
        bce = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        recon_error = tf.reduce_mean(bce(data, x_rec)) * 4096
        KL = 0.5 * tf.reduce_mean(tf.exp(z_log_var) + tf.square(z_mean) - 1. - z_log_var)
        tc_reg = self.gamma * tf.reduce_mean(logits_true[:, 0] - logits_true[:, 1], axis=0) #nånting blir fel här
        #tc_reg = self.gamma * tf.reduce_mean(tf.math.log((probs_true[:, 0] + 0.1) / (probs_true[:, 1] + 0.1)))
        #tc_reg = self.gamma * tf.math.reduce_mean(tf.math.subtract(tf.math.log(probs_true[:, 0]), tf.math.log(probs_true[:, 1])), axis=0) # denna körde vi med
        #tc_reg = self.gamma * tf.reduce_mean(logits_true[:, 0] - logits_true[:, 1], axis=0)
        total_loss = recon_error + KL + tc_reg
        #disc_loss = -0.5 * tf.add(tf.reduce_mean(tf.math.log(probs_true[:, 0])), tf.reduce_mean(tf.math.log(probs_permuted[:, 1])))
        #disc_loss = - tf.add(0.5 * tf.reduce_mean(tf.math.log(probs_true[:, 0])), 0.5 * tf.reduce_mean(tf.math.log(probs_permuted[:, 1])))
        disc_loss = tf.add(0.5 * tf.reduce_mean(tf.math.log(probs_true[:, 0]), axis=0), 0.5 * tf.reduce_mean(tf.math.log(probs_permuted[:, 1]), axis=0))
        #disc_loss = - tf.reduce_mean(tf.math.log(probs_true[:, 0]))
        #disc_loss = - tf.add(0.5 * tf.reduce_mean(tf.math.log(probs_true[:, 0])),
        #                   0.5 * tf.reduce_mean(tf.math.log(probs_permuted[:, 1])))
        return total_loss, disc_loss, recon_error, KL, tc_reg
    """

    def generate_loss(self, data):
        z, z_mean, z_log_var = self.factor.encoder(data)
        probs_true, logits_true = self.disc(z)
        z_permuted = self.permute_dims(z)
        probs_permuted, logits_permuted = self.disc(z_permuted)
        x_rec = self.factor.decoder(z)
        bce = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        recon_error = tf.reduce_mean(bce(data, x_rec)) * 4096
        KL = 0.5 * tf.reduce_mean(tf.exp(z_log_var) + tf.square(z_mean) - 1. - z_log_var)
        #tc_reg = self.gamma * tf.reduce_mean(tf.subtract(logits_true[:, 0], logits_true[:, 1]), axis=0)  # nånting blir fel här
        tc_reg = self.gamma * tf.math.reduce_mean((tf.math.subtract(tf.math.log(probs_true[:, 0]), tf.math.log(probs_true[:, 1]))), axis=0)
        tot_reg = tf.add(KL, tc_reg)
        total_loss = tf.add(recon_error, tot_reg)#recon_error + KL + tc_reg
        #disc_loss = - 0.5 * tf.math.add(tf.reduce_mean(tf.math.log(probs_true[:, 0])),
        #                   tf.reduce_mean(tf.math.log(probs_permuted[:, 1])))
        disc_loss = - tf.add(0.5 * tf.reduce_mean(tf.math.log(probs_true[:, 0])),
                           0.5 * tf.reduce_mean(tf.math.log(probs_permuted[:, 1])))
        return total_loss, disc_loss, recon_error, KL, tc_reg

#spara allt, så vi behöver recon_error, KL, tc_reg, discloss m.m
    def train_step(self, data):
        data = data[0]
        with tf.GradientTape(persistent=True) as tape:
            total_loss, disc_loss, recon_error, KL, tc_reg = self.generate_loss(data)
        grads_vae = tape.gradient(total_loss, self.factor.trainable_weights)
        #print(grads_vae)
        self.vae_optimizer.apply_gradients(zip(grads_vae, self.factor.trainable_weights))
        grads_disc = tape.gradient(disc_loss, self.disc.trainable_weights)
        #print(grads_disc)
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

    @property  # kan ehöva se upp med det
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.loss_tracker, self.loss_tracker_val, self.disc_tracker, self.disc_tracker_val,
                self.recon_tracker, self.recon_tracker_val, self.kl_tracker, self.kl_tracker_val,
                self.tc_reg_tracker, self.tc_reg_tracker_val]
