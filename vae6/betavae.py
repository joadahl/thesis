from tensorflow import keras
import tensorflow as tf
from vae import vae

class betavae(keras.Model):
    def __init__(self, latent_dim, beta, seed):
        super(betavae, self).__init__()
        self.vae = vae(latent_dim, seed)
        self.loss_tracker = keras.metrics.Mean(name="loss") #denna la vi till
        self.kl_tracker = keras.metrics.Mean(name="kl")
        self.recon_tracker = keras.metrics.Mean(name="recon")
        self.loss_tracker_val = keras.metrics.Mean(name="val_loss")
        self.kl_tracker_val = keras.metrics.Mean(name="kl_val")
        self.recon_tracker_val = keras.metrics.Mean(name="recon_val")
        self.beta = beta

    def call(self, inputs, training=False):
        inputs = tf.cast(inputs, dtype=tf.float64)
        z, z_mean, z_log = self.vae.encoder(inputs)
        x_rec = self.vae.decoder(z)
        return x_rec

    def generate_loss(self, data):
        z, z_mean, z_log_var = self.vae.encoder(data)
        x_rec = self.vae.decoder(z)
        bce = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        recon_error = tf.reduce_mean(bce(data, x_rec)) * 4096
        KL = 0.5 * self.beta * tf.reduce_mean(tf.exp(z_log_var) + tf.square(z_mean) - 1. - z_log_var)
        loss = recon_error + KL
        return loss, recon_error, KL

    def train_step(self, data):
        data = data[0]
        with tf.GradientTape() as tape:
            loss, recon_error, KL = self.generate_loss(data)
        grads = tape.gradient(loss, self.vae.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.vae.trainable_weights))
        self.loss_tracker.update_state(loss)
        self.recon_tracker.update_state(recon_error)
        self.kl_tracker.update_state(KL)
        return {"loss": self.loss_tracker.result(), "recon_error": self.recon_tracker.result(),
                "KL": self.kl_tracker.result()}

    def test_step(self, data):
        data = data[0]
        loss, recon_error, KL = self.generate_loss(data)
        self.loss_tracker_val.update_state(loss)
        self.recon_tracker_val.update_state(recon_error)
        self.kl_tracker_val.update_state(KL)
        return {"loss": self.loss_tracker_val.result(), "recon_error": self.recon_tracker_val.result(), "KL": self.kl_tracker_val.result()}

    @property
    def metrics(self):
        return [self.loss_tracker, self.recon_tracker, self.kl_tracker, self.loss_tracker_val, self.recon_tracker_val, self.kl_tracker_val]

