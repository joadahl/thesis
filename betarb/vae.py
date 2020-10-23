from tensorflow import keras
from betaencoder import encoder
from betadecoder import decoder
import tensorflow as tf

class betavae(keras.Model):
    def __init__(self, latent_dim, beta):
        super(betavae, self).__init__()
        self.encoder = encoder(latent_dim)
        self.decoder = decoder()
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.loss_tracker_val = keras.metrics.Mean(name="val_loss") #denna la vi till
        self.beta = beta #borde ändras i nått läge
        #self.checkpoint = None
        #self.epoch_val_loss = {}  # loss at given epoch
        #self.epoch_loss = {}
        #self.epochs = {}

    def call(self, inputs, training=None):
        inputs = tf.cast(inputs, dtype=tf.float32)
        z, z_mean, z_log_var = self.encoder(inputs)
        x_rec = self.decoder(z)
        return x_rec

    def generate_loss(self, data):
        z, z_mean, z_log_var = self.encoder(data)
        x_rec = self.decoder(z)
        bce = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        recon_error = tf.reduce_mean(bce(data, x_rec)) * 4096
        KL = 0.5 * self.beta * tf.reduce_mean(tf.exp(z_log_var) + tf.square(z_mean) - 1. - z_log_var)
        loss = recon_error + KL
        return loss

    def train_step(self, data):
        data = data[0]
        with tf.GradientTape() as tape:
            loss = self.generate_loss(data)
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        data = data[0]
        loss = self.generate_loss(data)
        self.loss_tracker_val.update_state(loss)
        return {"loss": self.loss_tracker_val.result()}

    #def my_metric_fn(y_true, y_pred):
    #    squared_difference = tf.square(y_true - y_pred)
    #    return tf.reduce_mean(squared_difference, axis=-1)

    @property #kan ehöva se upp med det
    def metrics(self):
    # We list our `Metric` objects here so that `reset_states()` can be
    # called automatically at the start of each epoch
    # or at the start of `evaluate()`.
    # If you don't implement this property, you have to call
    # `reset_states()` yourself at the time of your choosing.
        return [self.loss_tracker, self.loss_tracker_val]

    #@property
    #def metrics(self):
    #    return [self.loss_tracker]
