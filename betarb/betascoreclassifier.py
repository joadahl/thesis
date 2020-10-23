import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense


class beta_score_classifier(keras.Model):
    def __init__(self):
        super(beta_score_classifier, self).__init__()
        self.layer_1 = Dense(units=6, activation="softmax") #6 is output classes
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.acc_tracker = tf.keras.metrics.CategoricalAccuracy(name="acc") #denna måste implementeras så att det storas i loggen

    def call(self, inputs):
        y_pred = self.layer_1(inputs)
        return y_pred

    def generate_loss(self, data):
        z_diff, labels = data
        reconstruction = self.call(z_diff)
        reconstruction_loss = tf.reduce_mean(
            keras.losses.categorical_crossentropy(labels, reconstruction)
        )
        reconstruction_loss *= 6
        return reconstruction_loss, reconstruction

    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss, reconstruction = self.generate_loss(data)
        grads = tape.gradient(loss, self.trainable_weights)
        self.loss_tracker.update_state(loss)
        self.acc_tracker(data[1], reconstruction)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {"loss": self.loss_tracker.result(), "acc": self.acc_tracker.result()}

    def test_step(self, data):
        loss, reconstruction = self.generate_loss(data)
        self.loss_tracker.update_state(loss)
        self.acc_tracker(data[1], reconstruction)
        return {"loss": self.loss_tracker.result(), "acc": self.acc_tracker.result()}

