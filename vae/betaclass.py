import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense


class beta_score_classifier(keras.Model):
    def __init__(self):
        super(beta_score_classifier, self).__init__()
        self.layer_1 = Dense(units=5, activation="softmax") #6 is output classes
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.loss_tracker_val = keras.metrics.Mean(name="val_loss")
        self.acc_tracker = tf.keras.metrics.CategoricalAccuracy(name="acc") #denna måste implementeras så att det storas i loggen
        self.acc_tracker_val = tf.keras.metrics.CategoricalAccuracy(name="val_acc")

    def call(self, inputs):
        y_pred = self.layer_1(inputs)
        return y_pred

    def generate_loss(self, data):
        z_diff, labels = data
        reconstruction = self.call(z_diff)
        reconstruction_loss = tf.reduce_mean(
            keras.losses.binary_crossentropy(labels, reconstruction) #keras.losses.categorical_crossentropy(labels, reconstruction))
        )
        reconstruction_loss *= 5
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
        self.loss_tracker_val.update_state(loss)
        self.acc_tracker_val(data[1], reconstruction)
        return {"loss": self.loss_tracker_val.result(), "acc": self.acc_tracker_val.result()}

    @property #kan ehöva se upp med det
    def metrics(self):
    # We list our `Metric` objects here so that `reset_states()` can be
    # called automatically at the start of each epoch
    # or at the start of `evaluate()`.
    # If you don't implement this property, you have to call
    # `reset_states()` yourself at the time of your choosing.
        return [self.loss_tracker, self.acc_tracker, self.loss_tracker_val, self.acc_tracker_val]