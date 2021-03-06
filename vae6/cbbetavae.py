from tensorflow import keras
import tensorflow as tf
import csv


class cbbeta(keras.callbacks.Callback):
    def __init__(self, path):
        super(cbbeta, self).__init__()
        self.epoch_val_loss = {}  # loss at given epoch
        self.epoch_loss = {}  # accuracy at given epoch
        self.path = path


    def on_epoch_end(self, epoch, logs=None):
        self.epoch_val_loss[epoch] = logs.get("val_loss")
        if min(self.epoch_val_loss.values()) == self.epoch_val_loss[epoch] and epoch != 0: #min(self.epoch_val_loss, key=d.get):, denna verkar typ funka
            checkpoint = tf.train.Checkpoint(vae_optimizer=self.model.optimizer,
                                             vae=self.model.vae)
            checkpoint.save(file_prefix=self.path + "/modelstore/" + "ckpt")

        if epoch % 5 == 0:
            checkpoint = tf.train.Checkpoint(vae_optimizer=self.model.optimizer,
                                             vae=self.model.vae)
            checkpoint.save(file_prefix=self.path + "/modelstore/" + "ckpt" + str(epoch))


        with open(self.path + "/history/lossvae.csv", 'a', newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([epoch, logs.get("loss"), logs.get("recon_error"), logs.get("KL"),
                             logs.get("val_loss"), logs.get("val_recon_error"), logs.get("val_KL")])

        print(
            "Mean loss for epoch {} is {:7.2f} "
            "and mean val_loss is {:7.2f}." "and mean val_KL is {:7.2f}".format(
                epoch, logs["loss"], logs["val_loss"], logs["val_KL"]
            )
        )

