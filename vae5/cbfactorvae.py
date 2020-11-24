from tensorflow import keras
import csv
import tensorflow as tf
import os

class cbfactor(keras.callbacks.Callback):
    def __init__(self, path):
        super(cbfactor, self).__init__()
        self.epoch_val_loss = {}  # loss at given epoch
        self.epoch_loss = {}  # accuracy at given epoch
        self.path = path
        self.checkpoint_prefix = os.path.join(self.path + "\modelstore", "ckpt")

    def on_epoch_end(self, epoch, logs=None):
        #print(list(logs.keys()))
        self.epoch_loss[epoch] = logs.get("loss")
        self.epoch_val_loss[epoch] = logs.get("val_loss")
        if min(self.epoch_val_loss.values()) == self.epoch_val_loss[epoch] and epoch != 0: #min(self.epoch_val_loss, key=d.get):, denna verkar typ funka
            checkpoint = tf.train.Checkpoint(vae_optimizer=self.model.vae_optimizer,
                                             discriminator_optimizer=self.model.disc_optimizer,
                                             discriminator=self.model.disc,
                                             vae=self.model.vae
                                             )
            checkpoint.save(file_prefix=self.checkpoint_prefix)

        if epoch % 5 == 0:
            checkpoint = tf.train.Checkpoint(vae_optimizer=self.model.vae_optimizer,
                                             discriminator_optimizer=self.model.disc_optimizer,
                                             discriminator=self.model.disc,
                                             vae=self.model.vae
                                             )
            checkpoint.save(file_prefix=self.checkpoint_prefix + str(epoch))

        with open(self.path + "/history/lossvae.csv", 'a', newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([epoch, logs.get("loss"), logs.get("disc_loss"), logs.get("recon_error"), logs.get("KL"),
                             logs.get("tc_reg"), logs.get("val_loss"), logs.get("val_disc_loss"), logs.get("val_recon_error"),
                             logs.get("val_KL"), logs.get("val_tc_reg")])



        print(
            "Mean loss for epoch {} is {:7.2f} "
            "and mean val_loss is {:7.2f}." "and tc_reg is {:7.2f}" "and KL is {:7.2f}".format(
                epoch, logs["loss"], logs["val_loss"], logs["tc_reg"], logs["KL"]
            )
        )

    def on_train_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print(
            "Mean loss for batch {} is {:7.2f} "
            "and tc_reg is {:7.2f}" "and KL is {:7.2f}" "and disc_loss is {:7.2f}".format(
                batch, logs["recon_error"], logs["tc_reg"], logs["KL"], logs["disc_loss"]
            )
        )
