from tensorflow import keras
import csv
import pickle
import keras.backend as K

class cbbeta(keras.callbacks.Callback):
    def __init__(self, path):
        super(cbbeta, self).__init__()
        self.epoch_val_loss = {}  # loss at given epoch
        self.epoch_loss = {}  # accuracy at given epoch
        self.path = path


    def on_epoch_end(self, epoch, logs=None):
        self.epoch_val_loss[epoch] = logs.get("val_loss")
        if min(self.epoch_val_loss.values()) == self.epoch_val_loss[epoch] and epoch != 0: #min(self.epoch_val_loss, key=d.get):, denna verkar typ funka
            self.model.save_weights(self.path + "/modelstore/betavae.ckpt") #denna används för vikter
            symbolic_weights = getattr(self.model.optimizer, 'weights')
            weight_values = K.batch_get_value(symbolic_weights)
            with open(self.path + '/modelstore/optimizer.pkl', 'wb') as f:
                pickle.dump(weight_values, f)

        if epoch % 5 == 0:
            self.model.save_weights(self.path + '/modelstore/betavae' + str(epoch) + '.ckpt')  # denna används för vikter
            symbolic_weights = getattr(self.model.optimizer, 'weights')
            weight_values = K.batch_get_value(symbolic_weights)
            with open(self.path + '/modelstore/optimizer' + str(epoch) + '.pkl', 'wb') as f:
                pickle.dump(weight_values, f)

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

