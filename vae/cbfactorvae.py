from tensorflow import keras
import csv
import pickle
import keras.backend as K

class cbfactor(keras.callbacks.Callback):
    def __init__(self, path):
        super(cbfactor, self).__init__()
        self.epoch_val_loss = {}  # loss at given epoch
        self.epoch_loss = {}  # accuracy at given epoch
        self.path = path

    def on_epoch_end(self, epoch, logs=None):
        #print(list(logs.keys()))
        self.epoch_loss[epoch] = logs.get("loss")
        self.epoch_val_loss[epoch] = logs.get("val_loss")
        if min(self.epoch_val_loss.values()) == self.epoch_val_loss[epoch] and epoch != 0: #min(self.epoch_val_loss, key=d.get):, denna verkar typ funka
            self.model.factor.save_weights(self.path + "/modelstore/factorvae.ckpt") #denna används för vikter, här la vi till factor
            symbolic_weights_vae = getattr(self.model.vae_optimizer, 'weights') #vae_optimizer
            weight_values_vae = K.batch_get_value(symbolic_weights_vae)
            with open(self.path + '/modelstore/optimizer_vae.pkl', 'wb') as f:
                pickle.dump(weight_values_vae, f)
            self.model.disc.save_weights(self.path + "/modelstore/factordisc.ckpt")
            symbolic_weights_disc = getattr(self.model.disc_optimizer, 'weights')
            weight_values_disc = K.batch_get_value(symbolic_weights_disc)
            with open(self.path + '/modelstore/optimizer_disc.pkl', 'wb') as f:
                pickle.dump(weight_values_disc, f)

        if epoch % 5 == 0:
            self.model.factor.save_weights(self.path + '/modelstore/factorvae' + str(epoch) + '.ckpt')  # denna används för vikter
            symbolic_weights = getattr(self.model.vae_optimizer, 'weights')
            weight_values = K.batch_get_value(symbolic_weights)
            with open(self.path + '/modelstore/optimizer' + str(epoch) + '.pkl', 'wb') as f:
                pickle.dump(weight_values, f)

            self.model.disc.save_weights(self.path + '/modelstore/factordisc' + str(epoch) + '.ckpt')
            symbolic_weights_disc = getattr(self.model.disc_optimizer, 'weights')
            weight_values_disc = K.batch_get_value(symbolic_weights_disc)
            with open(self.path + '/modelstore/optimizer_disc' + str(epoch) + '.pkl', 'wb') as f:
                pickle.dump(weight_values_disc, f)

        with open(self.path + "/history/lossvae.csv", 'a', newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([epoch, logs.get("loss"), logs.get("disc_loss"), logs.get("recon_error"), logs.get("KL"),
                             logs.get("tc_reg"), logs.get("val_loss"), logs.get("val_disc_loss"), logs.get("val_recon_error"),
                             logs.get("val_KL"), logs.get("val_tc_reg")])

            #writer.writerow([epoch, logs.get("loss"), logs.get("recon_error"), logs.get("KL"),
            #                 logs.get("val_loss"), logs.get("val_recon_error"), logs.get("val_KL")])


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
