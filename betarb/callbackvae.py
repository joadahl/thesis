from tensorflow import keras
import tensorflow as tf
import os
import csv
import pickle
import keras.backend as K

class callbackvae(keras.callbacks.Callback):
    def __init__(self):
        super(callbackvae, self).__init__()
        self.epoch_val_loss = {}  # loss at given epoch
        self.epoch_loss = {}  # accuracy at given epoch
        csv_columns = ['epoch', 'loss', 'val_loss']
        with open("history/loss.csv", 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()

    def on_epoch_begin(self, epoch, logs=None):
        #keys = list(logs.keys())
        print("Start epoch {}".format(epoch))

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_loss[epoch] = logs.get("loss")
        #print(logs.get("loss"))
        self.epoch_val_loss[epoch] = logs.get("val_loss")
        with open("history/loss.csv", 'a') as csvfile:
            #    f.write(str(logs.get("val_loss")))
            # csvwriter.writerow(fields)
            writer = csv.writer(csvfile)
            # for data in [[self.epoch_loss]]:
            writer.writerow([epoch, logs.get("loss"), logs.get("val_loss")])

        if min(self.epoch_val_loss.values()) == self.epoch_val_loss[epoch] and epoch != 0: #min(self.epoch_val_loss, key=d.get):, denna verkar typ funka
            self.model.save_weights("modelstore/betavae.ckpt") #denna används för vikter
            symbolic_weights = getattr(self.model.optimizer, 'weights')
            weight_values = K.batch_get_value(symbolic_weights)
            with open('modelstore/optimizer.pkl', 'wb') as f:
                pickle.dump(weight_values, f)
            #self.model(tf.random.normal((2, 64, 64, 1), dtype=tf.float32)) #måste ha denna, annars fattar den inte
            #self.model.save('modelstore/betavae', save_format='tf') #spara modell
            #tf.keras.models.save_model(self.model, 'modelstore/betavae.hp5', save_format="h5")
            #with open("history/loss.csv", 'a') as csvfile:
            #    f.write(str(logs.get("val_loss")))
                #csvwriter.writerow(fields)
            #    writer = csv.writer(csvfile)
                #for data in [[self.epoch_loss]]:
            #    writer.writerow(self.store)

                    #writer.writerow(data)
            #    for data in logs.get("loss"):
            #        writer.writerow(data)
        print(
            "Mean loss for epoch {} is {:7.2f} "
            "and mean val_loss is {:7.2f}.".format(
                epoch, logs["loss"], logs["val_loss"]
            )
        )
        #print(self.epoch_loss)

    #def on_train_batch_begin(self, batch, logs=None):
        #keys = list(logs.keys())
    #    print("...Training: start of batch {}".format(batch))
