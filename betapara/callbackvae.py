from tensorflow import keras

class callbackvae(keras.callbacks.Callback):

    def on_epoch_begin(self, epoch, logs=None):
        #keys = list(logs.keys())
        print("Start epoch {}".format(epoch))

    def on_epoch_end(self, epoch, logs=None):
        #print(list(logs.keys()))
        print(
            "Mean loss for epoch {} is {:7.2f} "
            "and mean val_loss is {:7.2f}.".format(
                epoch, logs["loss"], logs["val_loss"]
            )
        )

    #def on_train_batch_begin(self, batch, logs=None):
        #keys = list(logs.keys())
    #    print("...Training: start of batch {}".format(batch))
