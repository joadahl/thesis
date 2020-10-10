from tensorflow import keras

class callbackvae(keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        #print(list(logs.keys()))
        print(
            "Mean loss for epoch {} is {:7.2f} "
            "and mean val_loss is {:7.2f}.".format(
                epoch, logs["loss"], logs["val_loss"]
            )
        )
