from tensorflow import keras

class callbackclass(keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        #print(list(logs.keys()))
        print(
            "Mean loss for epoch {} is {:7.2f} "
            ",mean val_loss is {:7.2f}" ", mean acc is {:7.2f}" ", mean val_acc is {:7.2f}".format(
                epoch, logs["loss"], logs["val_loss"]
            , logs["acc"], logs["val_acc"])
        )