from tensorflow import keras
import csv

class callbackbeta(keras.callbacks.Callback):
    def __init__(self, path):
        super(callbackbeta, self).__init__()
        self.path = path

    def on_epoch_end(self, epoch, logs=None):
        with open(self.path + "/history/lossbeta.csv", 'a', newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([epoch, logs.get("loss"), logs.get("acc"), logs.get("val_loss"),
                             logs.get("val_acc")])
        print(
            "Mean loss for epoch {} is {:7.2f} "
            ",mean val_loss is {:7.2f}" ", mean acc is {:7.2f}" ", mean val_acc is {:7.2f}".format(
                epoch, logs["loss"], logs["val_loss"]
            , logs["acc"], logs["val_acc"])
        )
