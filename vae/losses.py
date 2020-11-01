from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
"""
class losses:
    def __init__(self, model, latent_dims):
        self.model = model
        self.latent_dims = latent_dims
        self.path = model + str(latent_dims)
        self.trained_models_beta = [10]
"""

def plot_loss_model(model, lat_dims, x_axis, y_axis):
    if model == "betavae":
        try:
            data = np.genfromtxt(model + str(lat_dims) + "/history/lossvae.csv", delimiter=",",
                                 names=["epoch", "loss", "recon_error", "KL", "val_loss", "val_recon_error",
                                        "val_KL"])
        except:
            print("no such data")
    if model == "factorvae":
        try:
            data = np.genfromtxt(model + str(lat_dims) + "/history/lossvae.csv", delimiter=",",
                                 names=["epoch", "loss", "disc_loss", "recon_error", "KL", "tc_reg", "val_loss",
                                        "val_disc_loss",
                                        "val_recon_error", "val_KL", "val_tc_reg"])
        except:
            print("no such data")
    plt.plot(data[x_axis], data[y_axis], label=y_axis)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    # plt.title('fillin')
    plt.legend()
    plt.show()

def hinton(model, lat_dims): #model ska ges i index
    df = pd.read_csv(model + str(lat_dims) + "/history/scores.csv", delimiter=",")
    R = np.array(df["R"])
    R = R.reshape((lat_dims), 5)
    ax = plt.gca()
    ax.set_xticks(np.arange(lat_dims))
    ax.set_yticks(np.arange(5))  # np.arange(self.latent_dims)
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)
    plt.xlabel(r"$\bf{z}$")
    plt.ylabel(r"$\bf{y}$")
    max_weight = 2 ** np.ceil(np.log(np.abs(R).max()) / np.log(2))
    ax.patch.set_facecolor('gray')
    ax.set_aspect('equal', 'box')

    for (x, y), w in np.ndenumerate(R):
        if w > 0:
            color = 'white'
        else:
            color = 'black'
        size = np.sqrt(np.abs(w) / max_weight)
        rect = plt.Rectangle((x - size / 2, y - size / 2), size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    ax.autoscale_view()
    ax.invert_yaxis()
    plt.show()

def plot_dims_metric_models(model):
    trained_models_beta = [10]
    if model == "betavae":
        for step, value in enumerate(trained_models_beta):
            #try:
            df = pd.read_csv(model + str(value) + "/history/scores.csv", delimiter=",")
            betascore = np.array(df["betascore"])
            factorscore = np.array(df["factorscore"])
            DCIscore = np.mean(np.array(df["DCIscore"].dropna()))
            #scores_store[0, step] = score[0][np.logical_not(np.isnan(score[0]))]
            #scores_store[1, step] = score[1][np.logical_not(np.isnan(score[1]))]
            #scores_store[2, step] = np.mean(score[2][np.logical_not(np.isnan(score[2]))])
            #except:
                #print("file missing")
    """
    plt.plot(self.trained_models, scores_store[0], '-o', label="beta_score")
    plt.plot(self.trained_models, scores_store[1], '-o', label="factor_score")
    plt.plot(self.trained_models, scores_store[2], '-o', label="dci_score")
    ax = plt.gca()
    ax.set_xticks(self.trained_models)
    ax.set_yticks(np.arange(0, 1, 0.1))  # np.arange(self.latent_dims)
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)
    plt.show()
    """

#målsättning måndag: att kunna visa att några av dem här fungerar

#l.hinton()
#plot_loss_model("betavae", 10, "epoch", "val_recon_error")
hinton("betavae", 10)