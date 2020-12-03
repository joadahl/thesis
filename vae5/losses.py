from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

#global
trained_models = [4, 6, 8, 10, 15]


def plot_loss_model_betavae(lat_dims, weight, x_axis, y_axis):
    data = np.genfromtxt("betavae" + "lat" + str(lat_dims) + "weight" + str(weight) + "/history/lossvae.csv", delimiter=",",
                         names=["epoch", "loss", "recon_error", "KL", "val_loss", "val_recon_error",
                                "val_KL"])
    plt.plot(data[x_axis], data[y_axis], label="Betavae, " + r"$D=$" + str(lat_dims) + ", " + r"$\beta=$" + str(weight))
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    # plt.title('fillin')
    plt.legend()
    plt.show()

"""
def plot_loss_model_factorvae(lat_dims, weight, x_axis, y_axis):
    data = np.genfromtxt("factorvae" + "lat" + str(lat_dims) + "weight" + str(weight) + "/history/lossvae.csv", delimiter=",",
                         names=["epoch", "loss", "disc_loss", "recon_error", "KL", "tc_reg", "val_loss",
                                "val_disc_loss",
                                "val_recon_error", "val_KL", "val_tc_reg"])
    plt.plot(data[x_axis], data[y_axis], label="Factorvae, " + r"$D=$" + str(lat_dims) + ", " + r"$\gamma=$" + str(weight))
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    # plt.title('fillin')
    plt.legend()
    plt.show()


def plot_dims_metric_models(model):
    trained_models_beta = [10]
    score_store = np.zeros((3, len(trained_models_beta)))
    if model == "betavae":
        for step, value in enumerate(trained_models_beta):
            df = pd.read_csv(model + str(value) + "/history/scores.csv", delimiter=",")
            print(np.array(df["betascore"].dropna()))
            score_store[0, step] = np.array(df["betascore"].dropna())
            score_store[1, step] = np.array(df["factorscore"].dropna())
            score_store[2, step] = np.mean(np.array(df["DCIscore"].dropna()))


    plt.plot(trained_models_beta, score_store[0], '-o', label="betascore")
    plt.plot(trained_models_beta, score_store[1], '-o', label="factorscore")
    plt.plot(trained_models_beta, score_store[2], '-o', label="dciscore") #vi saknar betavaescore på grund av tabbe
    ax = plt.gca()
    ax.set_xticks(trained_models_beta)
    ax.set_yticks(np.arange(0, 1, 0.1))  # np.arange(self.latent_dims)
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)
    ax.legend()
    plt.show()

def plot_dims_metric_models(model, weight):
    trained_models_beta = [4]
    score_store = np.zeros((3, len(trained_models_beta)))
    if model == "betavae":
        for step, value in enumerate(trained_models_beta):
            df = pd.read_csv(model + "lat" + str(value) + "weight" + str(weight) + "/history/scores.csv", delimiter=",")
            #print(np.array(df["betascore"].dropna()))
            score_store[0, step] = np.array(df["betascore"].dropna())
            score_store[1, step] = np.array(df["factorscore"].dropna())
            score_store[2, step] = np.mean(np.array(df["DCIscore"].dropna()))
    plt.plot(trained_models_beta, score_store[0], '-o', label="betascore")
    plt.plot(trained_models_beta, score_store[1], '-o', label="factorscore")
    plt.plot(trained_models_beta, score_store[2], '-o', label="dciscore")  # vi saknar betavaescore på grund av tabbe
    ax = plt.gca()
    ax.set_xticks(trained_models_beta)
    ax.set_yticks(np.arange(0, 1, 0.1))  # np.arange(self.latent_dims)
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)
    ax.legend()
    plt.show()

def hinton(model, lat_dims, weight):
    df = pd.read_csv(model + "lat" + str(lat_dims) + "weight" + str(weight) + "/history/scores.csv", delimiter=",")
    R = np.array(df["R"])
    R = R.reshape((lat_dims), 5)
    ax = plt.gca()
    ax.set_xticks(np.arange(lat_dims))
    ax.set_yticks(np.arange(5))
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)
    plt.xlabel(r"$\bf{z}$")
    plt.ylabel(r"$\bf{y}$")
    max_weight = 2 ** np.ceil(np.log(np.abs(R).max()) / np.log(2))
    ax.patch.set_facecolor('black')
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
    #ax.invert_yaxis()
    plt.show()
"""

plot_loss_model_betavae(10, 4, "epoch", "KL")
#hinton("betavae", 10, 4)
#nånting måste vara fel med DCI-score, kolla med latent dim = 10
#plot_dims_metric_models("betavae", 4)
#hinton("betavae", 10)
#plot_dims_metric_models("betavae")