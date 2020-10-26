import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

class dsprite:
    def __init__(self):
        self.x, self.y, self.x_train, self.x_test, self.x_train_gen, self.x_test_gen = self.init_data()
        self.gen_factor_size = [1, 3, 6, 40, 32, 32] #har ändar här

    #"User/Desktop"
    def init_data(self):
        data = np.load("dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz") #"Midgard/home/joadahl/Workplace/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
        x = data['imgs']
        y = data['latents_classes']
        x = x.reshape((x.shape[0], 64, 64, 1))
        x, y = shuffle(x, y)
        x_train, x_test = np.split(x, [int(.6 * len(x))])
        y_train, y_test = np.split(y, [int(.6 * len(x))]) #måste lägga till y_train och y_test för batch_generation tror jag
        #x_train = x_train[0:1500] #denna bör teas bort sen
        #x_test = x_test[0:1500]
        return x, y, x_train, x_test, tf.data.Dataset.from_tensor_slices((x_train, x_train)).batch(64), tf.data.Dataset.from_tensor_slices((x_test, x_test)).batch(64)

    """
    def batch_generation(self, L, metric):
        while True:
            x_1 = np.zeros((L, self.x.shape[1], self.x.shape[2], 1))
            random_index = np.random.randint(low=1, high=len(self.gen_factor_size))
            y = np.zeros((1, len(self.gen_factor_size)))
            y[:, random_index] = 1  # one-hot encodings
            ran_unique_feature_ind = np.random.randint(self.gen_factor_size[random_index])
            ind_fixed_feature = np.where(self.y[:, random_index] == ran_unique_feature_ind)
            x_fixed_feature = self.x[ind_fixed_feature[0]]
            rand_pick_1 = np.random.randint(x_fixed_feature.shape[0] - L)  # kan dessa raderna knäcka sig?
            x_1[:L] = x_fixed_feature[rand_pick_1: rand_pick_1 + L]
            if metric == "beta":
                x_2 = np.zeros((L, self.x.shape[1], self.x.shape[2], 1))
                rand_pick_2 = np.random.randint(x_fixed_feature.shape[0] - L) #kanske måste ändra denna så att den är mer random, just nu är det en klump
                x_2[:L] = x_fixed_feature[rand_pick_2: rand_pick_2 + L]
                yield (x_1, x_2, y)
            else:
                yield (x_1, y)
    """

    def batch_generation(self, L, metric):
        while True:
            #x_1 = np.zeros((L, self.x.shape[1], self.x.shape[2], 1))
            random_index = np.random.randint(low=1, high=len(self.gen_factor_size))
            y = np.zeros((1, len(self.gen_factor_size)))
            y[:, random_index] = 1  # one-hot encodings
            ran_unique_feature_ind = np.random.randint(self.gen_factor_size[random_index])
            ind_fixed_feature = np.where(self.y[:, random_index] == ran_unique_feature_ind)
            x_fixed_feature = self.x[ind_fixed_feature[0]]
            rand_pick_1 = np.random.randint(low=0, high=x_fixed_feature.shape[0], size=L)  # kan dessa raderna knäcka sig?
            x_1 = x_fixed_feature[rand_pick_1]
            if metric == "beta":
                #x_2 = np.zeros((L, self.x.shape[1], self.x.shape[2], 1))
                rand_pick_2 = np.random.randint(low=0, high=x_fixed_feature.shape[0], size=L) #kanske måste ändra denna så att den är mer random, just nu är det en klump
                x_2 = x_fixed_feature[rand_pick_2]
                yield (x_1, x_2, y)
            else:
                yield (x_1, y)


    def batch_generation_dci(self):
        while True:
            random_index = np.random.randint(low=1, high=len(self.gen_factor_size))
            ran_unique_feature_ind = np.random.randint(self.gen_factor_size[random_index])
            ind_fixed_feature = np.where(self.y[:, random_index] == ran_unique_feature_ind)
            y_fixed_feature = self.y[ind_fixed_feature[0]]
            x_fixed_feature = self.x[ind_fixed_feature[0]]
            ran_values = np.random.randint(low=0, high=y_fixed_feature.shape[0], size=1)
            y = y_fixed_feature[ran_values]
            x_1 = x_fixed_feature[ran_values]
            x_1 = x_1.reshape((1, 64, 64, 1))
            yield (x_1, y)

    """

    def batch_generation_dci(self):
        while True:
            # x_1 = np.zeros((1, self.x.shape[1], self.x.shape[2], 1))
            random_index = np.random.randint(low=0, high=len(self.gen_factor_size))
            y = np.zeros((1, len(self.gen_factor_size)))
            y[:, random_index] = 1
            ran_unique_feature_ind = np.random.randint(self.gen_factor_size[random_index])
            ind_fixed_feature = np.where(self.y[:, random_index] == ran_unique_feature_ind)
            x_fixed_feature = self.x[ind_fixed_feature[0]]
            rand_pick_1 = np.random.randint(x_fixed_feature.shape[0])
            x_1 = x_fixed_feature[rand_pick_1]
            x_1 = x_1.reshape((1, 64, 64, 1))
            yield (x_1, y)

    """

    def batch_generation_beta(self, L):
        while True:
            random_index = np.random.randint(low=1, high=len(self.gen_factor_size))
            y = np.zeros((1, len(self.gen_factor_size)))
            y[:, random_index] = 1
            ran_unique_feature_ind = np.random.randint(self.gen_factor_size[random_index])
            ind_fixed_feature = np.where(self.y[:, random_index] == ran_unique_feature_ind)
            x_fixed_feature = self.x[ind_fixed_feature[0]]
            rand_pick_1 = np.random.randint(low=0, high=x_fixed_feature.shape[0], size=L)
            x_1 = x_fixed_feature[rand_pick_1]
            rand_pick_2 = np.random.randint(low=0, high=x_fixed_feature.shape[0], size=L)
            x_2 = x_fixed_feature[rand_pick_2]
            yield (x_1, x_2, y)



#data = dsprite()
#data.dci_real()
#det kan vara så att DCI ska ha riktiga labels och inte one-hots över bla.bla