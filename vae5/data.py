import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
import pandas as pd
import os

class dsprite:
    def __init__(self):
        self.gen_factor_size = [1, 3, 6, 40, 32, 32]
        self.batch_size = 64
        self.x, self.y, self.x_train, self.x_test, self.x_train_gen, self.x_test_gen = self.init_data()

    def init_data(self):
        data = np.load("dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz") #"Midgard/home/joadahl/Workplace/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
        x = data['imgs']
        y = data['latents_classes']
        x = x.reshape((x.shape[0], 64, 64, 1))
        x, y = shuffle(x, y)
        x_train, x_test = np.split(x, [int(.7 * len(x))])
        y_train, y_test = np.split(y, [int(.7 * len(y))]) #måste lägga till y_train och y_test för batch_generation tror jag
        #x_train = x_train[0:1000] #denna bör teas bort sen
        #x_test = x_test[1000:1200]
        return x, y, x_train, x_test, tf.data.Dataset.from_tensor_slices((x_train, x_train)).batch(self.batch_size), \
               tf.data.Dataset.from_tensor_slices((x_test, x_test)).batch(self.batch_size),

    def batch_generation_beta_score(self, L):
        while True:
            y = np.random.randint(low=1, high=len(self.gen_factor_size))
            ran_unique_feature_ind = np.random.randint(low=0, high=self.gen_factor_size[y])
            ind_fixed_feature = np.where(self.y[:, y] == ran_unique_feature_ind)
            x_fixed_feature = self.x[ind_fixed_feature[0]]
            rand_pick_1 = np.random.randint(low=0, high=x_fixed_feature.shape[0], size=L)
            x_1 = x_fixed_feature[rand_pick_1]
            rand_pick_2 = np.random.randint(low=0, high=x_fixed_feature.shape[0], size=L)
            x_2 = x_fixed_feature[rand_pick_2]
            yield (x_1, x_2, y)

    def batch_generation_factor_score(self, L):
        while True:
            y = np.random.randint(low=1, high=len(self.gen_factor_size))
            ran_unique_feature_ind = np.random.randint(low=0, high=self.gen_factor_size[y])
            ind_fixed_feature = np.where(self.y[:, y] == ran_unique_feature_ind)
            x_fixed_feature = self.x[ind_fixed_feature[0]]
            rand_pick_1 = np.random.randint(low=0, high=x_fixed_feature.shape[0], size=L)
            x_1 = x_fixed_feature[rand_pick_1]
            yield (x_1, y)

    def batch_generation_dci_score(self, L):
        while True:
            rand_idx = np.random.randint(low=0, high=self.x.shape[0], size=L)
            x = self.x[rand_idx]
            y = self.y[rand_idx]
            yield (x, y)




#data = dsprite()
#print(np.unique(data.y[:,3])) #[0], [0,1,2], [0,1,2,3,4,5]
#print(data.y[3])
#data.batch_generation_beta_score(5)