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
        y_train, y_test = np.split(y, [int(.6 * len(y))]) #måste lägga till y_train och y_test för batch_generation tror jag
        #x_train = x_train[0:1000] #denna bör teas bort sen
        #x_test = x_test[1000:1200]
        return x, y, x_train, x_test, tf.data.Dataset.from_tensor_slices((x_train, x_train)).batch(64), tf.data.Dataset.from_tensor_slices((x_test, x_test)).batch(64)

#har ändrat batch_size
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

    def batch_generation_factor(self, L):
        while True:
            y = np.random.randint(low=1, high=len(self.gen_factor_size))
            ran_unique_feature_ind = np.random.randint(self.gen_factor_size[y])
            ind_fixed_feature = np.where(self.y[:, y] == ran_unique_feature_ind)
            x_fixed_feature = self.x[ind_fixed_feature[0]]
            rand_pick_1 = np.random.randint(low=0, high=x_fixed_feature.shape[0], size=L)
            x_1 = x_fixed_feature[rand_pick_1]
            yield (x_1, y)

    def batch_generation_beta_2(self, L):
        while True:
            random_index = np.random.randint(low=1, high=len(self.gen_factor_size))
            #y = np.zeros((1, len(self.gen_factor_size)))
            ran_unique_feature_ind = np.random.randint(self.gen_factor_size[random_index])
            ind_fixed_feature = np.where(self.y[:, random_index] == ran_unique_feature_ind)
            x_fixed_feature = self.x[ind_fixed_feature[0]]
            rand_pick_1 = np.random.randint(low=0, high=x_fixed_feature.shape[0], size=L)
            x_1 = x_fixed_feature[rand_pick_1]
            rand_pick_2 = np.random.randint(low=0, high=x_fixed_feature.shape[0], size=L)
            x_2 = x_fixed_feature[rand_pick_2]
            yield (x_1, x_2, random_index)

    def test(self, L):
        random_index = np.random.randint(low=1, high=len(self.gen_factor_size))
        # y = np.zeros((1, len(self.gen_factor_size)))
        ran_unique_feature_ind = np.random.randint(self.gen_factor_size[random_index])
        ind_fixed_feature = np.where(self.y[:, random_index] == ran_unique_feature_ind)
        tt = ind_fixed_feature[0][0:5]
        print(random_index)
        print(self.y[tt])
        #print(ind_fixed_feature[0][0:5])
        #x_fixed_feature = self.x[ind_fixed_feature[0]]
        #rand_pick_1 = np.random.randint(low=0, high=x_fixed_feature.shape[0], size=L)
        #x_1 = x_fixed_feature[rand_pick_1]
        #rand_pick_2 = np.random.randint(low=0, high=x_fixed_feature.shape[0], size=L)
        #x_2 = x_fixed_feature[rand_pick_2]
        #return (x_1, x_2, random_index)


#data = dsprite()
#data.test(4)
#print(data.y[0:10])

