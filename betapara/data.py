from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf

class dsprite:
    def __init__(self):
        self.x, self.y, self.x_train, self.x_test = self.init_data()
        self.gen_factor_size = [1, 3, 6, 40, 32, 32]

    #"User/Desktop"
    def init_data(self):
        data = np.load("dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz") #"Midgard/home/joadahl/Workplace/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
        x = data['imgs']
        y = data['latents_classes']
        x = x.reshape((x.shape[0], 64, 64, 1))
        x_train, x_test = np.split(x, [int(.6 * len(x))])
        y_train, y_test = np.split(y, [int(.6 * len(x))]) #måste lägga till y_train och y_test för batch_generation tror jag
        #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
        #return x, y, x_train[0:150], x_test[0:150], y_train[0:150], y_test[0:150]
        return x, y, tf.data.Dataset.from_tensor_slices((x_train, x_train)).batch(64), tf.data.Dataset.from_tensor_slices((x_test, x_test)).batch(64)

    #return (
    #    tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size),
    #    tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size),
    #    tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size),
    #)


    def batch_generation(self, L):  #borde ta ett helt dsprite objekt
        while True:
            x_1 = np.zeros((L, self.x.shape[1], self.x.shape[2], 1))
            x_2 = np.zeros((L, self.x.shape[1], self.x.shape[2], 1))
            random_index = np.random.randint(low=1, high=len(self.gen_factor_size))
            y = np.zeros((1, len(self.gen_factor_size)))
            y[:, random_index] = 1 #one-hot encodings
            ran_unique_feature_ind = np.random.randint(self.gen_factor_size[random_index])
            ind_fixed_feature = np.where(self.y[:, random_index] == ran_unique_feature_ind)
            x_fixed_feature = self.x[ind_fixed_feature[0]]
            rand_pick_1 = np.random.randint(x_fixed_feature.shape[0] - L)  # kan dessa raderna knäcka sig?
            rand_pick_2 = np.random.randint(x_fixed_feature.shape[0] - L)
            x_1[:L] = x_fixed_feature[rand_pick_1: rand_pick_1 + L]
            x_2[:L] = x_fixed_feature[rand_pick_2: rand_pick_2 + L]
            yield (x_1, x_2, y)

    def batch_generation_2(self, L, metric):
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
                rand_pick_2 = np.random.randint(x_fixed_feature.shape[0] - L)
                x_2[:L] = x_fixed_feature[rand_pick_2: rand_pick_2 + L]
                yield (x_1, x_2, y)
            else:
                yield (x_1, y)
