from sklearn.model_selection import train_test_split
import numpy as np


class dsprite:
    def __init__(self):
        self.x, self.y, self.x_train, self.x_test, self.y_train, self.y_test = self.init_data()
        self.gen_factor_size = [1, 3, 6, 40, 32, 32]

    def init_data(self):
        data = np.load("C:\\Users\\andre\\Desktop\\projectstore\\images\\dsprites_ndarray_"
                       "co1sh3sc6or40x32y32_64x64.npz")
        x = data['imgs']
        y = data['latents_classes']
        x = x.reshape((x.shape[0], 64, 64, 1))
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
        return x, y, x_train[0:150], x_test[0:150], y_train[0:150], y_test[0:150] #bör ändra här

