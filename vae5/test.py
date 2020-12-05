import numpy as np
from sklearn.utils import shuffle
import random as r
r.seed(10)
a = np.array(([1,1,1], [2,2,2]))

b = np.sum(a, axis=1)
c = np.sum(a, axis=0)
print(b)
print(c)