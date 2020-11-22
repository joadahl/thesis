import numpy as np


a = np.array(([1,2], [4,5], [15, 15]))
#print(np.sum(a, axis=1))
#1 2
#4 5
b = np.array(([7,8], [10, 11], [20, 20]))
print(a[:,0])
print(b[:,0])
print(a[:, 0] - b[:, 0])