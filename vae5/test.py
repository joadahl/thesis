import numpy as np
from sklearn.preprocessing import normalize

a = np.array(([1,1,1], [2,4,6]))
P_ij = normalize(a, axis=1, norm='l1')
Q_ij = normalize(a, axis=0, norm='l1')
print(P_ij)
print(Q_ij)