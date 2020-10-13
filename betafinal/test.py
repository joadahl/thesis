import csv
import numpy as np
#score = 0.7
#with open('beta_score.csv', 'a') as f:
#    f.write("beta_score:" + str(score))
#    f.write("\n")



x = np.arange(10)
print("Original array:")
print(x)
np.random.shuffle(x)
n = 3
print (x[np.argsort(x)[-n:]])