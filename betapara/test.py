import tensorflow as tf
import numpy as np

#with tf.device('/gpu:0'):
#    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
#    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
#    c = tf.matmul(a, b)
#with tf.Session() as sess:
#    print(c)
a = np.zeros((10,2))
a[0][1] = 1
x_train, x_test = np.split(a, [int(.6*len(a))])
print(x_train)
print(x_test)
#a = np.zeros((10,2))
