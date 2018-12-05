# coding=utf-8
"""
@author: congying
@email: wangcongyinga@gmail.com 
@time: 2018/12/5
"""
import tensorflow as tf

shape = [2, 4]
n = tf.random.truncated_normal(shape, stddev=1.0)

with tf.Session() as sess:
    print(n.eval())

# demo data
# [[-0.4643322 -1.275165   0.6510005  1.2381268]
#  [-1.270426  -1.7889445  0.9864413  1.0746601]]

# [[ 0.85359126 -1.1476594   1.6714956  -0.77176553]
#  [-0.7453271  -1.2064893   0.06317201  0.47094908]]
