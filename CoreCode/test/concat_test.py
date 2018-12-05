# coding=utf-8
"""
@author: congying
@email: wangcongyinga@gmail.com 
@time: 2018/12/5
"""

from tensorflow.python.ops import array_ops
import tensorflow as tf

a = tf.constant([[1, 12, 8, 6], [3, 4, 6, 7]])  # shape (2,4)
b = tf.constant([[10, 20, 6, 88], [30, 40, 7, 8]])  # shape (2,4)
c = tf.constant([[10, 20, 6, 88, 99], [30, 40, 7, 8, 15]])  # shape (2,5)
d = tf.constant([[10, 20, 6, 88], [30, 40, 7, 8], [30, 40, 7, 8]])  # shape (3,4)

nn = tf.concat([a, d], 0)  # 按第一纬度（行）进行连接
nn_1 = tf.concat([a, c], 1)  # 按第二维度（列）进行concat
mn = array_ops.concat([a, d], 0)
mn_1 = array_ops.concat([a, c], 1)

with tf.Session() as sess:
    print(nn)
    print(nn.eval())
    print(nn_1)
    print(nn_1.eval())
    print(mn)
    print(mn.eval())
    print(mn_1)
    print(mn_1.eval())

#######  Data ############

# Tensor("concat:0", shape=(5, 4), dtype=int32)
# [[ 1 12  8  6]
#  [ 3  4  6  7]
#  [10 20  6 88]
#  [30 40  7  8]
#  [30 40  7  8]]
# Tensor("concat_1:0", shape=(2, 9), dtype=int32)
# [[ 1 12  8  6 10 20  6 88 99]
#  [ 3  4  6  7 30 40  7  8 15]]
# Tensor("concat_2:0", shape=(5, 4), dtype=int32)
# [[ 1 12  8  6]
#  [ 3  4  6  7]
#  [10 20  6 88]
#  [30 40  7  8]
#  [30 40  7  8]]
# Tensor("concat_3:0", shape=(2, 9), dtype=int32)
# [[ 1 12  8  6 10 20  6 88 99]
#  [ 3  4  6  7 30 40  7  8 15]]
