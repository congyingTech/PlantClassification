# coding=utf-8
"""
@author: congying
@email: wangcongyinga@gmail.com 
"""
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import array_ops


def matmul_test():
    """
    矩阵相乘
    :return:
    """
    output = tf.constant(np.array([[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7]]), dtype=tf.int32)
    weight = tf.constant(np.array([1, 1, 1, 1, 1, 1]).reshape(6, -1), dtype=tf.int32)
    matmul = tf.matmul(output, weight)
    with tf.Session() as sess:
        print(output.eval())
        print(weight.eval())
        print(matmul.eval())


def concat_test():
    """
    矩阵拼接
    :return:
    """
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


def reshape_test():
    """
    矩阵重塑
    :return:
    """
    a = tf.constant([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    b = tf.reshape(a, [-1, 6])  #
    with tf.Session() as sess:
        print(a)
        print(a.eval())
        print(b)
        print(b.eval())


def truncated_normal_test():
    """
    截断随机
    :return:
    """
    shape = [2, 4]
    n = tf.truncated_normal(shape, stddev=1.0)

    with tf.Session() as sess:
        print(n.eval())


def cast_test():
    """
    转换
    :return:
    """
    a = [1, 1, 3, 5, 3]
    b = tf.cast(a, tf.float32)
    with tf.Session() as sess:
        print(b.eval())


def main():
    cast_test()


if __name__ == "__main__":
    main()
