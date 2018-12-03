# coding=utf-8
"""
@author: congying
@email: wangcongyinga@gmail.com 
@time: 2018/11/29
"""
import functools
import tensorflow as tf


def lazy_property(func):
    attribute = '_' + func.__name__
    @property
    @functools.wraps(func)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, func(self))
        return getattr(self, attribute)
    return wrapper


class VariableSequenceClassification(object):
    def __init__(self, data, target, dropout, num_hidden=200, num_layers=2):
        self.data = data
        self.target = target
        self.dropout = dropout
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.prediction = None
        self.error = None
        self.optmize = None

    @lazy_property
    def max_length(self):
        used = tf.sign(tf.reduce_max(tf.abs(self.data), axis=2))  # -1,0,1函数
        length = tf.reduce_sum(used, axis=1)
        length = tf.cast(length, tf.int32)
        max_length_num = tf.reduce_max(length)
        return max_length_num

    @lazy_property
    def length(self):
        used = tf.sign(tf.reduce_max(tf.abs(self.data), axis=2))
        length = tf.reduce_sum(used, axis=1)
        length = tf.cast(length, tf.int32)
        return length



