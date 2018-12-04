# coding=utf-8
"""
@author: congying
@email: wangcongyinga@gmail.com 
@time: 2018/11/29
"""
import functools
import tensorflow as tf
from tensorflow.python.ops import rnn,array_ops
from tensorflow.contrib.rnn import GRUCell, DropoutWrapper, MultiRNNCell


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

    @lazy_property
    def prediction(self):
        max_length_com = tf.shape(self.target)[1]
        num_classes = int(self.target.get_shape()[2])

        with tf.variable_scope('bidirectional_rnn'):
            gru_cell_fw = GRUCell(self.num_hidden)
            gru_cell_fw = DropoutWrapper(gru_cell_fw, output_keep_prob=self.dropout)
            output_fw, _ = rnn.dynamic_rnn(gru_cell_fw, self.data, dtype=tf.float32, sequence_length=self.length)
            tf.get_variable_scope().reuse_variables()







