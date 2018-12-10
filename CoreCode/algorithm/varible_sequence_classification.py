# coding=utf-8
"""
@author: congying
@email: wangcongyinga@gmail.com 
@time: 2018/11/29
"""
import functools
import math
import tensorflow as tf
from tensorflow.python.ops import rnn, array_ops
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

    @staticmethod
    def weight_and_bias(in_size, out_size):
        #  todo:把stddev这里的参数改一下，会不会有变化？
        weight = tf.truncated_normal([in_size, out_size], stddev=1.0/math.sqrt(float(in_size*2)))
        bias = tf.zeros([out_size])
        return tf.Variable(weight), tf.Variable(bias)

    @lazy_property
    def prediction(self):
        max_length_com = tf.shape(self.target)[1]
        num_classes = int(self.target.get_shape()[2])

        with tf.variable_scope('bidirectional_rnn'):

            # 正向rnn
            gru_cell_fw = GRUCell(self.num_hidden)
            gru_cell_fw = DropoutWrapper(gru_cell_fw, output_keep_prob=self.dropout)
            output_fw, _ = rnn.dynamic_rnn(gru_cell_fw, self.data, dtype=tf.float32, sequence_length=self.length)
            tf.get_variable_scope().reuse_variables()

            # 反向(因为是双向rnn，所以还有一个反向的过程)
            data_reverse = array_ops.reverse_sequence(
                input=self.data, seq_lengths=self.length,
                seq_dim=1, batch_dim=0)
            gru_cell_re = GRUCell(self.num_hidden)
            gru_cell_re = DropoutWrapper(gru_cell_re, output_keep_prob=self.dropout)
            tmp, _ = rnn.dynamic_rnn(gru_cell_re, data_reverse, dtype=tf.float32, sequence_length=self.length)

            output_re = array_ops.reverse_sequence(  # output的结果也要进行一个反向操作
                input=tmp, seq_lengths=self.length,
                seq_dim=1, batch_dim=0)

            # 双向rnn拼接起来，然后进行softmax
            output = tf.concat(axis=2, values=[output_fw, output_re])

            weight, bias = self.weight_and_bias(2*self.num_hidden, num_classes)  # 因为是双向的，所以要*2
            output = tf.reshape(output, [-1, 2*self.num_hidden])
            prediction = tf.nn.softmax(tf.matmul(output, weight)+bias)
            self.regularizer = tf.nn.l2_loss(weight)   # todo:总结，罚项和损失函数-》l1，l2

            prediction = tf.reshape(prediction, [-1, max_length_com, num_classes])

            return prediction

    @lazy_property
    def cost(self):

        # 计算每一次的交叉熵, -y*log(f(x))
        cross_entropy = self.target*tf.log(self.prediction)
        cross_entropy = -tf.reduce_sum(cross_entropy, axis=2)
        mask = tf.sign(tf.reduce_max(tf.abs(self.target), axis=2))
        cross_entropy *= mask

        cross_entropy = tf.reduce_sum(cross_entropy, axis=1)
        cross_entropy /= tf.cast(self.length, tf.float32)
        tf.summary.scalar("cost", cross_entropy)  # todo:看一下summary到底是个啥子东东？？？
        return cross_entropy

    @lazy_property
    def error(self):
        mistakes = tf.equal(
            tf.argmax(self.target, 2), tf.argmax(self.prediction, 2))
        mistakes = tf.cast(mistakes, tf.float32)  # true -> 1, false -> 0
        mask = tf.sign(tf.reduce_max(tf.abs(self.target), axis=2))  # shape = (batch_size, max_seg)
        mistakes *= mask
        # Average over actual sequence lengths.
        mistakes = tf.reduce_sum(mistakes, axis=1)  # shape = (batch_size,1)
        mistakes /= tf.cast(self.length, tf.float32)
        accuracy = tf.reduce_mean(mistakes)  # shape = (1,1)
        tf.summary.scalar("Accuracy", accuracy)

        return accuracy

    @lazy_property
    def optmize(self):
        learning_rate = 0.0001
        beta = 0.0001

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-4)
        loss = tf.reduce_mean(self.cost + beta*self.regularizer)
        return optimizer.minimize(loss)
