# coding=utf-8
"""
@author: congying
@email: wangcongyinga@gmail.com 
@time: 2018/11/29
"""
import math
import functools
import tensorflow as tf
from tensorflow.python.ops import rnn, array_ops
from tensorflow.contrib.rnn import GRUCell, DropoutWrapper, MultiRNNCell
import numpy as np

# 定义一个lazy的装饰器


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
    pass
