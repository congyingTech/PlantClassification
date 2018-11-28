# coding=utf-8
"""
@author: congying
@email: wangcongyinga@gmail.com 
@time: 2018/11/28
"""

PLATFORM_ID_MAP = {
    "train": 0,
    "test": 1
}


class BaseCreateFactory(object):
    """
    """
    pass


class CreateTrain(BaseCreateFactory):
    pass


class CreateTest(BaseCreateFactory):
    pass


class ConstructLookUpTable(object):
    def __init__(self):
        pass
    
    def construct(self, obs_lists, platform_name):
        """
        construct new model list
        :return: 
        """
        pass
