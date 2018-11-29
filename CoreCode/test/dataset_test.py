# coding=utf-8
"""
@author: congying
@email: wangcongyinga@gmail.com
"""
import scipy.io
import os

TEST_OBS_CLASS_PATH = os.path.abspath(os.path.dirname(os.getcwd()))+'/mat_data/test_obs_class.mat'
# len:13887
TEST_OBS_LIST_PATH = os.path.abspath(os.path.dirname(os.getcwd()))+'/mat_data/test_obs_list.mat'
# len:27907 training 2/3
TRAIN_OBS_CLASS_PATH = os.path.abspath(os.path.dirname(os.getcwd()))+'/mat_data/train_obs_class.mat'
# len:27907
TRAIN_OBS_LIST_PATH = os.path.abspath(os.path.dirname(os.getcwd()))+'/mat_data/train_obs_list.mat'


def test(path, tag_name):
    mat = scipy.io.loadmat(path)
    values = mat.get(tag_name)
    return len(values)


def testlist(path):
    mat = scipy.io.loadmat(path)
    return mat


def testmat(path, tag_name):
    mat = scipy.io.loadmat(path)
    values = mat.get(tag_name)
    print(values.shape[0])
    print(values.shape[1])
    print(type(values))
    return values[10, 0]


class TT(object):
    def __init__(self):
        self.batch = [1, 2, 3, 4, 5]

    @property
    def pretraining_batch(self):
        return self.batch


if __name__ == "__main__":
    # print(test(TEST_OBS_CLASS_PATH, 'test_obs_class'))
    # print(test(TEST_OBS_LIST_PATH, 'test_obs_list'))
    print(testmat(TEST_OBS_CLASS_PATH, 'test_obs_class'))
    # print(testlist(TEST_OBS_LIST_PATH))
    # print(testlist(TEST_OBS_CLASS_PATH))
    print(TT().pretraining_batch)

