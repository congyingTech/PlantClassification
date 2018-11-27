# coding=utf-8
"""
@author: congying
@email: wangcongying@kuaishou.com 
"""
import scipy.io
import numpy as np

# ObservationIDs and species class PATH
# One plant has one obs id, but may be has many media ids to record different organ.

# len:13887 testing 1/3
TEST_OBS_CLASS_PATH = 'mat_data/test_obs_class.mat'
# len:13887
TEST_OBS_LIST_PATH = 'mat_data/test_obs_list.mat'
# len:27907 training 2/3
TRAIN_OBS_CLASS_PATH = 'mat_data/train_obs_class.mat'
# len:27907
TRAIN_OBS_LIST_PATH = 'mat_data/train_obs_list.mat'


class DataSet(object):
    def __init__(self, layer_name, channel_num):
        mat_data = scipy.io.loadmat(TRAIN_OBS_LIST_PATH)
        self.train_lists = mat_data.get('train_obs_list')
        mat_data = scipy.io.loadmat(TEST_OBS_LIST_PATH)
        self.test_lists = mat_data.get('test_obs_list')
        mat_data = scipy.io.loadmat(TRAIN_OBS_CLASS_PATH)
        self.train_labels = mat_data.get('train_obs_class')
        mat_data = scipy.io.loadmat(TEST_OBS_CLASS_PATH)
        self.test_labels = mat_data.get('test_obs_class')

        self.layer_name = layer_name
        self.channel_num = channel_num
        self.train_length = self.train_labels.shape[0]
        self.permutation_lists = np.arange(self.train_length)
        np.random.shuffle(self.permutation_lists)  # random shuffle index of training data


def main():
    pass


if __name__ == "__main__":
    main()
