# coding=utf-8
"""
@author: congying
@email: wangcongyinga@gmail.com 
"""
import scipy.io
import numpy as np
from utils import ConstructLookUpTable

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
        self.permutation_lists_train = np.arange(self.train_length)
        np.random.shuffle(self.permutation_lists_train)  # random shuffle index of training data
        self.train_labels_perm = self.train_labels[self.permutation_lists_train]
        self.test_length = self.test_labels.shape[0]
        self.permutation_lists_test = np.arange(self.test_length)

        self.batch_seq = 0  # ?
        self.epochs_completed = 0  # 完成的轮数
        self.index_in_epoch = 0  # 计数每一轮的index，当index与train length一样长的时候，这一轮training完成
        self.index_in_epoch_test = 0
        self.max_seq = 0

        self.batch_up_model = ConstructLookUpTable()
        self.test_demo = self.batch_up_model.construct(self.test_lists, "test")  # 创建test的demo

        if self.layer_name != 'fc_final':
            self.feature_size = self.channel_num*14*14  # 输入是14*14大小的图片
        else:
            self.feature_size = 4096  # 当是fc_final的时候，就是进入了全连接层，feature的大小是4096

    def pretraining_batch(self, perm_batch, batch_size, platform):
        if platform == 'train':  # 如果是train的话 就construct train_lists
            current_dict = self.batch_up_model.construct(self.train_lists, platform)
        else:  # 否则的话是初始化时创建好的test_lists  ## TODO:test lists是不是可以挪到这里进行construct
            current_dict = self.test_demo
        i = 0
        temp = np.zeros(batch_size)
        while i < batch_size:
            temp[i] = None

    def gen_next_batch(self, batch_size):
        """
        下一个training batch
        :return: 
        """
        start = self.index_in_epoch
        self.index_in_epoch += batch_size
        if self.index_in_epoch > self.train_length:  # 此时epoch finish
            self.epochs_completed += 1
            # shuffle the data, 每一轮都增加其随机性, 进行随机初始化
            self.permutation_lists_train = np.arange(self.train_length)
            np.random.shuffle(self.permutation_lists_train)
            self.train_labels_perm = self.train_labels[self.permutation_lists_train]
            # start next epoch
            start = 0
            self.index_in_epoch = batch_size
            assert batch_size <= self.train_length
        end = self.index_in_epoch
        return self.pretraining_batch()


def main():
    pass


if __name__ == "__main__":
    main()
