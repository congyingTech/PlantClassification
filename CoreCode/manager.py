# coding=utf-8
"""
@author: congying
@email: wangcongyinga@gmail.com 
"""
import scipy.io
import os
import numpy as np
import tensorflow as tf
import pickle
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
        self.train_labels_perm = self.train_labels[self.permutation_lists_train]  # 这里只有train的label进行了随机
        self.test_length = self.test_labels.shape[0]
        self.permutation_lists_test = np.arange(self.test_length)

        self.batch_seq = 0  # ?
        self.epochs_completed = 0  # 完成的轮数
        self.index_in_epoch = 0  # 计数每一轮的index，当index与train length一样长的时候，这一轮training完成
        self.index_in_epoch_test = 0
        self.max_seq = 0

        self.batch_up_model = ConstructLookUpTable()
        self.test_demo = self.batch_up_model.construct(self.test_lists, "test")  # 创建test的demo

        if self.layer_name != 'fc7_final':
            self.feature_size = self.channel_num*14*14  # 输入是14*14大小的图片
        else:
            self.feature_size = 4096  # 当是fc_final的时候，就是进入了全连接层，feature的大小是4096

    def pretraining_batch(self, perm_batch, batch_size, indicator):
        if indicator == 1:  # 如果是train的话 就construct train_lists
            current_dict = self.batch_up_model.construct(self.train_lists, 1)
        else:  # 否则的话是初始化时创建好的test_lists  ## TODO:test lists是不是可以挪到这里进行construct
            current_dict = self.test_demo
        i = 0
        temp = np.zeros(batch_size)
        while i < batch_size:
            temp[i] = len(current_dict[perm_batch[i]][1])
            i += 1
        self.max_seq = int(np.amax(temp))
        self.batch_seq = temp

        batch = np.zeros([batch_size, self.max_seq, self.feature_size])

        i = 0
        while i < batch_size:
            media_length = len(current_dict[perm_batch[i]][1])
            j = 0
            while j < media_length:
                pkl_file = open(current_dict[perm_batch[i]][1][j], 'rb')
                output = pickle.load(pkl_file)
                pkl_file.close()
                mat_contents = output[self.layer_name]

                if self.layer_name == 'fc7_final':
                    batch[i][j][:] = mat_contents
                else:
                    batch[i][j][:] = mat_contents.reshape[self.feature_size]
                j += 1

            if indicator == 1:
                temp_arr = batch[i][0: media_length]
                np.random.shuffle(temp_arr)  # 随机一下初始化的数据
                batch[i][0:media_length] = temp_arr
            i += 1

        return batch

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
        return self.pretraining_batch(self.train_lists[start:end], batch_size, self.dense_to_one_hot(self.train_labels_perm[start:end]))

    def dense_to_one_hot(self, labels_dense, classes_length=1000):
        """

        :param labels_dense:
        :param num_classes: 物种分为1000种
        :return:
        """
        labels_dense = labels_dense.astype(int)
        labels_length = labels_dense.shape[0]
        index_offset = np.arange(labels_length) * classes_length  # (0, 1000, 2000, 3000....) 测试集长度是labels_length:13887
        labels_one_hot = np.zeros((labels_length, classes_length))  # 是一个m*n的全零二维数组 长度是13887*1000
        labels_one_hot.flat[index_offset+labels_dense.ravel()] = 1  # 这里的意思是把二维全零数组铺平成一个一维的数组，然后[x]第x值补充为1,铺平后填充值后仍然是a 13887*1000的二维数组

        temp = np.zeros((labels_one_hot.shape[0], self.max_seq, classes_length))
        i = 0
        while i < labels_one_hot.shape[0]:
            temp[i][0:int(self.batch_seq[i])] = labels_one_hot[i]
            i += 1
        return temp


def main():
    """
    开始执行分类任务
    :return:
    """
    training_iters = 10000000
    batch_size = 30
    display_size = 280
    test_num_total = 1000
    layer_name = 'fc7_final'
    channel_num = 512
    classes_length = 1000
    drop_train = 0.5
    drop_test = 1

    if layer_name != 'fc7_final':
        row_size = channel_num * 14 * 14
    else:
        row_size = 4096

    plant_data = DataSet(layer_name, channel_num)

    data = tf.placeholder("float", [None, None, row_size])
    target = tf.placeholder("float", [None, None, classes_length])
    dropout = tf.placeholder(tf.float32)

    save_dir = "models/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model = VaribleSequenceClassification(data, target, dropout)






if __name__ == "__main__":
    main()
