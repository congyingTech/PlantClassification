# coding=utf-8
"""
@author: congying
@email: wangcongyinga@gmail.com 
@time: 2018/11/28
"""

from utils.object_utils import load_object
import scipy.io as sio
import numpy as np

func_map = {1: "CreateTrain", 2: "CreateTest"}


class BaseCreateFactory(object):
    def __init__(self, K, platform_id):
        self.K = K
        self.platform_id = platform_id
        self.origin_folder_train = "/Users/mohn/PlantData/deep-plant/?"
        self.origin_folder_test = "/Users/mohn/PlantData/deep-plant/?"
        self.media_folder_train = "/Users/mohn/PlantData/deep-plant/train_obs_media"
        self.media_folder_test = "/Users/mohn/PlantData/deep-plant/test_obs_media"

    def create_object(self):
        class_name = func_map.get(self.platform_id)
        create_cls = load_object(class_name)
        object = create_cls(self.K, self.platform_id).do_detail_work()
        return object

    def do_detail_work(self):
        raise NotImplementedError


class CreateTrain(BaseCreateFactory):
    def do_detail_work(self):
        obs_id = self.K
        media_ids = []
        mat_content = sio.loadmat(self.media_folder_train + obs_id + '.mat', mat_dtype=True)
        used = mat_content['B']  # TODO:待看这个mat里面是啥

        LL = np.arange(used.shape[1])
        np.random.shuffle(LL)

        n = 0
        while n < used.shape[1]:
            read_string = str(used[0, LL[n]])  # todo：test
            k2 = read_string.split('/')
            media_id = k2[9].split('.mat')[0].split('_')[1]
            N = [0, 1, 2]
            np.random.shuffle(N)
            if N[0] == 0:
                pkl_file = self.origin_folder_train+'512_'+media_id+'.pkl'
            elif N[0] == 1:
                pkl_file = self.origin_folder_train+'384_'+media_id+'.pkl'
            else:
                pkl_file = self.origin_folder_train+'256_'+media_id+'.pkl'
            media_ids.append(pkl_file)
            n = n + 1
        return [obs_id, media_ids]


class CreateTest(BaseCreateFactory):
    def do_detail_work(self):
        obs_id = self.K
        media_ids = []
        mat_content = sio.loadmat(self.media_folder_test+obs_id+'.mat', mat_dtype=True)
        used = mat_content['B']
        n = 0
        while n < used.shape[1]:
            read_string = str(used[0, n][0])  # todo:test数据格式
            k2 = read_string.split('/')
            media_id = k2[9].split('.mat')[0]
            pkl_file = self.origin_folder_test+media_id+'.pkl'
            media_ids.append(pkl_file)
            n += 1
        return [obs_id, media_ids]


class ConstructLookUpTable(object):

    @classmethod
    def construct(cls, obs_lists, platform_id):
        """
        construct new model list
        :return: 
        """
        step = 0
        LookupTable = []
        while step < obs_lists.shape[0]:
            K = str(int(obs_lists[step, 0]))
            LookupTable.append(BaseCreateFactory(K, platform_id).create_object())
            step += 1
        return LookupTable
