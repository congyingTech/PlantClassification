# coding=utf-8
"""
@author: congying
@email: wangcongying@kuaishou.com 
"""
import scipy.io

TEST_OBS_CLASS_PATH = '/Users/wangcongying/PlantClassification/CoreCode/mat_data/test_obs_class.mat'
# len:13887
TEST_OBS_LIST_PATH = '/Users/wangcongying/PlantClassification/CoreCode/mat_data/test_obs_list.mat'
# len:27907 training 2/3
TRAIN_OBS_CLASS_PATH = '/Users/wangcongying/PlantClassification/CoreCode/mat_data/train_obs_class.mat'
# len:27907
TRAIN_OBS_LIST_PATH = '/Users/wangcongying/PlantClassification/CoreCode/mat_data/train_obs_list.mat'


def test(path, tag_name):
    mat = scipy.io.loadmat(path)
    values = mat.get(tag_name)
    return len(values)


def testlist(path):
    mat = scipy.io.loadmat(path)
    return mat


if __name__ == "__main__":
    print test(TEST_OBS_CLASS_PATH, 'test_obs_class')
    print test(TEST_OBS_LIST_PATH, 'test_obs_list')
    print testlist(TEST_OBS_LIST_PATH)
    print testlist(TEST_OBS_CLASS_PATH)
