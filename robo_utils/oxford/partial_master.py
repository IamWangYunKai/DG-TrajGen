import carla_utils as cu

import numpy as np
import os
from os.path import join
import time

from .partial_augment import PartialDatasetAugment
from .process import ProcessUtils


class PartialDatasetMaster(object):
    def __init__(self, param, data_index=None):
        self.param = param
        self.basedir = param.basedir
        # self.process = ProcessUtils(param.models_dir, param.extrinsics_dir)
        self.process = ProcessUtils(join(param.utils_basedir, 'models'), join(param.utils_basedir, 'extrinsics'))

        self.num_costmap = param.net.num_costmap
        self.num_trajectory = param.net.num_trajectory

        partial_dataset_names = []
        for basedir in self.basedir:
            partial_dataset_names.extend([(basedir, name) for name in os.listdir(basedir)])
        if data_index is not None:
            partial_dataset_names = [partial_dataset_names[data_index % len(partial_dataset_names)]]

        t1 = time.time()
        self.partial_datasets = [PartialDatasetAugment(param, join(basedir, name), self.process) for (basedir, name) in partial_dataset_names]
        t2 = time.time()
        print('[PartialDatasetMaster] init dataset time: ', t2-t1)


        '''train'''
        split_path = join(param.utils_basedir, 'split')
        cu.system.mkdir(split_path)
        try:
            straight = np.loadtxt(join(split_path, 'train_s.txt'), delimiter=' ', usecols=[], dtype=np.int64)
            left     = np.loadtxt(join(split_path, 'train_l.txt'), delimiter=' ', usecols=[], dtype=np.int64)
            right    = np.loadtxt(join(split_path, 'train_r.txt'), delimiter=' ', usecols=[], dtype=np.int64)
            self.train_straight, self.train_left, self.train_right = list(straight), list(left), list(right)
            self.train_key_list = self.train_straight + self.train_left + self.train_right
        except IOError:
            t1 = time.time()
            key_list = self.generate_valid_key(list(range(7)))
            straight, left, right = self.generate_type_split(key_list)
            self.train_straight, self.train_left, self.train_right = straight, left * (int((len(straight)+1)/(len(left)+1))+1), right * (int((len(straight)+1)/(len(right)+1))+1)
            self.train_key_list = self.train_straight + self.train_left + self.train_right

            np.savetxt(join(split_path, 'train_s.txt'), np.array(self.train_straight), delimiter=' ', fmt='%d')
            np.savetxt(join(split_path, 'train_l.txt'), np.array(self.train_left    ), delimiter=' ', fmt='%d')
            np.savetxt(join(split_path, 'train_r.txt'), np.array(self.train_right   ), delimiter=' ', fmt='%d')

            t2 = time.time()
            print('generate train time: ', t2-t1)


        '''eval'''
        try:
            straight = np.loadtxt(join(split_path, 'eval_s.txt'), delimiter=' ', usecols=[], dtype=np.int64)
            left     = np.loadtxt(join(split_path, 'eval_l.txt'), delimiter=' ', usecols=[], dtype=np.int64)
            right    = np.loadtxt(join(split_path, 'eval_r.txt'), delimiter=' ', usecols=[], dtype=np.int64)
            self.eval_straight, self.eval_left, self.eval_right = list(straight), list(left), list(right)
            self.eval_key_list = self.eval_straight + self.eval_left + self.eval_right
        except IOError:
            t1 = time.time()
            key_list = self.generate_valid_key([7])
            straight, left, right = self.generate_type_split(key_list)
            self.eval_straight, self.eval_left, self.eval_right = straight, left, right
            self.eval_key_list = self.eval_straight + self.eval_left + self.eval_right

            np.savetxt(join(split_path, 'eval_s.txt'), np.array(self.eval_straight), delimiter=' ', fmt='%d')
            np.savetxt(join(split_path, 'eval_l.txt'), np.array(self.eval_left    ), delimiter=' ', fmt='%d')
            np.savetxt(join(split_path, 'eval_r.txt'), np.array(self.eval_right   ), delimiter=' ', fmt='%d')

            t2 = time.time()
            print('generate eval time: ', t2-t1)

        
        '''test'''
        try:
            straight = np.loadtxt(join(split_path, 'test_s.txt'), delimiter=' ', usecols=[], dtype=np.int64)
            left     = np.loadtxt(join(split_path, 'test_l.txt'), delimiter=' ', usecols=[], dtype=np.int64)
            right    = np.loadtxt(join(split_path, 'test_r.txt'), delimiter=' ', usecols=[], dtype=np.int64)
            self.test_straight, self.test_left, self.test_right = list(straight), list(left), list(right)
            self.test_key_list = self.test_straight + self.test_left + self.test_right
        except IOError:
            t1 = time.time()
            key_list = self.generate_valid_key([8,9])
            straight, left, right = self.generate_type_split(key_list)
            self.test_straight, self.test_left, self.test_right = straight, left, right
            self.test_key_list = self.test_straight + self.test_left + self.test_right

            np.savetxt(join(split_path, 'test_s.txt'), np.array(self.test_straight), delimiter=' ', fmt='%d')
            np.savetxt(join(split_path, 'test_l.txt'), np.array(self.test_left    ), delimiter=' ', fmt='%d')
            np.savetxt(join(split_path, 'test_r.txt'), np.array(self.test_right   ), delimiter=' ', fmt='%d')

            t2 = time.time()
            print('generate test time: ', t2-t1)
        
        return
    
    def __len__(self):
        return len(self.partial_datasets)
    def __iter__(self):
        for dataset in self.partial_datasets:
            yield dataset
    def __getitem__(self, index):
        try:
            return self.partial_datasets[index]
        except IndexError as e:
            raise e

    def generate_valid_key(self, dataset_indexs):
        key_list = []
        for key in dataset_indexs:
            dataset = self.partial_datasets[key]
            num_valid = max(0, dataset.length_valid-(self.num_costmap+self.num_trajectory-1)+1)
            if num_valid > 0:
                index_list = list(range(self.num_costmap-1, self.num_costmap+num_valid-1))
                key_list.extend([(key, i) for i in index_list])
        return key_list
    
    def generate_type_split(self, key_list):
        straight, left, right = [], [], []
        for key, index in key_list:
            dataset = self.partial_datasets[key]
            traj_type = dataset.get_trajectory_type(dataset.ref_timestamp_array[index])
            if traj_type == 0:
                straight.append((key, index))
            elif traj_type == 1:
                left.append((key, index))
            else:
                right.append((key, index))
        return straight, left, right
        
    def get_costmaps(self, dataset, index, mode='L'):
        indexs = list(range(index-self.num_costmap, index))
        # print('indexs', indexs)
        if len(indexs) > 1: indexs = indexs[1::2]
        return [dataset.get_costmap(dataset.ref_timestamp_array[i], mode=mode) for i in indexs]
    
    def get_images(self, dataset, index, crop=True):
        indexs = list(range(index-self.num_costmap, index))
        indexs = indexs[1::2]
        return [dataset.get_image(dataset.ref_timestamp_array[i], crop) for i in indexs]

    def get_trajectory(self, dataset: PartialDatasetAugment, index):
        timestamps = dataset.ref_timestamp_array[index:index+self.num_trajectory]

        poses = dataset.generate_pose_array(timestamps[0], self.num_trajectory)
        x, y = poses[0,:],poses[1,:]
        yaw = dataset.get_yaw(timestamps[0])

        vxs, vys = [], []
        for ref_timestamp in timestamps:
            vx, vy = dataset.get_velocity(ref_timestamp)
            vxs.append(vx)
            vys.append(vy)
        vx_array, vy_array = np.array(vxs), np.array(vys)

        R = cu.basic.RotationMatrix2D(-yaw)
        vxy = np.dot(R, np.vstack((vx_array, vy_array)))
        vx_array, vy_array = vxy[0,:], vxy[1,:]
        
        times = timestamps - timestamps[0]
        times = times * 1e-6
        return times, x, y, vx_array, vy_array
