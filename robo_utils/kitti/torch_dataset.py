import carla_utils as cu

import os
from os.path import join
import glob
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from .raw_master import RawMaster, SceneType
from .raw import Raw


def segment_dataset(length):
    ''' train : eval : test = 7:1:2 '''
    a, b = round(length * 0.7), round(length * 0.1)
    return (0, a), (a, a+b), (a+b, length)

class BoostDataset(Dataset):
    def __init__(self, param, mode, opt=None, data_index=None):
        '''
        Load past costmap and future trajectory
        '''
        self.opt = opt
        self.mode = mode
        data_index = data_index
        raw_master = RawMaster(param.basedir)

        self.num_costmap = param.net.num_costmap
        self.num_trajectory = param.net.num_trajectory

        image_transforms = [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
        self.image_transforms = transforms.Compose(image_transforms)
                
        '''get valid'''
        self.dataset_dict = dict()
        self.key_list = []
        key_city_list, key_residential_list, key_road_list, key_campus_list = [], [], [], []
        for iii, dataset in enumerate(raw_master):
            key, scene_type = raw_master.get_dataset_scene(dataset)
            num_valid = max(0, dataset.length_valid-(self.num_costmap+self.num_trajectory-1)+1)
            if num_valid > 0:
                self.dataset_dict[key] = dataset
                index_list = list(range(self.num_costmap-1, self.num_costmap+num_valid-1))
                key_list = [(key, i) for i in index_list]
                self.key_list.extend(key_list)
                if scene_type == SceneType.CITY:
                    key_city_list.extend(key_list)
                elif scene_type == SceneType.RESIDENTIAL:
                    key_residential_list.extend(key_list)
                elif scene_type == SceneType.ROAD:
                    key_road_list.extend(key_list)
                else:
                    key_campus_list.extend(key_list)
        
        ''' train : eval : test = 7:1:2 '''
        (train11, train12), (eval11, eval12), (test11, test12) = segment_dataset(len(key_city_list))
        (train21, train22), (eval21, eval22), (test21, test22) = segment_dataset(len(key_residential_list))
        (train31, train32), (eval31, eval32), (test31, test32) = segment_dataset(len(key_road_list))
        (train41, train42), (eval41, eval42), (test41, test42) = segment_dataset(len(key_campus_list))
        self.train_key_list = key_city_list[train11:train12]+key_residential_list[train21:train22]+key_road_list[train31:train32]+key_road_list[train41:train42]
        self.eval_key_list  = key_city_list[eval11:eval12]+key_residential_list[eval21:eval22]+key_road_list[eval31:eval32]+key_road_list[eval41:eval42]
        self.test_key_list  = key_city_list[test11:test12]+key_residential_list[test21:test22]+key_road_list[test31:test32]+key_road_list[test41:test42]
        
        lines, curves = [], []
        for key, index in self.train_key_list:
            dataset = self.dataset_dict[key]
            traj_type = dataset.get_trajectory_type(index)
            if traj_type == 0:
                lines.append((key, index))
            elif traj_type == 1:
                curves.append((key, index))
        self.train_key_list = lines + curves * (int(len(lines)/len(curves))+1)

        self.eval_key_list = self.eval_key_list + self.train_key_list

        random.shuffle(self.train_key_list)
        random.shuffle(self.eval_key_list)
        random.shuffle(self.test_key_list)

    def __len__(self):
        return 100000000000

    def __getitem__(self, pesudo_index):
        if self.mode == 'train':
            # key, index = self.train_key_list[pesudo_index]
            key, index = random.sample(self.train_key_list, 1)[0]

        elif self.mode == 'eval':
            #key, index = self.eval_key_list[pesudo_index]
            key, index = random.sample(self.eval_key_list, 1)[0]
        else:
            #key, index = self.test_key_list[pesudo_index]
            key, index = random.sample(self.test_key_list, 1)[0]
        dataset = self.dataset_dict[key]

        w = dataset.oxts[index].packet.wz
        
        image = dataset.get_image(index)
        image = self.image_transforms(image)
        nav = dataset.get_nav(index)
        nav = self.image_transforms(nav)

        image = torch.cat((image, nav), 0)


        times, x_list, y_list, vx_list, vy_list, ax, ay, wz = self.get_trajectory(dataset, index)
        times = np.array(times).astype(np.float32) / self.opt.max_t
        v_0 = np.sqrt(vx_list[0]**2+vy_list[0]**2) / self.opt.max_speed

        if self.mode == 'train':
            mask = np.random.randint(0, len(times)-1, (16))
            times = times[mask]
            x = x_list[mask]
            y = y_list[mask]
            vx = vx_list[mask]
            vy = vy_list[mask]
        else:
            times = times[::3]
            x = x_list[::3]
            y = y_list[::3]
            vx = vx_list[::3]
            vy = vy_list[::3]

        x /= self.opt.max_dist
        y /= self.opt.max_dist
        vx /= self.opt.max_speed
        vy /= self.opt.max_speed

        xy = torch.FloatTensor([x, y]).T
        vxy = torch.FloatTensor([vx, vy]).T
        v0_array = torch.FloatTensor([v_0]*len(x))
        v_0 = torch.FloatTensor([v_0])

        t = torch.FloatTensor(times)

        domian = [0]*5
        domian[list(self.dataset_dict.keys()).index(key) % 5] = 1.
        domian = torch.FloatTensor(domian)

        return {'img': image, 't': t, 
                'v_0':v_0, 'v0_array':v0_array, 
                'xy':xy, 'vxy':vxy,
                'domian':domian,
                }


    def get_trajectory(self, dataset, index):
        indexs = list(range(index, index+self.num_trajectory))

        poses = dataset.pose_array[:,indexs]
        p0 = dataset.pose_array[:,0]
        p0 = poses[:,0]
        p0[:3] -= dataset.pose_array[:3,0]

        T = cu.basic.HomogeneousMatrix.xyzrpy(p0)
        point_array = np.dot(np.linalg.inv(T), np.vstack((poses[:3,:], np.ones((1,poses.shape[1])))))

        t0 = dataset.timestamps[index]

        times = [(dataset.timestamps[i]-t0).microseconds*1e-6 + (dataset.timestamps[i]-t0).seconds for i in indexs]
        x = point_array[0,:]
        y = point_array[1,:]
        vx_ = np.array([oxts.packet.vf for oxts in dataset.oxts[index:index+self.num_trajectory]])
        vy_ = np.array([oxts.packet.vl for oxts in dataset.oxts[index:index+self.num_trajectory]])
        ax_ = np.array([oxts.packet.af for oxts in dataset.oxts[index:index+self.num_trajectory]])
        ay_ = np.array([oxts.packet.al for oxts in dataset.oxts[index:index+self.num_trajectory]])
        wz = np.array([oxts.packet.wz for oxts in dataset.oxts[index:index+self.num_trajectory]])
        
        yaw = poses[-1,:] - poses[-1,0]
        vx = np.cos(yaw)*vx_ - np.sin(yaw)*vy_
        vy = np.cos(yaw)*vy_ + np.sin(yaw)*vx_
        
        ax = np.cos(yaw)*ax_ - np.sin(yaw)*ay_
        ay = np.cos(yaw)*ay_ + np.sin(yaw)*ax_

        return times, x, y, vx, vy, ax, ay, wz

class BoostDatasetDomain(Dataset):
    def __init__(self, param, mode, opt=None, data_index=None):
        '''
        Load past costmap and future trajectory
        '''
        self.opt = opt
        self.mode = mode
        self.data_index = data_index
        raw_master = RawMaster(param.basedir)

        self.num_costmap = param.net.num_costmap
        self.num_trajectory = param.net.num_trajectory

        image_transforms = [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
        self.image_transforms = transforms.Compose(image_transforms)
                
        '''get valid'''
        self.dataset_dict = dict()
        self.key_list = []
        key_city_list, key_residential_list, key_road_list, key_campus_list = [], [], [], []
        for iii, dataset in enumerate(raw_master):
            key, scene_type = raw_master.get_dataset_scene(dataset)
            num_valid = max(0, dataset.length_valid-(self.num_costmap+self.num_trajectory-1)+1)
            if num_valid > 0:
                self.dataset_dict[key] = dataset
                index_list = list(range(self.num_costmap-1, self.num_costmap+num_valid-1))
                key_list = [(key, i) for i in index_list]
                self.key_list.extend(key_list)
                if scene_type == SceneType.CITY:
                    key_city_list.extend(key_list)
                elif scene_type == SceneType.RESIDENTIAL:
                    key_residential_list.extend(key_list)
                elif scene_type == SceneType.ROAD:
                    key_road_list.extend(key_list)
                else:
                    key_campus_list.extend(key_list)
        
        ''' train : eval : test = 7:1:2 '''
        (train11, train12), (eval11, eval12), (test11, test12) = segment_dataset(len(key_city_list))
        (train21, train22), (eval21, eval22), (test21, test22) = segment_dataset(len(key_residential_list))
        (train31, train32), (eval31, eval32), (test31, test32) = segment_dataset(len(key_road_list))
        (train41, train42), (eval41, eval42), (test41, test42) = segment_dataset(len(key_campus_list))
        self.train_key_list = key_city_list[train11:train12]+key_residential_list[train21:train22]+key_road_list[train31:train32]+key_road_list[train41:train42]
        self.eval_key_list  = key_city_list[eval11:eval12]+key_residential_list[eval21:eval22]+key_road_list[eval31:eval32]+key_road_list[eval41:eval42]
        self.test_key_list  = key_city_list[test11:test12]+key_residential_list[test21:test22]+key_road_list[test31:test32]+key_road_list[test41:test42]
        
        lines, curves = [], []
        for key, index in self.train_key_list:
            dataset = self.dataset_dict[key]
            traj_type = dataset.get_trajectory_type(index)
            if traj_type == 0:
                lines.append((key, index))
            elif traj_type == 1:
                curves.append((key, index))
        self.train_key_list = lines + curves * (int(len(lines)/len(curves))+1)

        self.eval_key_list = self.eval_key_list + self.train_key_list

        random.shuffle(self.train_key_list)
        random.shuffle(self.eval_key_list)
        random.shuffle(self.test_key_list)

    def __len__(self):
        return 100000000000

    def __getitem__(self, pesudo_index):
        if self.mode == 'train':
            # key, index = self.train_key_list[pesudo_index]
            while True:
                key, index = random.sample(self.train_key_list, 1)[0]
                if int(key[1]) % 5 == self.data_index:
                    break

        elif self.mode == 'eval':
            #key, index = self.eval_key_list[pesudo_index]
            key, index = random.sample(self.eval_key_list, 1)[0]
        else:
            #key, index = self.test_key_list[pesudo_index]
            key, index = random.sample(self.test_key_list, 1)[0]
        dataset = self.dataset_dict[key]

        w = dataset.oxts[index].packet.wz
        
        image = dataset.get_image(index)
        image = self.image_transforms(image)
        nav = dataset.get_nav(index)
        nav = self.image_transforms(nav)

        image = torch.cat((image, nav), 0)

        times, x_list, y_list, vx_list, vy_list, ax, ay, wz = self.get_trajectory(dataset, index)
        times = np.array(times).astype(np.float32) / self.opt.max_t
        v_0 = np.sqrt(vx_list[0]**2+vy_list[0]**2) / self.opt.max_speed

        times = times[::7][1:]
        x = x_list[::7][1:]
        y = y_list[::7][1:]
        vx = vx_list[::7][1:]
        vy = vy_list[::7][1:]

        x /= self.opt.max_dist
        y /= self.opt.max_dist
        vx /= self.opt.max_speed
        vy /= self.opt.max_speed

        xy = torch.FloatTensor([x, y]).T
        vxy = torch.FloatTensor([vx, vy]).T
        v0_array = torch.FloatTensor([v_0]*len(x))
        v_0 = torch.FloatTensor([v_0])

        t = torch.FloatTensor(times)

        return {'img': image, 't': t, 
                'v_0':v_0, 'v0_array':v0_array, 
                'xy':xy, 'vxy':vxy,
                }


    def get_trajectory(self, dataset, index):
        indexs = list(range(index, index+self.num_trajectory))

        poses = dataset.pose_array[:,indexs]
        p0 = dataset.pose_array[:,0]
        p0 = poses[:,0]
        p0[:3] -= dataset.pose_array[:3,0]

        T = cu.basic.HomogeneousMatrix.xyzrpy(p0)
        point_array = np.dot(np.linalg.inv(T), np.vstack((poses[:3,:], np.ones((1,poses.shape[1])))))

        t0 = dataset.timestamps[index]

        times = [(dataset.timestamps[i]-t0).microseconds*1e-6 + (dataset.timestamps[i]-t0).seconds for i in indexs]
        x = point_array[0,:]
        y = point_array[1,:]
        vx_ = np.array([oxts.packet.vf for oxts in dataset.oxts[index:index+self.num_trajectory]])
        vy_ = np.array([oxts.packet.vl for oxts in dataset.oxts[index:index+self.num_trajectory]])
        ax_ = np.array([oxts.packet.af for oxts in dataset.oxts[index:index+self.num_trajectory]])
        ay_ = np.array([oxts.packet.al for oxts in dataset.oxts[index:index+self.num_trajectory]])
        wz = np.array([oxts.packet.wz for oxts in dataset.oxts[index:index+self.num_trajectory]])
        
        yaw = poses[-1,:] - poses[-1,0]
        vx = np.cos(yaw)*vx_ - np.sin(yaw)*vy_
        vy = np.cos(yaw)*vy_ + np.sin(yaw)*vx_
        
        ax = np.cos(yaw)*ax_ - np.sin(yaw)*ay_
        ay = np.cos(yaw)*ay_ + np.sin(yaw)*ax_

        return times, x, y, vx, vy, ax, ay, wz

class BoostDatasetDomain2(Dataset):
    def __init__(self, param, mode, opt=None, data_index=None):
        '''
        Load past costmap and future trajectory
        '''
        self.opt = opt
        self.mode = mode
        self.data_index = data_index
        raw_master = RawMaster(param.basedir)

        self.num_costmap = param.net.num_costmap
        self.num_trajectory = param.net.num_trajectory

        image_transforms = [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
        self.image_transforms = transforms.Compose(image_transforms)
                
        '''get valid'''
        self.dataset_dict = dict()
        self.key_list = []
        key_city_list, key_residential_list, key_road_list, key_campus_list = [], [], [], []
        for iii, dataset in enumerate(raw_master):
            key, scene_type = raw_master.get_dataset_scene(dataset)
            num_valid = max(0, dataset.length_valid-(self.num_costmap+self.num_trajectory-1)+1)
            if num_valid > 0:
                self.dataset_dict[key] = dataset
                index_list = list(range(self.num_costmap-1, self.num_costmap+num_valid-1))
                key_list = [(key, i) for i in index_list]
                self.key_list.extend(key_list)
                if scene_type == SceneType.CITY:
                    key_city_list.extend(key_list)
                elif scene_type == SceneType.RESIDENTIAL:
                    key_residential_list.extend(key_list)
                elif scene_type == SceneType.ROAD:
                    key_road_list.extend(key_list)
                else:
                    key_campus_list.extend(key_list)
        
        ''' train : eval : test = 7:1:2 '''
        (train11, train12), (eval11, eval12), (test11, test12) = segment_dataset(len(key_city_list))
        (train21, train22), (eval21, eval22), (test21, test22) = segment_dataset(len(key_residential_list))
        (train31, train32), (eval31, eval32), (test31, test32) = segment_dataset(len(key_road_list))
        (train41, train42), (eval41, eval42), (test41, test42) = segment_dataset(len(key_campus_list))
        self.train_key_list = key_city_list[train11:train12]+key_residential_list[train21:train22]+key_road_list[train31:train32]+key_road_list[train41:train42]
        self.eval_key_list  = key_city_list[eval11:eval12]+key_residential_list[eval21:eval22]+key_road_list[eval31:eval32]+key_road_list[eval41:eval42]
        self.test_key_list  = key_city_list[test11:test12]+key_residential_list[test21:test22]+key_road_list[test31:test32]+key_road_list[test41:test42]
        
        lines, curves = [], []
        for key, index in self.train_key_list:
            dataset = self.dataset_dict[key]
            traj_type = dataset.get_trajectory_type(index)
            if traj_type == 0:
                lines.append((key, index))
            elif traj_type == 1:
                curves.append((key, index))
        self.train_key_list = lines + curves * (int(len(lines)/len(curves))+1)

        self.eval_key_list = self.eval_key_list + self.train_key_list

        random.shuffle(self.train_key_list)
        random.shuffle(self.eval_key_list)
        random.shuffle(self.test_key_list)

    def __len__(self):
        return 100000000000

    def __getitem__(self, pesudo_index):
        if self.mode == 'train':
            # key, index = self.train_key_list[pesudo_index]
            key, index = random.sample(self.train_key_list, 1)[0]

        elif self.mode == 'eval':
            #key, index = self.eval_key_list[pesudo_index]
            key, index = random.sample(self.eval_key_list, 1)[0]
        else:
            #key, index = self.test_key_list[pesudo_index]
            key, index = random.sample(self.test_key_list, 1)[0]
        dataset = self.dataset_dict[key]

        w = dataset.oxts[index].packet.wz
        
        image = dataset.get_image(index)
        image = self.image_transforms(image)
        nav = dataset.get_nav(index)
        nav = self.image_transforms(nav)

        image = torch.cat((image, nav), 0)


        times, x_list, y_list, vx_list, vy_list, ax, ay, wz = self.get_trajectory(dataset, index)
        times = np.array(times).astype(np.float32) / self.opt.max_t
        v_0 = np.sqrt(vx_list[0]**2+vy_list[0]**2) / self.opt.max_speed

        times = times[::3]
        x = x_list[::3]
        y = y_list[::3]
        vx = vx_list[::3]
        vy = vy_list[::3]

        x /= self.opt.max_dist
        y /= self.opt.max_dist
        vx /= self.opt.max_speed
        vy /= self.opt.max_speed

        xy = torch.FloatTensor([x, y]).T
        vxy = torch.FloatTensor([vx, vy]).T
        v0_array = torch.FloatTensor([v_0]*len(x))
        v_0 = torch.FloatTensor([v_0])

        t = torch.FloatTensor(times)

        domian = [0]*5
        domian[list(self.dataset_dict.keys()).index(key) % 5] = 1.
        domian = torch.FloatTensor(domian)

        return {'img': image, 't': t, 
                'v_0':v_0, 'v0_array':v0_array, 
                'xy':xy, 'vxy':vxy,
                'domian':domian,
                }


    def get_trajectory(self, dataset, index):
        indexs = list(range(index, index+self.num_trajectory))

        poses = dataset.pose_array[:,indexs]
        p0 = dataset.pose_array[:,0]
        p0 = poses[:,0]
        p0[:3] -= dataset.pose_array[:3,0]

        T = cu.basic.HomogeneousMatrix.xyzrpy(p0)
        point_array = np.dot(np.linalg.inv(T), np.vstack((poses[:3,:], np.ones((1,poses.shape[1])))))

        t0 = dataset.timestamps[index]

        times = [(dataset.timestamps[i]-t0).microseconds*1e-6 + (dataset.timestamps[i]-t0).seconds for i in indexs]
        x = point_array[0,:]
        y = point_array[1,:]
        vx_ = np.array([oxts.packet.vf for oxts in dataset.oxts[index:index+self.num_trajectory]])
        vy_ = np.array([oxts.packet.vl for oxts in dataset.oxts[index:index+self.num_trajectory]])
        ax_ = np.array([oxts.packet.af for oxts in dataset.oxts[index:index+self.num_trajectory]])
        ay_ = np.array([oxts.packet.al for oxts in dataset.oxts[index:index+self.num_trajectory]])
        wz = np.array([oxts.packet.wz for oxts in dataset.oxts[index:index+self.num_trajectory]])
        
        yaw = poses[-1,:] - poses[-1,0]
        vx = np.cos(yaw)*vx_ - np.sin(yaw)*vy_
        vy = np.cos(yaw)*vy_ + np.sin(yaw)*vx_
        
        ax = np.cos(yaw)*ax_ - np.sin(yaw)*ay_
        ay = np.cos(yaw)*ay_ + np.sin(yaw)*ax_

        return times, x, y, vx, vy, ax, ay, wz

class GANDataset(Dataset):
    def __init__(self, param, mode, opt=None, data_index=None):
        self.opt = opt
        self.mode = mode
        data_index = data_index
        raw_master = RawMaster(param.basedir)

        self.num_costmap = param.net.num_costmap
        self.num_trajectory = param.net.num_trajectory

                
        '''get valid'''
        self.dataset_dict = dict()
        self.key_list = []
        key_city_list, key_residential_list, key_road_list, key_campus_list = [], [], [], []
        for iii, dataset in enumerate(raw_master):
            key, scene_type = raw_master.get_dataset_scene(dataset)
            num_valid = max(0, dataset.length_valid-(self.num_costmap+self.num_trajectory-1)+1)
            if num_valid > 0:
                self.dataset_dict[key] = dataset
                index_list = list(range(self.num_costmap-1, self.num_costmap+num_valid-1))
                key_list = [(key, i) for i in index_list]
                self.key_list.extend(key_list)
                if scene_type == SceneType.CITY:
                    key_city_list.extend(key_list)
                elif scene_type == SceneType.RESIDENTIAL:
                    key_residential_list.extend(key_list)
                elif scene_type == SceneType.ROAD:
                    key_road_list.extend(key_list)
                else:
                    key_campus_list.extend(key_list)
        
        ''' train : eval : test = 7:1:2 '''
        (train11, train12), (eval11, eval12), (test11, test12) = segment_dataset(len(key_city_list))
        (train21, train22), (eval21, eval22), (test21, test22) = segment_dataset(len(key_residential_list))
        (train31, train32), (eval31, eval32), (test31, test32) = segment_dataset(len(key_road_list))
        (train41, train42), (eval41, eval42), (test41, test42) = segment_dataset(len(key_campus_list))
        self.train_key_list = key_city_list[train11:train12]+key_residential_list[train21:train22]+key_road_list[train31:train32]+key_road_list[train41:train42]
        self.eval_key_list  = key_city_list[eval11:eval12]+key_residential_list[eval21:eval22]+key_road_list[eval31:eval32]+key_road_list[eval41:eval42]
        self.test_key_list  = key_city_list[test11:test12]+key_residential_list[test21:test22]+key_road_list[test31:test32]+key_road_list[test41:test42]
        
        lines, curves = [], []
        for key, index in self.train_key_list:
            dataset = self.dataset_dict[key]
            traj_type = dataset.get_trajectory_type(index)
            if traj_type == 0:
                lines.append((key, index))
            elif traj_type == 1:
                curves.append((key, index))
        self.train_key_list = lines + curves * (int(len(lines)/len(curves))+1)

        self.eval_key_list = self.eval_key_list + self.train_key_list

    def __len__(self):
        return 100000000000

    def interpolation(self, time_list, x_list, y_list, t):
        if t >= time_list[-1]:
            dt = time_list[-1] - time_list[-2]
            dx = x_list[-1] - x_list[-2]
            dy = y_list[-1] - y_list[-2]
            x = x_list[-1] + (dx/dt)*(t - time_list[-1])
            y = y_list[-1] + (dy/dt)*(t - time_list[-1])
        else:
            index = np.argmin(np.abs(time_list-t))
            if t >= time_list[index]:
                dt = time_list[index+1] - time_list[index]
                dx = x_list[index+1] - x_list[index]
                dy = y_list[index+1] - y_list[index]
                x = x_list[index] + (dx/dt)*(t - time_list[index])
                y = y_list[index] + (dy/dt)*(t - time_list[index])
            else:
                dt = time_list[index] - time_list[index-1]
                dx = x_list[index] - x_list[index-1]
                dy = y_list[index] - y_list[index-1]
                x = x_list[index-1] + (dx/dt)*(t - time_list[index-1])
                y = y_list[index-1] + (dy/dt)*(t - time_list[index-1])
        return x, y

    def __getitem__(self, pesudo_index):
        while True:
            if self.mode == 'train':
                # key, index = self.train_key_list[pesudo_index]
                key, index = random.sample(self.train_key_list, 1)[0]

            elif self.mode == 'eval':
                #key, index = self.eval_key_list[pesudo_index]
                key, index = random.sample(self.eval_key_list, 1)[0]
            else:
                #key, index = self.test_key_list[pesudo_index]
                key, index = random.sample(self.test_key_list, 1)[0]
            dataset : Raw = self.dataset_dict[key]

            w = dataset.oxts[index].packet.wz

            times, x_list, y_list, vx_list, vy_list, ax, ay, wz = self.get_trajectory(dataset, index)
            times = np.array(times).astype(np.float32) / self.opt.max_t
            v_0 = np.sqrt(vx_list[0]**2+vy_list[0]**2) / self.opt.max_speed


            times = times[::3]
            x = x_list[::3]
            y = y_list[::3]
            vx = vx_list[::3]
            vy = vy_list[::3]

            break
        
        oxford_time = [0.0000, 0.1875, 0.3750, 0.5625, 0.7500, 0.9376, 1.1251, 1.3125, 1.5000, 1.6876, 1.8751, 2.0626, 2.2501, 2.4376, 2.6251, 2.8126]
        
        xs = []
        ys = []
        for t in oxford_time:
            _x, _y = self.interpolation(times, x, y, t/self.opt.max_t)
            xs.append(_x)
            ys.append(_y)

        x = np.array(xs).astype(np.float32)/self.opt.max_dist
        y = np.array(ys).astype(np.float32)/self.opt.max_dist
        vx /= self.opt.max_speed
        vy /= self.opt.max_speed

        xy = torch.FloatTensor([x, y]).T
        vxy = torch.FloatTensor([vx, vy]).T
        v0_array = torch.FloatTensor([v_0]*len(x))
        v_0 = torch.FloatTensor([v_0])

        t = torch.FloatTensor(np.array(oxford_time).astype(np.float32)/self.opt.max_t)

        return {'t': t, 
                'v_0':v_0, 'v0_array':v0_array, 
                'xy':xy, 'vxy':vxy,
                }


    def get_trajectory(self, dataset, index):
        indexs = list(range(index, index+self.num_trajectory))

        poses = dataset.pose_array[:,indexs]
        p0 = dataset.pose_array[:,0]
        p0 = poses[:,0]
        p0[:3] -= dataset.pose_array[:3,0]

        T = cu.basic.HomogeneousMatrix.xyzrpy(p0)
        point_array = np.dot(np.linalg.inv(T), np.vstack((poses[:3,:], np.ones((1,poses.shape[1])))))

        t0 = dataset.timestamps[index]

        times = [(dataset.timestamps[i]-t0).microseconds*1e-6 + (dataset.timestamps[i]-t0).seconds for i in indexs]
        x = point_array[0,:]
        y = point_array[1,:]
        vx_ = np.array([oxts.packet.vf for oxts in dataset.oxts[index:index+self.num_trajectory]])
        vy_ = np.array([oxts.packet.vl for oxts in dataset.oxts[index:index+self.num_trajectory]])
        ax_ = np.array([oxts.packet.af for oxts in dataset.oxts[index:index+self.num_trajectory]])
        ay_ = np.array([oxts.packet.al for oxts in dataset.oxts[index:index+self.num_trajectory]])
        wz = np.array([oxts.packet.wz for oxts in dataset.oxts[index:index+self.num_trajectory]])
        
        yaw = poses[-1,:] - poses[-1,0]
        vx = np.cos(yaw)*vx_ - np.sin(yaw)*vy_
        vy = np.cos(yaw)*vy_ + np.sin(yaw)*vx_
        
        ax = np.cos(yaw)*ax_ - np.sin(yaw)*ay_
        ay = np.cos(yaw)*ay_ + np.sin(yaw)*ax_
        
        return times, x, y, vx, vy, ax, ay, wz


    def get_images(self, dataset, index):
        indexs = list(range(index, index+self.num_costmap))
        images = [self.image_transforms(dataset.get_image(index)) for index in indexs]
        return images