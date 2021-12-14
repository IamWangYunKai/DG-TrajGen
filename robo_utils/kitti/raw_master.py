import carla_utils as cu

import os
import random
import copy
import cv2
import numpy as np
import time
from enum import IntEnum

from .raw import Raw


class SceneType(IntEnum):
    CITY = 0
    RESIDENTIAL = 1
    ROAD = 2
    CAMPUS = 3


class RawMaster(object):
    def __init__(self, basedir):
        self.basedir = basedir
        dates = os.listdir(basedir)
        '''store (date, drive) pair into list'''
        self.date_drive_list = []
        for date in dates:
            names = os.listdir(os.path.join(basedir, date))
            for name in names:
                if 'drive' not in name:
                    continue
                drive = name.split('drive_')[1][:4]
                self.date_drive_list.append((date, drive))
        '''need a random one'''
        self.date_drive_list_random = copy.copy(self.date_drive_list)
        random.shuffle(self.date_drive_list_random)

        dir_path = os.path.dirname(__file__)
        city_scene = np.loadtxt('{}/city.scene'.format(dir_path), delimiter=' ', dtype=np.object)
        residential_scene = np.loadtxt('{}/residential.scene'.format(dir_path), delimiter=' ', dtype=np.object)
        road_scene = np.loadtxt('{}/road.scene'.format(dir_path), delimiter=' ', dtype=np.object)
        campus_scene = np.loadtxt('{}/campus.scene'.format(dir_path), delimiter=' ', dtype=np.object)
        # self.bad_drive = np.loadtxt('{}/bad_drive.txt'.format(dir_path), dtype=np.object)
        self.city_scene = [tuple(t) for t in city_scene]
        self.residential_scene = [tuple(t) for t in residential_scene]
        self.road_scene = [tuple(t) for t in road_scene]
        self.campus_scene = [tuple(t) for t in campus_scene]


    def __len__(self):
        return len(self.date_drive_list)
    def __iter__(self):
        for (date, drive) in self.date_drive_list:
            yield self.get_specific_dataset(date, drive)
    def __getitem__(self, key):
        try:
            date, drive = self.date_drive_list[key]
            return self.get_specific_dataset(date, drive)
        except IndexError as e:
            raise e
    def iter_random(self):
        for (date, drive) in self.date_drive_list_random:
            yield self.get_specific_dataset(date, drive)
    def iter_scene(self, scene):
        if scene in ['city', 'residential', 'road', 'campus']:
            for (date, drive) in getattr(self, scene+'_scene'):
                yield self.get_specific_dataset(date, drive)


    def get_specific_dataset(self, date, drive):
        return Raw(self.basedir, date, drive)
    def get_random_dataset(self):
        date, drive = random.choice(self.date_drive_list)
        print(date, drive)
        return self.get_specific_dataset(date, drive)
    

    def get_dataset_key(self, dataset):
        s = dataset.drive.split('_drive_')
        return (s[0], s[1][:4])
    
    def get_dataset_scene(self, dataset):
        key = self.get_dataset_key(dataset)
        types = []
        if key in self.city_scene:
            types.append(SceneType.CITY)
        elif key in self.residential_scene:
            types.append(SceneType.RESIDENTIAL)
        elif key in self.road_scene:
            types.append(SceneType.ROAD)
        else:
            types.append(SceneType.CAMPUS)
        return key, types[0]





def filter_ground(pointcloud):
    height = -Raw.imu_height
    z_array = pointcloud[2,:]
    mask = np.where((z_array > height+0.8) & (z_array < height+2))[0]
    pointcloud = pointcloud[:,mask]
    i_array = pointcloud[3,:]
    mask = np.where((i_array > 0.2))[0]
    pointcloud = pointcloud[:,mask]
    return pointcloud
