import carla_utils as cu
import os, sys
from os.path import join


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import cv2
import copy
import time

from .utils import load_velodyne_binary, load_velodyne_png


def find_nearest(timestamp, timestamp_array):
    idx = (np.abs(timestamp_array - timestamp)).argmin()
    return timestamp_array[idx]


class PartialDataset(object):
    def __init__(self, path, process, pc_type='png'):
        self.path = path
        self.name = os.path.split(path)[1]
        self.process = process

        '''lidar'''
        self.velodyne_left_timestamp_array = np.loadtxt(join(path, 'velodyne_left.timestamps'), delimiter=' ', usecols=[0], dtype=np.int64)
        self.velodyne_right_timestamp_array= np.loadtxt(join(path, 'velodyne_right.timestamps'),delimiter=' ', usecols=[0], dtype=np.int64)
        (postfix, func) = ('.png', load_velodyne_png) if pc_type == 'png' else ('.bin', load_velodyne_binary)
        self.velodyne_left_data = lambda timestamp: func(join(path, 'velodyne_left', str(timestamp)+postfix))
        self.velodyne_right_data= lambda timestamp: func(join(path, 'velodyne_right',str(timestamp)+postfix))

        '''stereo centre'''
        self.stereo_centre_timestamp_array = np.loadtxt(join(path, 'stereo.timestamps'), delimiter=' ', usecols=[0], dtype=np.int64)
        self.stereo_centre_data = lambda timestamp: cv2.imread(join(path, 'stereo/centre', str(timestamp)+'.png'), cv2.IMREAD_GRAYSCALE)

        '''gps'''
        self.gps_df = pd.read_csv(join(path, 'gps', 'gps.csv'))
        self.ins_df = pd.read_csv(join(path, 'gps', 'ins.csv'))
        self.ins_timestamp_array = self.ins_df['timestamp'].values

        '''vo'''
        self.vo_df = pd.read_csv(join(path, 'vo', 'vo.csv'))



    def image(self, ref_timestamp):
        stereo_centre_timestamp = find_nearest(ref_timestamp, self.stereo_centre_timestamp_array)
        stereo_centre_data = self.stereo_centre_data(stereo_centre_timestamp)
        return self.process.image(stereo_centre_data)

    def pointcloud(self, ref_timestamp):
        velodyne_left_timestamp = find_nearest(ref_timestamp, self.velodyne_left_timestamp_array)
        velodyne_left_data = self.velodyne_left_data(velodyne_left_timestamp)
        velodyne_right_timestamp= find_nearest(ref_timestamp, self.velodyne_right_timestamp_array)
        velodyne_right_data= self.velodyne_right_data(velodyne_right_timestamp)

        velodyne_left = self.process.pointcloud(velodyne_left_data, 'left')
        velodyne_right = self.process.pointcloud(velodyne_right_data, 'right')
        return np.hstack((velodyne_left, velodyne_right))
    
    
    
    

