import carla_utils as cu

import os
from os.path import join
import time
import numpy as np
import cv2
import copy
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear as demosaic

from .python.camera_model import CameraModel
from .utils import read_extrinsics_file



class ProcessUtils(object):
    def __init__(self, models_dir, extrinsics_dir):
        '''velodyne'''
        left = np.loadtxt(join(extrinsics_dir, 'velodyne_left.txt'), delimiter=' ', dtype=np.float64)
        roll, pitch, yaw = left[3], left[4], left[5]
        R = cu.basic.RotationMatrix.ypr(roll, pitch+np.deg2rad(5), yaw)
        t = np.array(left[:3]).reshape(3,1)
        self.left_T = cu.basic.HomogeneousMatrix.rotation_translation(R, t)
        
        right= np.loadtxt(join(extrinsics_dir, 'velodyne_right.txt'), delimiter=' ', dtype=np.float64)
        R = cu.basic.RotationMatrix.ypr(roll+np.deg2rad(-2), pitch+np.deg2rad(5), yaw)
        t = np.array(right[:3]).reshape(3,1)
        self.right_T = cu.basic.HomogeneousMatrix.rotation_translation(R, t)

        '''camera'''
        self.camera_model = CameraModel(models_dir, 'stereo/centre')

        '''matrix'''
        T_cam = read_extrinsics_file(join(extrinsics_dir, self.camera_model.camera+'.txt'))
        T_cam_imu = read_extrinsics_file(join(extrinsics_dir, 'ins.txt'))
        T_img_cam = np.linalg.inv(self.camera_model.G_camera_image)
        self.T_img_imu = cu.basic.np_dot(T_img_cam, T_cam, T_cam_imu)

        fx, fy = self.camera_model.focal_length[0], self.camera_model.focal_length[1]
        u0, v0 = self.camera_model.principal_point[0], self.camera_model.principal_point[1]
        K = np.array([ [fx, 0, u0],
                            [0, fy, v0],
                            [0,  0,  1] ])
        self.K = np.dot(K, np.eye(3,4))

        self.T_cam_imu = T_cam_imu
        self.T_imu_cam = np.linalg.inv(T_cam_imu)
        

    def pointcloud(self, pc, velodyne_sensor):
        # TODO
        intensity_array = copy.deepcopy(pc[3,:])
        pc[3,:] = 1
        pc_ed = np.dot(self.left_T, pc) if 'left' in velodyne_sensor else np.dot(self.right_T, pc)
        pc[3,:] = intensity_array
        pc_ed[3,:] = intensity_array
        return pc_ed

    def image(self, image):
        image = demosaic(image, 'gbrg')
        image = self.camera_model.undistort(image)
        image = np.array(image).astype(np.uint8)
        image[:,:,[0,2]] = image[:,:,[2,0]]   # change BGR to RGB
        return image
