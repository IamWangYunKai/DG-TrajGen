import carla_utils as cu
from carla_utils.basic import np_dot

import copy
import numpy as np
import cv2

from . import utils


def data_augmentation(pose_array, min_dist):
    number = pose_array.shape[1]
    augmented = None
    for i in range(number-1):      
        p1 = pose_array[:,i][:,np.newaxis]
        p2 = pose_array[:,i+1][:,np.newaxis]
        interpolated = utils.linear_interpolation(p1, p2, min_dist)
        if augmented is None:
            augmented = interpolated
        else:
            augmented = np.hstack((augmented, interpolated))
    return augmented


class NavMapUtils(object):
    def __init__(self, param):
        '''param'''
        self.vehicle_width = param.vehicle_width
        self.correction_length = param.correction_length
        self.trajectory_length = param.trajectory_length
        self.lateral_step_factor = param.lateral_step_factor
        self.image_width = param.costmap.image_width
        self.image_height = param.costmap.image_height
        self.max_pixel = np.array([self.image_height, self.image_width]).reshape(2,1)
        self.resolution = param.nav_map.resolution


    def trajectory_pose(self, pose_array, T_w_imu, color=(255,0,0)):
        augmented = data_augmentation(pose_array, self.resolution/2)
        point1, point2 = utils.expand_lateral(augmented, self.vehicle_width, self.correction_length)

        '''change coordinate so that pose_array starts with 0'''
        R, t = cu.basic.RotationMatrixTranslationVector.homogeneous_matrix(T_w_imu)
        T_imu_w = cu.basic.HomogeneousMatrixInverse.rotation_translation(R, t)
        '''scale'''
        K = np.diag([1/self.resolution, 1/self.resolution, 1, 1])
        '''change into image coordinate'''
        xyzrpy = [self.image_width/2, -self.image_height, 0, 0,0,np.pi/2]
        T = np.dot(np.diag([1,-1,0,1]), cu.basic.HomogeneousMatrix.xyzrpy(xyzrpy))  # TODO understand


        pixel_array1, pixel_array2 = np_dot(T, K, T_imu_w, point1), np_dot(T, K, T_imu_w, point2)
        pixel_array1, pixel_array2 = pixel_array1[:2,:], pixel_array2[:2,:]
        
        nav_map = np.full((self.image_height, self.image_width, 3), 255, dtype=np.uint8)
        for i in range(augmented.shape[1]):
            pixel1, pixel2 = pixel_array1[:,i][:,np.newaxis], pixel_array2[:,i][:,np.newaxis]
            pixel1, pixel2 = pixel1[::-1,:], pixel2[::-1,:]
            flag1 = (pixel1 >= 0).all() and (pixel1 < self.max_pixel).all()
            flag2 = (pixel2 >= 0).all() and (pixel2 < self.max_pixel).all()
            if not flag1 and not flag2: continue
            utils.draw_line(nav_map, pixel1, pixel2, color=color, step_factor=self.lateral_step_factor)

        nav_map = postprocess(nav_map)
        nav_map = np.flip(nav_map, axis=1)  ##!warning: for oxford, delete
        return nav_map
    

def postprocess(image):
    kernel = np.ones((5,5), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.erode(image, kernel, iterations=1)
    return image


def valid(nav_map):
    assert nav_map.dtype == np.uint8
    mask = np.where((nav_map < 255))[0]
    return len(mask) > 0
