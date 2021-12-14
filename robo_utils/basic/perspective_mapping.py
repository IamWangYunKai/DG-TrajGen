import carla_utils as cu
from carla_utils.basic import np_dot

import numpy as np
import copy
import cv2
import time

from . import utils


def data_augmentation(pose_array, x0=-float('inf'), weight=[0.014, 0.03, 0.09]):
    number = pose_array.shape[1]
    augmented = None
    for i in range(number-1):
        if i/number < 0.4:
            min_dist = weight[0]
        elif i/number < 0.7:
            min_dist = weight[1]
        else:
            min_dist = weight[2]
        
        p1 = pose_array[:,i][:,np.newaxis]
        p2 = pose_array[:,i+1][:,np.newaxis]
        interpolated = utils.linear_interpolation(p1, p2, min_dist)
        if augmented is None:
            augmented = interpolated
        else:
            augmented = np.hstack((augmented, interpolated))
    mask = np.where(augmented[0,:] > x0)[0]
    augmented = augmented[:, mask]
    return augmented

class PerspectiveMapping(object):
    def __init__(self, param, K, T_img_imu):
        '''
            K: numpy.ndarray 3×4
        '''

        '''param'''
        self.vehicle_width = param.vehicle_width
        self.correction_length = param.correction_length
        self.trajectory_length = param.trajectory_length
        self.lateral_step_factor = param.lateral_step_factor
        self.image_width = param.pm.image_width
        self.image_height = param.pm.image_height
        self.max_pixel = np.array([self.image_height, self.image_width]).reshape(2,1)

        self.K, self.T_img_imu = K, T_img_imu

    
    def trajectory_pose(self, pose_array, T_w_imu, image=None, color=(255,255,255)):
        '''
            pose_array: in world frame
            T_w_imu: transformation matrix from imu to world
            image: numpy.ndarray (height, width, 3) (optional)
            color: tuple, rgb
                gold: (255, 215, 0)
                red: (255, 0, 0)
                blue: (0, 191, 255)
                green: (0, 255, 0)
                purple: (123, 104, 238)
                gold: (255, 215, 0)
        '''

        weight = [0.014, 0.014, 0.014]   ## for oxford
        augmented = data_augmentation(pose_array, weight=weight)
        point1, point2 = utils.expand_lateral(augmented, self.vehicle_width/4, self.correction_length)

        R, t = cu.basic.RotationMatrixTranslationVector.homogeneous_matrix(T_w_imu)
        T_imu_w = cu.basic.HomogeneousMatrixInverse.rotation_translation(R, t)

        pixel1 = np_dot(self.K, self.T_img_imu, T_imu_w, point1)
        pixel2 = np_dot(self.K, self.T_img_imu, T_imu_w, point2)
        pixel_array1, pixel_array2 = pixel1[:2,:] / pixel1[-1,:], pixel2[:2,:] / pixel2[-1,:]

        if image is not None:
            image = cv2.resize(image, (self.image_width, self.image_height), interpolation=cv2.INTER_CUBIC)
        else:
            image = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
        for i in range(augmented.shape[1]):
            pixel1, pixel2 = pixel_array1[:,i][:,np.newaxis], pixel_array2[:,i][:,np.newaxis]
            pixel1, pixel2 = pixel1[::-1,:], pixel2[::-1,:]
            flag1 = (pixel1 >= 0).all() and (pixel1 < self.max_pixel).all()
            flag2 = (pixel2 >= 0).all() and (pixel2 < self.max_pixel).all()
            if not flag1 or not flag2: continue
            utils.draw_line(image, pixel1, pixel2, color=color, step_factor=self.lateral_step_factor)

        return self.postprocess(image)
    

    def trajectory_point(self, points, T_w_imu, color, image=None, width=0.1):
        '''
            points: numpy.ndarray, (x,y,z)×N
        '''
        pose_array = np.vstack((points, np.zeros((3,points.shape[1]))))
        augmented = data_augmentation(pose_array, [0.001, 0.001, 0.001])
        point1, point2 = utils.expand_lateral(augmented, width, self.correction_length)

        R, t = cu.basic.RotationMatrixTranslationVector.homogeneous_matrix(T_w_imu)
        T_imu_w = cu.basic.HomogeneousMatrixInverse(R, t)

        pixel1 = np_dot(self.K, self.T_img_imu, T_imu_w, point1)

        pixel2 = np_dot(self.K, self.T_img_imu, T_imu_w, point2)
        pixel_array1, pixel_array2 = pixel1[:2,:] / pixel1[-1,:], pixel2[:2,:] / pixel2[-1,:]

        if image is not None:
            image = cv2.resize(image, (self.image_width, self.image_height), interpolation=cv2.INTER_CUBIC)
        else:
            image = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)


        # mask = np.zeros((image.shape), dtype=np.uint8)

        for i in range(augmented.shape[1]):
            pixel1, pixel2 = pixel_array1[:,i][:,np.newaxis], pixel_array2[:,i][:,np.newaxis]
            pixel1, pixel2 = pixel1[::-1,:], pixel2[::-1,:]
            flag1 = (pixel1 >= 0).all() and (pixel1 < self.max_pixel).all()
            flag2 = (pixel2 >= 0).all() and (pixel2 < self.max_pixel).all()
            if not flag1 and not flag2: continue
            utils.draw_line(image, pixel1, pixel2, color=color, step_factor=self.lateral_step_factor)  ## TODO check

        # image = cv2.addWeighted(image, 1, mask, 0.3, 0)
        return image
    

    def postprocess(self, image):
        kernel = np.ones((5,5), np.uint8)
        image = cv2.dilate(image, kernel, iterations=1)
        image = cv2.erode(image, kernel, iterations=1)
        return image



def valid(image_pm):
    assert image_pm.dtype == np.uint8
    mask = np.where((image_pm > 0))[0]
    return len(mask) > 0
