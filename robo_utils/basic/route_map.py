import carla_utils as cu

import numpy as np
import copy
from PIL import Image
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

class RouteMapUtils(object):
    def __init__(self, param):
        self.resolution = param.resolution
        self.padding_length = param.padding_length
        self.vehicle_width = param.vehicle_width
        self.correction_length = param.correction_length
        self.lateral_step_factor = param.lateral_step_factor
    

    def get(self, pose_array):
        augmented = data_augmentation(pose_array, self.resolution/2)
        point1, point2 = utils.expand_lateral(augmented, self.vehicle_width, self.correction_length)
        point1, point2 = point1[[0,1,3],:], point2[[0,1,3],:]

        '''transform from world to map (image)'''
        x_min, x_max = min(augmented[0,:]), max(augmented[0,:])
        y_min, y_max = min(augmented[1,:]), max(augmented[1,:])
        image_width = int( (x_max-x_min+2*self.padding_length)/self.resolution )
        image_height= int( (y_max-y_min+2*self.padding_length)/self.resolution )
        route_map = np.full((image_height, image_width, 3), 255, dtype=np.uint8)

        t = -np.array([x_min-self.padding_length, y_max+self.padding_length]).reshape(2,1)
        K = np.diag([1/self.resolution, 1/self.resolution, 1])
        Rev = np.diag([1,-1,1])
        T = cu.basic_tools.np_dot(Rev, K, cu.basic_tools.HomogeneousMatrix2D(np.eye(2), t))
        pixel_array1, pixel_array2 = np.dot(T, point1)[:2,:], np.dot(T, point2)[:2,:]

        for i in range(augmented.shape[1]):
            pixel1, pixel2 = pixel_array1[:,i][:,np.newaxis], pixel_array2[:,i][:,np.newaxis]
            pixel1, pixel2 = pixel1[::-1,:], pixel2[::-1,:]
            utils.draw_line(route_map, pixel1, pixel2, color='red', step_factor=self.lateral_step_factor)

        return route_map
