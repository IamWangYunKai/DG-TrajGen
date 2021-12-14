import carla_utils as cu

import cv2
import numpy as np
import copy
import matplotlib.pyplot as plt

'''
    2013-Multimodal inverse perspective mapping
'''

class Reverse(object):
    @staticmethod
    def x():
        return np.diag([-1,1,1])
    @staticmethod
    def y():
        return np.diag([1,-1,1])
    @staticmethod
    def z():
        return np.diag([1,1,-1])

def image2DToWorld3D(image_vec, K, R, t, z):
    number = image_vec.shape[1]
    a, b, c, d = 0, 0, 1, z

    K3, K4 = K[:,:3], np.vstack((K, np.array([0,0,0,1])))
    r = K[:,-1][:,np.newaxis]

    P = np.dot(K3, R) + np.dot(r, np.array([a,b,c]).reshape(1,3))
    t = np.dot(K4, np.vstack((copy.deepcopy(t), d)))

    image_vec_group = np.expand_dims(np.vstack((image_vec, np.ones((1,number)))), axis=1)
    image_vec_group = np.transpose(image_vec_group, (2,0,1))
    P_group = np.expand_dims(P, axis=0).repeat(number, axis=0)

    A = np.concatenate((P_group, -image_vec_group), axis=2)
    plane = np.expand_dims(np.array([a,b,c,0]).reshape(1,4), axis=0).repeat(number, axis=0)
    A = np.concatenate((A, plane), axis=1)

    world_vec = np.dot(np.linalg.inv(A), -t)
    world_vec = np.squeeze(world_vec[:,:3], axis=2).T
    return world_vec


class InversePerspectiveMapping(object):
    def __init__(self, param, K, T_img_imu, z, reverse_y=True):
        '''
            K: numpy.ndarray (3×4)
            z: height from imu to ground
        '''

        '''ipm'''
        self.R, self.t = cu.basic.RotationMatrixTranslationVector.homogeneous_matrix(T_img_imu)
        self.K = K
        self.z = z
        
        '''raster'''
        self.img_width = param.ipm.image_width
        self.img_height = param.ipm.image_height
        self.ksize = param.ipm.kernel_size

        f = float(self.img_height) / param.ipm.longitudinal_length
        self.pesudo_K = np.array([  [f, 0, self.img_width/2],
                                    [0, f,  self.img_height],
                                    [0, 0,                1] ])
        reverse = Reverse.y() if reverse_y else np.identity(3)
        self.rotate = cu.basic.np_dot(cu.basic.RotationMatrix.yaw(-np.pi/2), reverse)

    
    def get(self, image, pose_array, T_w_imu):
        '''
            image: perspective mapping image, numpy.ndarray (height, width, channel)
        '''
        if image is None:
            return None
        index_array = np.argwhere(image > 200)
        index_array = index_array[:,:2]
        index_array = np.unique(index_array, axis=0)
        index_array = np.array([index_array[:,1], index_array[:,0]])

        vehicle_vec = image2DToWorld3D(index_array, self.K, self.R, self.t, self.z)
        vehicle_vec[2,:] = 1.0
        return self.raster(vehicle_vec)
    

    def raster(self, world_vec):
        new_image_vec = cu.basic.np_dot(self.pesudo_K, self.rotate, world_vec)
        new_image_vec = new_image_vec[:2,:]
        new_image_vec = new_image_vec[::-1,:]

        new_image_y_pixel = new_image_vec[0,:].astype(int)
        new_image_x_pixel = new_image_vec[1,:].astype(int)

        new_image = np.zeros((self.img_height, self.img_width), dtype=np.dtype("uint8"))

        mask = np.where((new_image_x_pixel >= 0)&(new_image_x_pixel < self.img_width))[0]
        new_image_x_pixel = new_image_x_pixel[mask]
        new_image_y_pixel = new_image_y_pixel[mask]
        
        mask = np.where((new_image_y_pixel >= 0)&(new_image_y_pixel < self.img_height))[0]
        new_image_x_pixel = new_image_x_pixel[mask]
        new_image_y_pixel = new_image_y_pixel[mask]
        new_image[new_image_y_pixel, new_image_x_pixel] = 255

        new_image[np.clip(new_image_y_pixel+1,0, self.img_height-1),new_image_x_pixel] = 255
        new_image[np.clip(new_image_y_pixel-1,0, self.img_height-1),new_image_x_pixel] = 255
        
        # new_image = cv2.GaussianBlur(new_image, (self.ksize, self.ksize), 25)
        return new_image
