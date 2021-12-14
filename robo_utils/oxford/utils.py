import carla_utils as cu
import os, sys
import numpy as np
import copy
import cv2


'''https://robotcar-dataset.robots.ox.ac.uk/documentation/'''

def load_ldmrs(file_path, auto=False):
    '''
    According to build_pointcloud.py
    Returns:
        numpy.ndarray, (3 × N)
    '''
    pc = None
    with open(file_path) as f:
        pc = np.fromfile(f, np.double)
    pc = pc.reshape((len(pc) // 3, 3)).transpose()
    if auto:
        pc = -np.ascontiguousarray(pc[[1, 0, 2]].transpose().astype(np.float64))
    return pc


from .python.velodyne import load_velodyne_binary, load_velodyne_raw, velodyne_raw_to_pointcloud
def load_velodyne_png(file_path):
    ranges, intensities, angles, approximate_timestamps = load_velodyne_raw(file_path)
    return velodyne_raw_to_pointcloud(ranges, intensities, angles)


def read_extrinsics_file(extrinsics_path):
    with open(extrinsics_path) as extrinsics_file:
        extrinsics = [float(x) for x in next(extrinsics_file).split(' ')]
    return cu.basic.HomogeneousMatrix.xyzrpy(extrinsics)








def filter_ground(pc, height):
    z_array = pc[2,:]
    mask = np.where((z_array < height-0.5) & (z_array > height-3))[0]
    pc = pc[:,mask]

    i_array = pc[3,:]
    mask = np.where((i_array > 5))[0]
    pc = pc[:,mask]

    condition = (pc[0,:] > -0.89) & (pc[0,:] < 3.7) & (np.abs(pc[1,:]) < 1)
    mask = np.where((~condition))[0]
    pc = pc[:,mask]
    return pc


def get_costmap(param, img, pc):
    point_cloud = copy.deepcopy(pc)
    # point_cloud[1,:] *= -1
    width = param.image_width
    height = param.image_height
    
    img2 = np.zeros((height, width), np.uint8)
    img2.fill(255)
    # res = np.where((point_cloud[2] > -1.95))
    # point_cloud = point_cloud[:, res[0]]
    
    pixs_per_meter = height / param.longitudinal_length
    u = (height-point_cloud[0]*pixs_per_meter).astype(int)
    v = (point_cloud[1]*pixs_per_meter+width//2).astype(int)
    
    mask = np.where((u >= 0)&(u < height))[0]
    u = u[mask]
    v = v[mask]
    
    mask = np.where((v >= 0)&(v < width))[0]
    u = u[mask]
    v = v[mask]

    img2[u,v] = 0
    kernel = np.ones((13,13),np.uint8)  
    img2 = cv2.erode(img2,kernel,iterations = 1)
    
    kernel_size = (3, 3)
    img = cv2.dilate(img,kernel_size,iterations = 3)
    
    img = cv2.addWeighted(img,0.5,img2,0.5,0)
    
    mask = np.where((img2 < 50))
    u = mask[0]
    v = mask[1]
    img[u, v] = 0
    # kernel_size = (17, 17)
    # kernel_size = (9, 9)
    # sigma = 9#21
    # img = cv2.GaussianBlur(img, kernel_size, sigma)
    return img