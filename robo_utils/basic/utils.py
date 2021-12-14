import carla_utils as cu

import numpy as np
import matplotlib.pyplot as plt
import copy


def linear_interpolation(pose1, pose2, min_dist):
    '''
    Args:
        pose1, pose2: xyzrpy, numpy.ndarray (6×1)
    '''
    distance = np.linalg.norm(pose1[:3,0] - pose2[:3,0])
    total = int(distance / min_dist)
    if total <= 0:
        total = 1
    t_array = np.arange(total) / total
    interpolated_pose_array = (1-t_array) * pose1 + (t_array) * pose2
    interpolated_pose_array[3:,:] = cu.basic.pi2pi(interpolated_pose_array[3:,:])
    return interpolated_pose_array


def draw_line(image, pixel1, pixel2, color, step_factor):
        length = np.linalg.norm(pixel2 - pixel1)
        direction = (pixel2 - pixel1) / length
        lateral_sample_number = round(length / step_factor) + 1
        distance_array = np.linspace(0, length, lateral_sample_number)
        
        pixel_vec = pixel1 + distance_array * direction

        x_pixel = pixel_vec.astype(int)[0]
        y_pixel = pixel_vec.astype(int)[1]

        mask = np.where((x_pixel >= 0)&(x_pixel < image.shape[0]))[0]
        x_pixel = x_pixel[mask]
        y_pixel = y_pixel[mask]
        
        mask = np.where((y_pixel >= 0)&(y_pixel < image.shape[1]))[0]
        x_pixel = x_pixel[mask]
        y_pixel = y_pixel[mask]

        image[x_pixel, y_pixel, 2] = color[0]
        image[x_pixel, y_pixel, 1] = color[1]
        image[x_pixel, y_pixel, 0] = color[2]
        return



def _get_trajectory_length_array(pose_array):
    point_array = pose_array[:3,:].T
    n = point_array.shape[0]
    M = np.eye(n-1,n, k=0) - np.eye(n-1,n, k=1)
    return np.sqrt(np.sum(np.square(np.dot(M, point_array)), 1))

def get_trajectory_length_array(pose_array):
    size = pose_array.shape[1]
    lengths = []
    for i in range(size-1):
        lengths.append(np.linalg.norm(pose_array[:3,i]-pose_array[:3,i+1]))
    return np.array(lengths)



def find_desired_length(trajectory_length_array, start_index, desired_length):
    # print(len(trajectory_length_array), start_index, desired_length)
    horizon = -1
    for i in range(len(trajectory_length_array)-start_index):
        length = sum(trajectory_length_array[start_index:start_index+i])
        if length >= desired_length:
            horizon = i
            break
    return horizon


def draw_gps(dataset, start=0, end=-1):
    x_list, y_list = [], []
    for i in range(len(dataset)):
        _, t = cu.basic.RotationMatrixTranslationVector.homogeneous_matrix(dataset.oxts[i].T_w_imu)
        x_list.append(t[0,0])
        y_list.append(t[1,0])
    plt.plot(x_list[start:end], y_list[start:end], 'or')
    plt.gca().set_aspect('equal', adjustable='box')


def expand_lateral(pose_array, vehicle_width, correction_length):
    '''
    Args:
        pose_array: xyzrpy, numpy.ndarray (6, N)
    Returns:
        homogeneous points, numpy.ndarray (4, N)
    '''
    yaw_array = pose_array[5,:][np.newaxis,:] + np.pi/2
    direction_array = np.vstack((np.cos(yaw_array), np.sin(yaw_array), np.zeros((1,pose_array.shape[1]))))

    one_row = np.ones((1,pose_array.shape[1]))
    point1 = np.vstack((pose_array[:3,:] + (vehicle_width/2-correction_length) * direction_array, one_row))
    point2 = np.vstack((pose_array[:3,:] - (vehicle_width/2+correction_length) * direction_array, one_row))
    return point1, point2

def _data_augmentation(pose_array):
    number = pose_array.shape[1]
    augmented = None
    for i in range(number-1):
        if i/number < 0.4:
            min_dist = 0.014
        elif i/number < 0.7:
            min_dist = 0.03
        else:
            min_dist = 0.09
        
        p1 = pose_array[:,i][:,np.newaxis]
        p2 = pose_array[:,i+1][:,np.newaxis]
        interpolated = linear_interpolation(p1, p2, min_dist)
        if augmented is None:
            augmented = interpolated
        else:
            augmented = np.hstack((augmented, interpolated))
    return augmented


def get_trajectory_points(pose_array, vehicle_width, correction_length, resolution):
    augmented = _data_augmentation(pose_array)
    point1, point2 = expand_lateral(augmented, vehicle_width, correction_length)

    num = round(vehicle_width / resolution)
    lambda_array = np.linspace(0,1, num).reshape(num,1,1).repeat(4, axis=1).repeat(point1.shape[1], axis=2)

    point1 = np.expand_dims(point1, axis=0).repeat(num, axis=0)
    point2 = np.expand_dims(point2, axis=0).repeat(num, axis=0)

    np.multiply(lambda_array, point2)

    point_stack = np.multiply(1-lambda_array, point1) + np.multiply(lambda_array, point2)
    points = np.transpose(point_stack, [1,0,2]).reshape(4,-1)
    return points


def transform_pointcloud(T, pointcloud):
    '''
    Args:
        pointcloud: numpy.ndarray (4×N), in frame A
        T: transform from A to B
    Returns:
        pointcloud in frame B
    '''
    assert pointcloud.shape[0] == 4
    intensity_array = copy.deepcopy(pointcloud[3,:])
    pointcloud[3,:] = 1
    pointcloud = np.dot(T, pointcloud)
    pointcloud[3,:] = intensity_array
    return pointcloud
