import carla_utils as cu

import os
from os.path import join
import glob
import numpy as np
import time
from tqdm import tqdm
import cv2
from PIL import Image

from ..basic import utils as basic_utils
from ..basic import perspective_mapping as pm
from  ..basic import inverse_perspective_mapping as ipm
from ..basic import nav_map as nm
from .partial import PartialDataset, find_nearest
from .utils import filter_ground, get_costmap


class PartialDatasetAugment(PartialDataset):
    imu_height = 0.45   # imu height w.r.t. ground
    def __init__(self, param, path, process, pc_type='png'):
        PartialDataset.__init__(self, path, process, pc_type)

        self.param = param
        self.save_path = join(path, 'augment')
        cu.system.mkdir(self.save_path)
        self.process = process

        self.imu_height = PartialDatasetAugment.imu_height
        self.trajectory_length = param.trajectory_length

        '''reference timestamps'''
        try:
            self.ref_timestamp_array = np.loadtxt(join(self.save_path, 'reference.timestamps'), delimiter=' ', usecols=[], dtype=np.int64)
        except IOError:
            max_timestamp = min(self.velodyne_left_timestamp_array[-1],
                                self.velodyne_right_timestamp_array[-1],
                                self.stereo_centre_timestamp_array[-1],
                                self.ins_df['timestamp'].values[-1],
                                self.vo_df['destination_timestamp'].values[-1])
            min_timestamp = max(self.velodyne_left_timestamp_array[0],
                                self.velodyne_right_timestamp_array[0],
                                self.stereo_centre_timestamp_array[0],
                                self.ins_df['timestamp'].values[0],
                                self.vo_df['destination_timestamp'].values[0])
            mask = np.argwhere((self.stereo_centre_timestamp_array >= min_timestamp)&(self.stereo_centre_timestamp_array <= max_timestamp))[:,0]
            self.ref_timestamp_array = self.stereo_centre_timestamp_array[mask]
            np.savetxt(join(self.save_path, 'reference.timestamps'), self.ref_timestamp_array, delimiter=' ', fmt='%d')

        '''trajectory'''
        try:
            self.lengths = np.loadtxt(join(self.save_path, 'lengths.txt'), delimiter=' ', usecols=[], dtype=np.float64)
            self.delta_pose_array = np.loadtxt(join(self.save_path, 'delta_pose_array.txt'), delimiter=' ', usecols=[], dtype=np.float64).T
        except IOError:
            min_timestamp, max_timestamp = min(self.ref_timestamp_array), max(self.ref_timestamp_array)
            df = self.vo_df[(self.vo_df['destination_timestamp'] >= min_timestamp) & (self.vo_df['destination_timestamp'] <= max_timestamp)]
            x_array, y_array, z_array = df['x'].values, df['y'].values, df['z'].values
            self.lengths = np.sqrt(x_array**2 + y_array**2 + z_array**2)
            self.delta_pose_array = np.vstack((x_array, y_array, z_array, df['roll'].values, df['pitch'].values, df['yaw'].values))

            np.savetxt(join(self.save_path, 'lengths.txt'), self.lengths, delimiter=' ', fmt='%f')
            np.savetxt(join(self.save_path, 'delta_pose_array.txt'), self.delta_pose_array.T, delimiter=' ', fmt='%f')
        
        '''trajectory type'''
        try:
            self.trajectory_types = np.loadtxt(join(self.save_path, 'trajectory_types.txt'), delimiter=' ', usecols=[], dtype=np.int64)
        except IOError:
            t1 = time.time()
            self.trajectory_types = self.generate_trajectory_types()
            np.savetxt(join(self.save_path, 'trajectory_types.txt'), self.trajectory_types, delimiter=' ', fmt='%d')
            t2 = time.time()
            print('traj type: ', t2-t1)

        '''pm and ipm'''
        self.pm = pm.PerspectiveMapping(param, process.K, process.T_img_imu)
        self.ipm = ipm.InversePerspectiveMapping(param, process.K, process.T_img_imu, z=self.imu_height, reverse_y=False)
        self.nm = nm.NavMapUtils(param)


    def __len__(self):
        return len(self.ref_timestamp_array)
    
    @property
    def length_valid(self):
        return len(glob.glob(join(self.save_path, 'costmap', '*')))

    def generate_pose_array(self, ref_timestamp, horizon=None):
        index = int(np.argwhere((self.ref_timestamp_array == ref_timestamp)))
        if horizon is None:
            horizon = basic_utils.find_desired_length(self.lengths, index, self.trajectory_length)
        if horizon <= 0:
            print('no enough trajectory length 3')
            horizon = 0
        delta_pose_array = self.delta_pose_array[:,index:index+horizon]
        pose_array = cum_vo(delta_pose_array, self.imu_height)[:,:-1]
        return pose_array
    

    def generate_trajectory_types(self):
        types = []
        # for ref_timestamp in self.ref_timestamp_array:
        for i in tqdm(range(len(self))):
        #     if i < 33000:
        #         continue
            ref_timestamp = self.ref_timestamp_array[i]

            pose_array = self.generate_pose_array(ref_timestamp)
            if pose_array.shape[1] == 0:
                break
            delta_yaw = pose_array[-1,:] - pose_array[-1,0]
            # np.set_printoptions(threshold=np.inf, precision=10, linewidth=65535)
            avg = np.average(delta_yaw)
            if abs(avg) <= self.param.pm.type_threshold:
                type_id = 0
            elif avg > self.param.pm.type_threshold:
                type_id = 1
            else:
                type_id = 2
            types.append([ref_timestamp, type_id])
        return np.array(types)
    

    def generate_costmap(self, ref_timestamp, pose_array):
        points = basic_utils.get_trajectory_points(pose_array, self.param.vehicle_width, self.param.correction_length, resolution=0.1)
        img_ipm = self.ipm.raster(points[[0,1,3],:])
        
        pointcloud = self.pointcloud(ref_timestamp)
        pc = basic_utils.transform_pointcloud(self.process.T_imu_cam, pointcloud)
        pc = filter_ground(pc, self.imu_height)
        return get_costmap(self.param.ipm, img_ipm, pc)

    def save_costmap(self, ref_timestamp, pose_array):
        costmap = self.generate_costmap(ref_timestamp, pose_array)
        dir_path = join(self.save_path, 'costmap')
        cu.system.mkdir(dir_path)
        image_path = join(dir_path, str(ref_timestamp)+'.png')
        cv2.imwrite(image_path, costmap)
    
    def get_costmap(self, ref_timestamp, mode='L'):
        image_path = join(self.save_path, 'costmap', str(ref_timestamp)+'.png')
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = Image.fromarray(image)
        return image

    def generate_nav_map(self, pose_array, T_w_imu):
        nav_map = self.nm.trajectory_pose(pose_array, T_w_imu)
        return nav_map
    
    def save_nav_map(self, ref_timestamp, pose_array, T_w_imu):
        nav_map = self.generate_nav_map(pose_array, T_w_imu)
        dir_path = join(self.save_path, 'nav_map')
        cu.system.mkdir(dir_path)
        image_path = join(dir_path, str(ref_timestamp)+'.png')
        cv2.imwrite(image_path, nav_map)
    
    def get_nav_map(self, ref_timestamp, mode='L'):
        image_path = join(self.save_path, 'nav_map', str(ref_timestamp)+'.png')
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = image.crop((50, image.size[1]//2, image.size[0]-50, image.size[1]))
        # print('nav:', image.size)
        return image
    
    def generate_image_pm(self, pose_array, T_w_imu, image=None):
        image_pm = self.pm.trajectory_pose(pose_array, T_w_imu, image)
        return image_pm
    
    def save_image_pm(self, ref_timestamp, pose_array, T_w_imu):
        image_pm = self.generate_image_pm(pose_array, T_w_imu)
        dir_path = join(self.save_path, 'image_pm')
        cu.system.mkdir(dir_path)
        image_path = join(dir_path, str(ref_timestamp)+'.png')
        image_pm = Image.fromarray(image_pm)
        image_pm = image_pm.crop([0,200, 1280,960])
        image_pm = image_pm.resize((self.param.image.image_width, self.param.image.image_height))
        image_pm.save(image_path)
    
    def get_image_pm(self, ref_timestamp):
        image_path = join(self.save_path, 'image_pm', str(ref_timestamp)+'.png')
        image = Image.open(image_path).convert("L")
        # image = image.resize((400, 200))
        # image = image.crop((120, image.size[1]//2-50, image.size[0]-120, image.size[1]))
        return image

    def generate_stereo_image(self, ref_timestamp):
        return self.image(ref_timestamp)

    def save_stereo_image(self, ref_timestamp):
        image = self.generate_stereo_image(ref_timestamp)
        image[:,:,[0,2]] = image[:,:,[2,0]]
        dir_path = join(self.save_path, 'image_ed')
        cu.system.mkdir(dir_path)
        image_path = join(dir_path, str(ref_timestamp)+'.png')
        image = Image.fromarray(image)
        image = image.crop([0,200, 1280,960])
        image = image.resize((self.param.image.image_width, self.param.image.image_height))
        image.save(image_path)

    def get_image(self, ref_timestamp, crop=True):
        if crop:
            image_path = join(self.save_path, 'image_ed', str(ref_timestamp)+'.png')
        else:
            image_path = join(self.save_path, 'image', str(ref_timestamp)+'.png')
        # image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        # image = Image.fromarray(image)
        image = Image.open(image_path)
        # image = image.crop((50, image.size[1]//2, image.size[0]-50, image.size[1]))
        # print('image', image.size)
        return image
        
    def get_pcd(self, ref_timestamp):
        # points = basic_utils.get_trajectory_points(pose_array, self.param.vehicle_width, self.param.correction_length, resolution=0.1)
        # img_ipm = self.ipm.raster(points[[0,1,3],:])
        
        pointcloud = self.pointcloud(ref_timestamp)
        pc = basic_utils.transform_pointcloud(self.process.T_imu_cam, pointcloud)
        # pc = filter_ground(pc, self.imu_height)
        # return get_costmap(self.param.ipm, img_ipm, pc)
        pc = np.array([pc[0], pc[1], -pc[2]]).astype(np.float32)
        return pc


    def get_velocity(self, ref_timestamp):
        timestamp = find_nearest(ref_timestamp, self.ins_timestamp_array)
        df = self.ins_df
        df = df[df['timestamp'] == timestamp]
        vx, vy = float(df['velocity_north'].values), float(df['velocity_east'].values)
        return vx, vy

    def get_yaw(self, ref_timestamp):
        timestamp = find_nearest(ref_timestamp, self.ins_timestamp_array)
        df = self.ins_df
        df = df[df['timestamp'] == timestamp]
        return float(df['yaw'].values)

    
    def get_trajectory_type(self, ref_timestamp):
        mask = int(np.argwhere((self.trajectory_types[:,0] == ref_timestamp)))
        return self.trajectory_types[mask,1]



def cum_vo(delta_pose_array, imu_height):
    num = delta_pose_array.shape[1]
    delta_point_array = np.vstack((delta_pose_array[:3,:], np.ones((1,num))))
    delta_euler_array = delta_pose_array[3:,:]

    pose_array = np.array([0.,0,0,0,0,0]).reshape(6,1)
    for i in range(num):
        p0 = pose_array[:,-1]
        T = cu.basic.HomogeneousMatrix.xyzrpy(p0)
        p = np.dot(T, delta_point_array[:,i])
        e = p0[3:] + delta_euler_array[:,i]
        pose_array = np.hstack((pose_array, np.vstack((p[:3], e)).reshape(6,1)))
    pose_array[2,:] += imu_height
    # pose_array[2,:] -= imu_height
    pose_array[3:,:] = cu.basic.pi2pi(pose_array[3:,:])
    return pose_array