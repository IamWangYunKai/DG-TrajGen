from carla_utils import carla

import numpy as np

from .sensor_callback import DefaultCallback


image_callback = DefaultCallback.sensor_camera_rgb
    
def lidar_callback(weak_self, data):
    # data: carla.LidarMeasurement
    self = weak_self()
    lidar_data = np.frombuffer(data.raw_data, dtype=np.float32).reshape([-1, 3])
    point_cloud = np.stack([-lidar_data[:,1], -lidar_data[:,0], -lidar_data[:,2]])
    mask = np.where((point_cloud[0] > 1.0)|(point_cloud[0] < -4.0)|(point_cloud[1] > 1.2)|(point_cloud[1] < -1.2))[0]
    point_cloud = point_cloud[:, mask]
    mask = np.where(point_cloud[2] > -1.95)[0]
    point_cloud = point_cloud[:, mask]
    self.raw_data = data
    self.data = point_cloud


def lidar_callback_2(weak_self, data):
    self = weak_self()
    lidar_data = np.frombuffer(data.raw_data, dtype=np.float32).reshape([-1, 3])
    self.raw_data = data
    self.data = lidar_data


def collision_callback(weak_self, data):
    # data: event
    self = weak_self()
    impulse = data.normal_impulse
    intensity = np.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
    self.raw_data = data
    if self.data:
        self.data.append((data.frame, data, intensity))
    else:
        self.data = [(data.frame, data, intensity)]
    # print('collison: ', data.actor.id, data.actor.type_id, ' --with-- ', data.other_actor.id, data.other_actor.type_id)



sensors_params = [
    {
        'type_id': 'sensor.other.collision',
        'role_name': 'default',
        'transform':carla.Transform(carla.Location()),
        'callback':collision_callback,
    },

    # {
    #     'type_id': 'sensor.lidar.ray_cast',
    #     'role_name': 'front',
    #     'channels': 32,
    #     'rpm': 10,
    #     'pps': 172800,
    #     'range': 50,
    #     'lower_fov': -30,
    #     'upper_fov': 10,
    #     'sensor_tick': 0.45,
    #     'transform':carla.Transform(carla.Location(x=0.5, z=2.5)),
    #     'callback':lidar_callback,
    # },
    # {
    #     'type_id': 'sensor.lidar.ray_cast',
    #     'role_name': 'front2',
    #     'channels': 32,
    #     'rpm': 10,
    #     'pps': 172800,
    #     'range': 50,
    #     'lower_fov': -30,
    #     'upper_fov': 10,
    #     'sensor_tick': 0.45,
    #     'transform':carla.Transform(carla.Location(x=0.5, z=2.5)),
    #     'callback':lidar_callback_2,
    # },
    # {
    #     'type_id': 'sensor.lidar.ray_cast',
    #     'role_name': 'front3',
    #     'channels': 32,
    #     'rpm': 10,
    #     'pps': 172800,
    #     'range': 50,
    #     'lower_fov': -30,
    #     'upper_fov': 10,
    #     'sensor_tick': 0.45,
    #     'transform':carla.Transform(carla.Location(x=0.5, z=2.5)),
    #     'callback':lidar_callback,
    # },

    {
        'type_id': 'sensor.camera.rgb',
        'role_name': 'view',
        'image_size_x': 640 *2,
        'image_size_y': 360 *2,
        'fov': 120,
        'sensor_tick': 1/20,
        'transform':carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
        'callback':image_callback,
    },

    {
        'type_id': 'sensor.camera.rgb',
        'role_name': 'bird',
        'image_size_x': 640 *2,
        'image_size_y': 360 *2,
        'fov': 90,
        'sensor_tick': 1/200,
        'transform':carla.Transform(carla.Location(x=30.0, z=40.0), carla.Rotation(pitch=-90, yaw=90)),
        'callback':image_callback,
    },

    # {
    #     'type_id': 'sensor.camera.semantic_segmentation',
    #     'role_name': 'bird',
    #     'image_size_x': 640 *2,
    #     'image_size_y': 360 *2,
    #     'fov': 90,
    #     'sensor_tick': 1/200,
    #     'transform':carla.Transform(carla.Location(x=30.0, z=40.0), carla.Rotation(pitch=-90, yaw=90)),
    #     'callback':image_callback,
    # },

]