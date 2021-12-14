
import numpy as np


class RawCallback(object):
    @staticmethod
    def sensor_camera_rgb(weak_self, data):
        # data: carla.Image
        self = weak_self()
        self.raw_data = data
        self.data = data

    @staticmethod
    def sensor_lidar_ray_cast(weak_self, data):
        # data: carla.LidarMeasurement
        self = weak_self()
        self.raw_data = data
        self.data = data

    @staticmethod
    def sensor_other_gnss(weak_self, data):
        # data: carla.GNSSMeasurement
        self = weak_self()
        self.raw_data = data
        self.data = data

    @staticmethod
    def sensor_other_imu(weak_self, data):
        # data: carla.IMUMeasurement
        self = weak_self()
        self.raw_data = data
        self.data = data

    @staticmethod
    def sensor_other_collision(weak_self, data):
        # data: carla.CollisionEvent
        self = weak_self()
        self.raw_data = data
        self.data = data



class DefaultCallback(object):
    @staticmethod
    def sensor_camera_rgb(weak_self, data):
        # data: carla.Image
        self = weak_self()
        array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8")) 
        array = np.reshape(array, (data.height, data.width, 4)) # RGBA format
        self.raw_data = data
        self.data = array

    @staticmethod
    def sensor_lidar_ray_cast(weak_self, data):
        # data: carla.LidarMeasurement
        self = weak_self()
        self.raw_data = data
        self.data = data

    @staticmethod
    def sensor_other_gnss(weak_self, data):
        # data: carla.GNSSMeasurement
        self = weak_self()
        self.raw_data = data
        self.data = data

    @staticmethod
    def sensor_other_imu(weak_self, data):
        # data: carla.IMUMeasurement
        self = weak_self()
        self.raw_data = data
        self.data = data

    @staticmethod
    def sensor_other_collision(weak_self, data):
        # data: carla.CollisionEvent
        self = weak_self()
        self.raw_data = True
        self.data = data
        # print('collison: ', data.actor.id, data.actor.type_id, ' --with-- ', data.other_actor.id, data.other_actor.type_id)
