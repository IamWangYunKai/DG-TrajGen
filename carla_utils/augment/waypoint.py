
import numpy as np


class Waypoint(object):
    def __init__(self, carla_waypoint, **kwargs):
        self.x = carla_waypoint.transform.location.x
        self.y = carla_waypoint.transform.location.y
        self.z = carla_waypoint.transform.location.z
        self.theta = np.deg2rad(carla_waypoint.transform.rotation.yaw)
        self.lane_id = kwargs.get('lane_id', 0)
        self.step = kwargs.get('step', 0)
        self.reference = kwargs.get('reference', False)

        # for visualization
        self.roll_rad = np.deg2rad(carla_waypoint.transform.rotation.roll)
        self.pitch_rad= np.deg2rad(carla_waypoint.transform.rotation.pitch)
        self.yaw_rad  = np.deg2rad(carla_waypoint.transform.rotation.yaw)


    def distance_xyz(self, waypoint):
        dx = self.x - waypoint.x
        dy = self.y - waypoint.y
        dz = self.z - waypoint.z
        return np.sqrt(dx**2 + dy**2 + dz**2)
    def distance_xy(self, waypoint):
        dx = self.x - waypoint.x
        dy = self.y - waypoint.y
        return np.sqrt(dx**2 + dy**2)

