from carla_utils import carla

import numpy as np

from .state import State
from .waypoint import Waypoint



class CarlaTransform(object):
    @staticmethod
    def cua_state(state: State):
        x, y, theta = state.x, state.y, state.theta
        location = carla.Location(x=x, y=y)
        transform = carla.Transform(location)
        transform.rotation.yaw = np.rad2deg(theta)
        return transform


class CUAState(object):
    @staticmethod
    def carla_transform(transform, **kwargs):
        """
        
        Args:
            transform: carla.Transform, need to be valid or real
            kwargs: contains except x, y, z, theta
        
        Returns:
            carla_utils.augment.State
        """
        
        location, rotation = transform.location, transform.rotation
        x, y, z = location.x, location.y, location.z
        theta = np.deg2rad(rotation.yaw)
        kwargs.update({'x':x, 'y':y, 'z':z, 'theta':theta})
        return State(**kwargs)

