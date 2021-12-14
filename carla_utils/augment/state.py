
import time
import numpy as np
import matplotlib.pyplot as plt

from .. import basic
from .tools import vector3DToArray


def getActorState(frame_id, timestamp, actor):
    location = actor.get_location()
    x, y, z = location.x, location.y, location.z
    theta = np.deg2rad(actor.get_transform().rotation.yaw)
    velocity = vector3DToArray(actor.get_velocity())
    v = np.linalg.norm(velocity)
    acceleration = vector3DToArray(actor.get_acceleration())
    a = np.linalg.norm(acceleration)
    return State(x=x, y=y, z=z, theta=theta, v=v, a=a)


get_actor_state = getActorState


class State(object):
    """
        2 dimension.
    """
    def __init__(self, **kwargs):
        self.x = float(kwargs.get('x', 0))
        self.y = float(kwargs.get('y', 0))
        self.z = float(kwargs.get('z', 0))

        self.theta = basic.pi2pi(float(kwargs.get('theta', 0)))

        self.k = float(kwargs.get('k', 0))

        self.s = float(kwargs.get('s', 0))
        self.v = float(kwargs.get('v', 0))
        self.a = float(kwargs.get('a', 0))
        self.t = float(kwargs.get('t', 0))

        self.velocity = kwargs.get('velocity', np.zeros((3,1))).astype(np.float64)
        self.acceleration = kwargs.get('acceleration', np.zeros((3,1))).astype(np.float64)
        

    def __str__(self):
        obj = 'State(x={}, y={}, theta={}, v={})'.format(self.x, self.y, self.theta, self.v)
        return obj

    def __repr__(self):
        return str(self)


    def distance_xyz(self, state):
        dx = self.x - state.x
        dy = self.y - state.y
        dz = self.z - state.z
        return np.sqrt(dx**2 + dy**2 + dz**2)

    def distance_xy(self, state):
        dx = self.x - state.x
        dy = self.y - state.y
        return np.sqrt(dx**2 + dy**2)

    def distance_xytheta(self, state):
        dx = self.x - state.x
        dy = self.y - state.y
        dtheta = self.theta - state.theta
        return np.sqrt(dx**2 + dy**2 + dtheta**2)

    def delta_theta(self, state):
        dx = self.x - state.x
        dy = self.y - state.y
        return np.arctan2(dy, dx)


    def find_nearest_state_in(self, states):
        min_dist = float('inf')
        min_state = None
        for state in states:
            d = self.distance(state)
            if d < min_dist:
                min_dist = d
                min_state = state
        return min_state

    def find_nearest_waypoint_in(self, waypoints):
        mind = float('inf')
        nearest_wp = None
        for waypoint in waypoints:
            d = self.distance(waypoint)
            if d < mind:
                mind = d
                nearest_wp = waypoint
        return nearest_wp, mind


    def world2local(self, state0):
        '''
            2-dimension
            state0: world coordinate
        '''

        x_world, y_world, theta_world = self.x, self.y, self.theta
        x0, y0, theta0 = state0.x, state0.y, state0.theta
        delta_theta = basic.pi2pi(self.theta - state0.theta)

        x_local = (x_world-x0)*np.cos(theta0) + (y_world-y0)*np.sin(theta0)
        y_local =-(x_world-x0)*np.sin(theta0) + (y_world-y0)*np.cos(theta0)

        local_state = State(
                x=x_local, y=y_local, theta=delta_theta,
                z=self.z, k=self.k, s=self.s, v=self.v, t=self.t
                )
        return local_state


    def local2world(self, state0):

        x_local, y_local, theta_local = self.x, self.y, self.theta
        x0, y0, theta0 = state0.x, state0.y, state0.theta

        x_world = x0 + x_local*np.cos(theta0) - y_local*np.sin(theta0)
        y_world = y0 + x_local*np.sin(theta0) + y_local*np.cos(theta0)
        theta_world = basic.pi2pi(theta_local + theta0)

        world_state = State(
                x=x_world, y=y_world, theta=theta_world,
                z=self.z, k=self.k, s=self.s, v=self.v, a=self.a, t=self.t
                )
        return world_state


    def numpy(self):
        return np.array([self.x, self.y, self.theta, self.v], dtype=np.float32)


    def draw_plt(self, style='arrow', fmt='og', color='red'):
        if style == 'point':
            plt.plot(self.x, self.y, fmt, color=color)
        elif style == 'arrow':
            basic.plotArrow2D(self, fc=color, ec=color)

