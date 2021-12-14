from carla_utils import carla

import numpy as np
from shapely.geometry import Polygon

from .. import basic


# ==============================================================================
# -- vector operation ----------------------------------------------------------
# ==============================================================================

def vector3DEMul(vec1, vec2):
    '''
        vec1, vec2: carla.Vector3D
    '''
    return vec1.x*vec2.x + vec1.y*vec2.y + vec1.z*vec2.z
def vector2DEMul(vec1, vec2):
    '''
        vec1, vec2: carla.Vector3D
    '''
    return vec1.x*vec2.x + vec1.y*vec2.y

def vector3DNorm(vec):
    return np.sqrt(vec.x**2 + vec.y**2 + vec.z**2)
def vector2DNorm(vec):
    return np.sqrt(vec.x**2 + vec.y**2)


def vectorYawRad(vec):
    vec_norm = vector2DNorm(vec)
    if vec_norm < 0.001:
        return 0.0
    v = vec / vec_norm
    x, y = v.x, v.y
    yaw_rad = np.arctan2(y, x)
    return yaw_rad


def vector3DToArray(vec):
    return np.array([vec.x, vec.y, vec.z]).reshape(3,1)



# ==============================================================================
# -- calculate error -----------------------------------------------------------
# ==============================================================================

def error_state(current_state, target_state):
    xr, yr, thetar = target_state.x, target_state.y, target_state.theta
    theta_e = basic.pi2pi(current_state.theta - thetar)

    d = (current_state.x - xr, current_state.y - yr)
    t = (np.cos(thetar), np.sin(thetar))

    longitudinal_e, lateral_e = _cal_long_lat_error(d, t)
    return longitudinal_e, lateral_e, theta_e

def error_transform(current_transform, target_transform):    
    xr, yr, thetar = target_transform.location.x, target_transform.location.y, np.deg2rad(target_transform.rotation.yaw)
    theta_e = basic.pi2pi(np.deg2rad(current_transform.rotation.yaw) - thetar)

    d = (current_transform.location.x - xr, current_transform.location.y - yr)
    t = (np.cos(thetar), np.sin(thetar))

    longitudinal_e, lateral_e = _cal_long_lat_error(d, t)
    return longitudinal_e, lateral_e, theta_e

def _cal_long_lat_error(d, t):
    '''
        Args:
            d, t: array-like
    '''
    dx, dy = d[0], d[1]
    tx, ty = t[0], t[1]
    longitudinal_e = dx*tx + dy*ty
    lateral_e = dx*ty - dy*tx
    return longitudinal_e, lateral_e


def distance_waypoint(waypoint1, waypoint2):
    '''
        Calculate distance between waypoin1 and waypoint2.
    '''
    d = waypoint1.transform.location.distance(waypoint2.transform.location)
    return d

def distance2d(l1: carla.Location, l2: carla.Location):
    return np.hypot(l1.x - l2.x, l1.y - l2.y)



class ArcLength(object):
    @staticmethod
    def states(current_state, target_state):
        d = current_state.distance_xy(target_state)
        theta = basic.pi2pi(current_state.theta - target_state.delta_theta(current_state))
        if abs(theta) < 0.001:
            return d, 0.0
        l = d * abs(theta / np.sin(theta))
        curvature = -2*np.sin(theta) / d
        return l, curvature




def get_actor_radius(actor : carla.Actor):
    bounding_box = actor.bounding_box.extent
    radius = np.hypot(bounding_box.x, bounding_box.y)
    return radius


class ActorVertices(object):
    @staticmethod
    def d2(actor, expand=carla.Vector2D(0.0,0.0)):
        if not hasattr(actor, 'bounding_box'): raise RuntimeError

        t = actor.get_transform()
        dx, dy = actor.bounding_box.extent.x +expand.x, actor.bounding_box.extent.y +expand.y
        center_x, center_y, theta = t.location.x, t.location.y, np.deg2rad(t.rotation.yaw)

        l, n = np.array([np.cos(theta), np.sin(theta)]), np.array([np.cos(theta+np.pi/2), np.sin(theta+np.pi/2)])
        vertices = np.expand_dims(np.array([center_x, center_y]), axis=0).repeat(4, axis=0)

        vertices[0] +=  l*dx + n*dy
        vertices[1] += -l*dx + n*dy
        vertices[2] += -l*dx - n*dy
        vertices[3] +=  l*dx - n*dy
        return vertices
    
    @staticmethod
    def d2arrow(actor, expand=carla.Vector2D(0.0,0.0)):
        if not hasattr(actor, 'bounding_box'): raise RuntimeError

        t = actor.get_transform()
        dx, dy = actor.bounding_box.extent.x +expand.x, actor.bounding_box.extent.y +expand.y
        center_x, center_y, theta = t.location.x, t.location.y, np.deg2rad(t.rotation.yaw)

        l, n = np.array([np.cos(theta), np.sin(theta)]), np.array([np.cos(theta+np.pi/2), np.sin(theta+np.pi/2)])
        vertices = np.expand_dims(np.array([center_x, center_y]), axis=0).repeat(7, axis=0)

        vertices[0] +=  l*dx + n*dy
        vertices[1] += -l*dx + n*dy
        vertices[2] += -l*dx - n*dy
        vertices[3] +=  l*dx - n*dy

        vertices[4] += l*dx
        vertices[5] += 0.3*l*dx + n*dy
        vertices[6] += 0.3*l*dx - n*dy

        lines = np.array([[0,1], [1,2], [2,3], [3,0],  [4,5], [5,6], [6,4]])
        return vertices, lines



class CollisionCheck(object):
    """
    Note: currently, for synchronous mode. TODO: for asynchronous mode.
    """
    
    @staticmethod
    def d2(actor1, actor2, expand=carla.Vector2D(0.0,0.0)):
        """
        
        
        Args:
            expand: for asynchronous mode, use carla.Vector2D(0.1, 0.05)
        
        Returns:
            bool
        """
        
        p1 = Polygon(ActorVertices.d2(actor1, expand))
        p2 = Polygon(ActorVertices.d2(actor2, expand))
        result = p1.intersects(p2)
        return result
