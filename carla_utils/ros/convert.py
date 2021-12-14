from carla_utils import carla
import rospy, tf

import numpy as np

from std_msgs.msg import Header
from geometry_msgs.msg import Point
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

from ..augment import State, GlobalPath
from ..trajectory import BaseCurve


def header(frame_id, timestamp):
    hd = Header()
    hd.frame_id = frame_id
    hd.stamp = rospy.Time.from_sec(timestamp)
    return hd


class GeoPoseStamped(object):
    @staticmethod
    def cua_state(header: Header, state: State):
        pose = PoseStamped()
        pose.header = header
        pose.pose.position.x = state.x
        pose.pose.position.y = state.y
        pose.pose.position.z = state.z

        quaternion = tf.transformations.quaternion_from_euler(0, 0, state.theta)
        pose.pose.orientation.x = quaternion[0]
        pose.pose.orientation.y = quaternion[1]
        pose.pose.orientation.z = quaternion[2]
        pose.pose.orientation.w = quaternion[3]
        return pose
    
    def carla_transform(header, transform: carla.Transform):
        pose = PoseStamped()
        pose.header = header
        pose.pose.position.x = transform.location.x
        pose.pose.position.y = transform.location.y
        pose.pose.position.z = transform.location.z

        roll, pitch, yaw = transform.rotation.roll, transform.rotation.pitch, transform.rotation.yaw
        quaternion = tf.transformations.quaternion_from_euler(np.deg2rad(roll), np.deg2rad(pitch), np.deg2rad(yaw))
        pose.pose.orientation.x = quaternion[0]
        pose.pose.orientation.y = quaternion[1]
        pose.pose.orientation.z = quaternion[2]
        pose.pose.orientation.w = quaternion[3]
        return pose


class GeoPoint(object):
    @staticmethod
    def carla_waypoint(waypoint: carla.Waypoint):
        point = Point()
        point.x = waypoint.transform.location.x
        point.y = waypoint.transform.location.y
        point.z = waypoint.transform.location.z
        return point



class GeoTransformStamped(object):
    @staticmethod
    def carla_transform(header, child_frame_id, transform: carla.Transform):
        geo_transform = TransformStamped()
        geo_transform.header = header
        geo_transform.child_frame_id = child_frame_id
        geo_transform.transform.translation.x = transform.location.x
        geo_transform.transform.translation.y = transform.location.y
        geo_transform.transform.translation.z = transform.location.z

        roll, pitch, yaw = transform.rotation.roll, transform.rotation.pitch, transform.rotation.yaw
        
        quaternion = tf.transformations.quaternion_from_euler(
                np.deg2rad(roll), np.deg2rad(pitch), np.deg2rad(yaw))
        geo_transform.transform.rotation.x = quaternion[0]
        geo_transform.transform.rotation.y = quaternion[1]
        geo_transform.transform.rotation.z = quaternion[2]
        geo_transform.transform.rotation.w = quaternion[3]
        return geo_transform


class NavPath(object):
    @staticmethod
    def cua_global_path(header: Header, global_path: GlobalPath):
        path = Path()
        path.header = header
        path.poses = [GeoPoseStamped.carla_transform(header, wp.transform) for wp in global_path.carla_waypoints]
        return path

    @staticmethod
    def cua_curve(header: Header, base_curve: BaseCurve, state0=None):
        curve = Path()
        curve.header = header
        curve.poses = [GeoPoseStamped.cua_state(header, base_curve.states(i, state0)) for i in range(len(base_curve))]
        return curve







