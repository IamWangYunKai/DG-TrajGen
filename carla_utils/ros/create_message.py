from carla_utils import carla
import rospy

import numpy as np
from typing import List

from std_msgs.msg import Header, ColorRGBA
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from tf2_msgs.msg import TFMessage

from ..world_map import Role, vehicle_frame_id
from . import convert


def GlobalRoute(frame_id, timestamp, waypoints):
    color = ColorRGBA(0.5, 0.5, 1.0, 1.0)
    scale = 0.2

    route = Marker()
    route.header = convert.header(frame_id, timestamp)
    route.ns = 'waypoints'
    route.id = 0
    route.type = Marker.SPHERE_LIST
    route.action = Marker.ADD
    route.lifetime = rospy.Duration(0.0)
    route.frame_locked = False
    route.pose.orientation.w = 1.0
    route.scale.x = 1.0 *scale
    route.scale.y = 1.0 *scale
    route.scale.z = 1.0 *scale
    route.color = color

    route.points = [convert.GeoPoint.carla_waypoint(wp) for wp in waypoints]
    route.colors = [color] * len(route.points)
    return route


def Junctions(frame_id, timestamp, waypoint_pairs):
    color = ColorRGBA(1.0, 1.0, 0.3, 1.0)

    waypoint_ids = []
    waypoints = []
    for wp_pair in waypoint_pairs:
        for wp in wp_pair:
            if wp.id not in waypoint_ids:
                waypoint_ids.append(wp.id)
                waypoints.append(wp)

    junctions = Marker()
    junctions.header = convert.header(frame_id, timestamp)
    junctions.ns = 'junctions'
    junctions.id = 0
    junctions.type = Marker.SPHERE_LIST
    junctions.action = Marker.ADD
    junctions.lifetime = rospy.Duration(0.0)
    junctions.frame_locked = False
    junctions.pose.orientation.w = 1.0
    junctions.scale.x = 1.0
    junctions.scale.y = 1.0
    junctions.scale.z = 1.0
    junctions.color = color

    junctions.points = [convert.GeoPoint.carla_waypoint(wp) for wp in waypoints]
    junctions.colors = [color] * len(junctions.points)
    return junctions


def Map(frame_id, timestamp, town_map):
    map = MarkerArray()
    map.markers.append( GlobalRoute(frame_id, timestamp, town_map.generate_waypoints(0.1)) )
    map.markers.append( Junctions(frame_id, timestamp, town_map.get_topology()) )
    return map



def VehicleTransform(frame_id, timestamp, vehicle: carla.Vehicle):
    child_frame_id = vehicle_frame_id(vehicle)
    header = convert.header(frame_id, timestamp)
    return convert.GeoTransformStamped.carla_transform(header, child_frame_id, vehicle.get_transform())

def VehiclesTransform(frame_id, timestamp, vehicles: List[carla.Vehicle]):
    tfmsg = TFMessage()
    tfmsg.transforms = [VehicleTransform(frame_id, timestamp, v) for v in vehicles]
    return tfmsg


def StaticVehiclesTransform(frame_id, timestamp, static_vehicles):
    """
        Note: if carla runs in synchronous mode, then needs world.tick() after this method.
    
    Args:
        static_vehicles: List[carla.EnvironmentObject]

    Returns:
        
    """

    tfmsg = TFMessage()
    header = convert.header(frame_id, timestamp)
    for v in static_vehicles:
        child_frame_id = v.name + '_' + str(v.id)
        tfmsg.transforms.append(convert.GeoTransformStamped.carla_transform(header, child_frame_id, v.transform))
    return tfmsg


def BoundingBox(frame_id, timestamp, vehicle, vi):
    bbx = vehicle.bounding_box.extent
    color = vehicle.attributes.get('color', '190,190,190')
    color = np.array(eval(color)).astype(np.float64) / 255

    marker = Marker()
    marker.header = convert.header(frame_id, timestamp)
    marker.header.frame_id = vehicle_frame_id(vehicle)
    marker.id = vi
    marker.type = Marker.CUBE
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0
    marker.scale.x = bbx.x * 2
    marker.scale.y = bbx.y * 2
    marker.scale.z = bbx.z * 2
    marker.color = ColorRGBA(*color, 1.0)
    return marker
    
def BoundingBoxes(frame_id, timestamp, vehicles):
    marker_array = MarkerArray()
    marker_array.markers = [BoundingBox(frame_id, timestamp, v, vi) for vi, v in enumerate(vehicles)]
    return marker_array


def StaticBoundingBox(frame_id, timestamp, static_vehicle, vi):
    bbx = static_vehicle.bounding_box.extent
    color = np.array(eval('100,100,100')).astype(np.float64) / 255

    marker = Marker()
    marker.header = convert.header(frame_id, timestamp)
    marker.header.frame_id = static_vehicle.name + '_' + str(static_vehicle.id)
    marker.id = vi
    marker.type = Marker.CUBE
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0
    marker.scale.x = bbx.x * 2
    marker.scale.y = bbx.y * 2
    marker.scale.z = bbx.z * 2
    marker.color = ColorRGBA(*color, 1.0)
    return marker

def StaticBoundingBoxes(frame_id, timestamp, static_vehicles):
    marker_array = MarkerArray()
    start = 10000
    marker_array.markers = [StaticBoundingBox(frame_id, timestamp, v, vi+start) for vi, v in enumerate(static_vehicles)]
    return marker_array




# =============================================================================
# -- augment  -----------------------------------------------------------------
# =============================================================================


from ..trajectory import BaseCurve, BaseCurves


def Curve(header: Header, base_curve: BaseCurve, color=(255,255,255)):
    visual_curve = Marker()
    visual_curve.header = header

    color = np.array(color).astype(np.float) /255
    visual_curve.color = ColorRGBA(*color, 1.0)
    visual_curve.ns = 'default'
    visual_curve.type = Marker.LINE_STRIP
    visual_curve.action = Marker.ADD
    visual_curve.lifetime = rospy.Duration()
    visual_curve.id = base_curve.id
    visual_curve.pose.orientation.w = 1.0
    visual_curve.scale.x = 0.2

    for (x, y) in zip(base_curve.x, base_curve.y):
        visual_curve.points.append( Point(x=x, y=y) )
    return visual_curve


def Curves(header: Header, base_curves: BaseCurves, colors=None):
    if colors == None:
        colors = [(255,255,255)] * len(base_curves)
    visual_curves = MarkerArray()
    for i in range(len(base_curves)):
        base_curve = base_curves.get(i)
        curve = Curve(header, base_curve, colors[i])
        visual_curves.markers.append(curve)
    return visual_curves
