from carla_utils import carla, navigation


import numpy as np

from .waypoint import Waypoint
from .lane_path import LanePath
RoadOption = navigation.local_planner.RoadOption



def getRoadPath(reference_route):
    unique_index = list(range(len(reference_route)))

    max_step = len(unique_index)
    road_path = RoadPath(max_step)
    for step, index in enumerate(unique_index):
        carla_waypoint = reference_route[index][0]
        if step == 0:
            center_lane_id = 0
        else:
            center_lane_id = 0
            if reference_route[index][1] == RoadOption.CHANGELANELEFT:
                center_lane_id += 1
            elif reference_route[index][1] == RoadOption.CHANGELANERIGHT:
                center_lane_id -= 1
            else:
                pass
                # raise NotImplementedError
        waypoints = _get_adjacent_waypoints(step, center_lane_id, carla_waypoint)
        # waypoints = _get_adjacent_waypoints_no_rule(step, center_lane_id, carla_waypoint)
        for waypoint in waypoints:
            lane_id = waypoint.lane_id
            if lane_id in road_path.lane_dict.keys():
                lane_path = road_path.lane_dict[lane_id]
                lane_path.append(step, waypoint)
            else:
                lane_path = LanePath(lane_id)
                lane_path.append(step, waypoint)
                road_path.append(lane_path)

    return road_path


def _get_adjacent_waypoints(step, lane_id, carla_waypoint):
    waypoints = [Waypoint(carla_waypoint, lane_id=lane_id, step=step, reference=True)]
    broken = carla.LaneMarkingType.Broken

    left_lane_id, left_wp = lane_id+1, carla_waypoint.get_left_lane()
    while left_wp is not None and left_wp.right_lane_marking.type == broken:
        waypoints.append( Waypoint(left_wp, lane_id=left_lane_id, step=step, reference=False) )
        left_lane_id += 1
        left_wp = left_wp.get_left_lane()

    right_lane_id, right_wp = lane_id-1, carla_waypoint.get_right_lane()
    while right_wp is not None and right_wp.left_lane_marking.type == broken:
        waypoints.append( Waypoint(right_wp, lane_id=right_lane_id, step=step, reference=False) )
        right_lane_id -= 1
        right_wp = right_wp.get_right_lane()
    return waypoints

def _get_adjacent_waypoints_no_rule(step, lane_id, carla_waypoint):
    waypoints = [Waypoint(carla_waypoint, lane_id=lane_id, step=step, reference=True)]
    carla_waypoints = [carla_waypoint]

    left_lane_id, left_wp = lane_id+1, carla_waypoint.get_left_lane()
    while left_wp is not None:
        print(left_wp.transform.location)
        if min([left_wp.transform.location.distance(wp.transform.location) for wp in carla_waypoints]) < 0.02:
            break
        waypoints.append( Waypoint(left_wp, lane_id=left_lane_id, step=step, reference=False) )
        carla_waypoints.append(left_wp)
        left_lane_id += 1
        left_wp = left_wp.get_left_lane()

    right_lane_id, right_wp = lane_id-1, carla_waypoint.get_right_lane()
    while right_wp is not None:
        if min([right_wp.transform.location.distance(wp.transform.location) for wp in carla_waypoints]) < 0.02:
            break
        waypoints.append( Waypoint(right_wp, lane_id=right_lane_id, step=step, reference=False) )
        carla_waypoints.append(right_wp)
        right_lane_id -= 1
        right_wp = right_wp.get_right_lane()
    print()
    return waypoints






class RoadPath(object):
    def __init__(self, max_step, **kwargs):
        # self.obey_rule = obey_rule
        self.max_step = max_step
        self.number = 0
        self.lane_dict = dict()

    def append(self, lane_path):
        self.lane_dict[lane_path.lane_id] = lane_path
        self.number += 1

    def sortByLaneId(self):
        sorted_list = sorted(self.lane_dict.items(), key=lambda d:d[0])
        return list(np.array(sorted_list)[:,0]), list(np.array(sorted_list)[:,1])


    def draw(self, world, life_time=0.4):
        lane_ids, lane_paths = self.sortByLaneId()
        for lane_path in lane_paths:
            for waypoint in lane_path.waypoints:
                location = carla.Location(x=waypoint.x, y=waypoint.y, z=waypoint.z)
                world.debug.draw_point(location, size=0.2, color=carla.Color(0,0,255,255), life_time=life_time)
                world.debug.draw_string(location, str((waypoint.lane_id, waypoint.step)), color=carla.Color(0,0,0,255), life_time=life_time)
