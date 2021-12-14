from carla_utils import carla, navigation

import numpy as np
import copy
import random

RoadOption = navigation.local_planner.RoadOption
from .tools import error_transform, distance2d, distance_waypoint
from .. import basic

lane_change_set = set([RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT])
lane_keeping_set = set([RoadOption.VOID, RoadOption.LEFT, RoadOption.RIGHT, RoadOption.STRAIGHT, RoadOption.LANEFOLLOW])


def calc_curvature_with_yaw_diff(x, y, yaw):
    a, b = np.diff(x), np.diff(y)
    dists = np.hypot(a, b)
    d_yaw = basic.pi2pi(np.diff(yaw))

    curvatures = d_yaw / dists
    curvatures = np.concatenate([curvatures, [0.0]])
    return curvatures, dists


class GlobalPath(object):
    def __init__(self, route, **kwargs):
        ### suggest that sampling_resolution <= 0.2

        # assert len(route) > 1

        self.route = copy.copy(route)
        self._destination = self.route[-1][0].transform

        self.carla_waypoints = copy.copy(kwargs.get('carla_waypoints', None))
        self.options = copy.copy(kwargs.get('options', None))
        self.x = copy.copy(kwargs.get('x', None))
        self.y = copy.copy(kwargs.get('y', None))
        self.z = copy.copy(kwargs.get('z', None))
        self.theta = copy.copy(kwargs.get('theta', None))
        self.curvatures = copy.copy(kwargs.get('curvatures', None))
        self.distances = copy.copy(kwargs.get('distances', None))
        self.sampling_resolution = copy.copy(kwargs.get('sampling_resolution', None))

        if None in [self.x, self.y, self.theta]:
            self.carla_waypoints, self.options = [], []
            self.x, self.y, self.z, self.theta = [], [], [], []
            for waypoint, option in route:
                self.carla_waypoints.append(waypoint)
                self.options.append(option)
                self.x.append(waypoint.transform.location.x)
                self.y.append(waypoint.transform.location.y)
                self.z.append(waypoint.transform.location.z)
                self.theta.append(np.deg2rad(waypoint.transform.rotation.yaw))

            self.curvatures, self.distances = calc_curvature_with_yaw_diff(self.x, self.y, self.theta)
            self.sampling_resolution = np.average(self.distances) if len(self) > 1 else 0.1

        self._max_coverage = 0


    def __len__(self):
        return len(self.route)
    
    def save_to_disk(self, file_path):
        option = [i.value for i in self.options]
        arr = np.stack([self.x, self.y, self.z, option]).T
        np.savetxt(file_path, arr, fmt='%.10f')
        return
    
    @staticmethod
    def read_from_disk(town_map, file_path):
        arr = np.loadtxt(file_path)
        xs = arr[:,0]
        ys = arr[:,1]
        zs = arr[:,2]
        options = arr[:,3]
        route = []
        for (x, y, z, option) in zip(xs, ys, zs, options):
            location = carla.Location(x, y, z)
            waypoint = town_map.get_waypoint(location)
            route.append((waypoint, RoadOption(int(option))))
        
        simplified_route = copy.copy(route)
        delete_index_list = []
        for i in range(len(route)-1):
            loc_old, loc_new = route[i][0].transform.location, route[i+1][0].transform.location
            if loc_old.distance(loc_new) < 0.025:
                delete_index_list.append(i+1)
        basic.list_del(simplified_route, delete_index_list)

        return GlobalPath(simplified_route)
    
    def copy(self):
        carla_waypoints = self.carla_waypoints
        options = self.options
        x, y, z = self.x, self.y, self.z
        theta = self.theta
        curvatures, distances = self.curvatures, self.distances
        sampling_resolution = self.sampling_resolution

        gp = GlobalPath(self.route,
                carla_waypoints=carla_waypoints, options=options,
                x=x, y=y, z=z, theta=theta,
                curvatures=curvatures, distances=distances,
                sampling_resolution=sampling_resolution,
            )
        return gp
    

    def extend(self, route):
        self.route.extend(copy.copy(route))
        self._destination = self.route[-1][0].transform

        for waypoint, option in route:
            self.carla_waypoints.append(waypoint)
            self.options.append(option)
            self.x.append(waypoint.transform.location.x)
            self.y.append(waypoint.transform.location.y)
            self.z.append(waypoint.transform.location.z)
            self.theta.append(np.deg2rad(waypoint.transform.rotation.yaw))

        # print('----- ', len(self.x), len(self.y), len(self.theta), len(self.curvatures), len(self.distances))
        self.curvatures, self.distances = calc_curvature_with_yaw_diff(self.x, self.y, self.theta)

        self.sampling_resolution = np.average(self.distances)
        return


    @property
    def origin(self):
        return self.carla_waypoints[0].transform
    @property
    def destination(self):
        return self._destination
    @property
    def max_coverage(self):
        return self._max_coverage
    
    def reached(self, preview_distance=0):
        preview_distance = max(0, preview_distance)
        preview_index = max(preview_distance // (self.sampling_resolution+1e-6) + 1, 0)
        return self._max_coverage >= len(self)-1 - preview_index

    
    def draw(self, world, size=0.1, color=(0,255,0), life_time=10):
        for waypoint in self.carla_waypoints:
            world.debug.draw_point(waypoint.transform.location, size=size, color=carla.Color(*color), life_time=life_time)
        return


    def next_waypoint(self, current_transform, distance):
        '''
            (warning: inaccurate) test on more scenes besides single lane, error is less than 10cm when sampling_resolution is 20cm
            Args:
                current_transform: carla.Transform
        '''
        self._step_coverage(current_transform)

        longitudinal_e, _, _ = error_transform(current_transform, self.carla_waypoints[self._max_coverage].transform)
        # assert longitudinal_e >= 0 or self._max_coverage == 0
        distance += longitudinal_e

        length = 0.0
        index = self._max_coverage
        for index in range(self._max_coverage, len(self)-1):
            length += distance_waypoint(self.carla_waypoints[index], self.carla_waypoints[index+1])
            if length >= distance:
                break
        
        if index == len(self)-1: return random.choice(self.carla_waypoints[index].next(distance))

        waypoint_i, waypoint_ip = self.carla_waypoints[index], self.carla_waypoints[index+1]

        distance_previous = max(length-distance, 0.00001)
        distance_next = max(distance_waypoint(waypoint_i, waypoint_ip) -(length-distance), 0.00001)
        waypoints_ip = waypoint_ip.previous(distance_previous)
        waypoints_i = waypoint_i.next(distance_next)

        result, min_dist = None, float('inf')
        for next_waypoint in waypoints_i:
            for waypoint in waypoints_ip:
                dist = next_waypoint.transform.location.distance(waypoint.transform.location)
                if dist < min_dist:
                    min_dist = dist
                    result = next_waypoint
        return result
    
    def target_waypoint(self, current_transform):
        self._step_coverage(current_transform)
        index = min(len(self)-1, self._max_coverage+1)
        return self.carla_waypoints[index], self.curvatures[index]
    
    def remaining_waypoints(self, current_transform):
        self._step_coverage(current_transform)
        return self.carla_waypoints[self._max_coverage:], sum(self.distances[self._max_coverage:])
    

    def _step_coverage(self, current_transform):
        '''
            Args:
                current_transform: carla.Transform
        '''
        index = self._max_coverage
        for index in range(self._max_coverage, len(self)):
            longitudinal_e, _, _ = error_transform(current_transform, self.carla_waypoints[min(len(self)-1, index+1)].transform)
            if longitudinal_e < 0:
                break
        self._max_coverage = index


    def error(self, current_transform):
        self._step_coverage(current_transform)
        longitudinal_e, lateral_e, theta_e = error_transform(current_transform, self.carla_waypoints[self._max_coverage].transform)
        return longitudinal_e, lateral_e, theta_e


    def nearest_waypoint(self, transform):
        wps = self.carla_waypoints[self._max_coverage:]
        dists = np.asarray([distance2d(wp.transform.location, transform.location) for wp in wps])
        return wps[np.argmin(dists)]
