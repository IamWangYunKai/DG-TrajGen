
from carla_utils import carla, navigation
import copy
import time

from .. import basic
from .. system import get_carla_version
from ..augment import GlobalPath
from ..world_map import draw_waypoints


class AgentsRoutePlanner(object):
    def __init__(self, config):
        '''
        Args:
            config: need to contain:
                config.sampling_resolution
        '''
        self.core = config.get('core', None)
        self.world, self.town_map = self.core.world, self.core.town_map

        self.sampling_resolution = config.sampling_resolution

        self.grp = get_route_planner(self.town_map, self.sampling_resolution)


    def trace_route(self, origin, destination):
        '''
        Args:
            origin, destination: carla.Location
        '''
        route = self.grp.trace_route(origin, destination)

        '''remove waypoint whose distance too small'''
        simplified_route = copy.copy(route)
        delete_index_list = []
        for i in range(len(route)-1):
            loc_old, loc_new = route[i][0].transform.location, route[i+1][0].transform.location
            if loc_old.distance(loc_new) < self.sampling_resolution * 0.75:
                delete_index_list.append(i+1)
        basic.list_del(simplified_route, delete_index_list)

        return GlobalPath(simplified_route)


    def draw_global_path(self, life_time=500):
        draw_waypoints(self.world, self.global_path.carla_waypoints, life_time=life_time)



def get_route_planner(town_map, sampling_resolution):
    carla_version = get_carla_version()
    if carla_version == '0.9.12':
        GlobalRoutePlanner = navigation.global_route_planner.GlobalRoutePlanner
        grp = GlobalRoutePlanner(town_map, sampling_resolution)
    else:
        GlobalRoutePlanner = navigation.global_route_planner.GlobalRoutePlanner
        GlobalRoutePlannerDAO = navigation.global_route_planner_dao.GlobalRoutePlannerDAO
        dao = GlobalRoutePlannerDAO(town_map, sampling_resolution)
        grp = GlobalRoutePlanner(dao)
        grp.setup()
    return grp

