from carla_utils import carla

import numpy as np
import time
import random

from ..basic import YamlConfig
from ..augment import GlobalPath
from ..sensor import createSensorListMaster, CarlaSensorListMaster
from ..world_map import Core
from ..world_map import get_spawn_transform, Role, add_vehicle, add_vehicles, default_settings
from ..world_map import get_reference_route_wrt_waypoint
from ..world_map import ScenarioRole
from ..agents import AgentListMaster, BaseAgentObstacle, BaseAgent, NaiveAgent



class ScenarioSingleAgent(object):
    time_tolerance = None

    map_name = None
    max_velocity = None
    num_vehicles = None
    obstacle_color = None        ### None means random, otherwise, e.g., (255,255,255)
    type_id = None
    obstacle_type_id = None
    agent_role = ScenarioRole.learnable
    obstacle_role = ScenarioRole.obstacle

    def __init__(self, config):
        # config.set('max_velocity', self.max_velocity)
        # config.set('num_vehicles', self.num_vehicles)
        self.set_parameters(config)
        self.config = config
        self.core: Core = config.get('core', None)
        self.town_map = self.core.town_map
        self.spawn_points = np.array(self._generate_spawn_points())
        return


    @classmethod
    def set_parameters(cls, config: YamlConfig):
        config.set('time_tolerance', cls.time_tolerance)
        config.set('max_velocity', cls.max_velocity)
        config.set('num_vehicles', cls.num_vehicles)
        return


    def _generate_spawn_points(self):
        return [t.location for t in self.town_map.get_spawn_points()]


    def register_agents(self, agents_master: AgentListMaster, sensors_params):
        spawn_points = self.get_spawn_points(self.num_vehicles)
        type_ids = self.type_ids
        colors = self.colors
        role_names = self.role_names

        '''add ego vehicle'''
        ego_vehicle = add_vehicle(self.core, True, spawn_points[0], type_ids[0], role_name=role_names[0], color=colors[0])

        ### route_planner has a bug: global_path.origin != vehicle.get_transform()
        start = get_spawn_transform(self.core, ego_vehicle.get_location(), height=0.1)
        global_path: GlobalPath = self.generate_global_path(start)
        ego_vehicle.set_transform(global_path.origin)
        self.core.tick()

        sensors_master = createSensorListMaster(self.core, ego_vehicle, sensors_params)

        '''register agent'''
        Agent = agents_master.get_agent_type()
        agent: BaseAgent = Agent(self.config, ego_vehicle, sensors_master, global_path)
        agent.extend_route()
        agents_master.register(agent, learnable=True)
        
        '''add vehicles'''
        vehicles = add_vehicles(self.core, True, spawn_points[1:], type_ids[1:], role_names=role_names[1:], colors=colors[1:])
        for v in vehicles:
            sm = CarlaSensorListMaster(self.core, v)
            agents_master.register_obs( BaseAgentObstacle(self.config, v, sm) )
        self.core.tick()
        return



    def get_spawn_points(self, num_vehicles, *args):
        if num_vehicles > len(self.spawn_points):
            msg = 'requested {} vehicles, but could only find {} spawn points'.format(num_vehicles, len(self.spawn_points))
            # raise RuntimeError(msg)   ### deprecated
            print('warning: [Scenario] {}'.format(msg))
            # num_vehicles = len(self.spawn_points)
        spawn_points = np.random.choice(self.spawn_points, size=num_vehicles, replace=False)
        return spawn_points


    @property
    def type_ids(self):
        _type_ids = [self.type_id] + [self.obstacle_type_id] * (self.num_vehicles-1)
        return _type_ids
    @property
    def colors(self):
        _colors = [(255,0,0)] + [self.obstacle_color] * (self.num_vehicles-1)
        return _colors
    @property
    def role_names(self):
        _role_names = [Role(vi=0, atype=self.agent_role,)]
        _role_names += [Role(vi=vi, atype=self.obstacle_role,) for vi in range(1, self.num_vehicles)]
        return _role_names


    def generate_global_path(self, spawn_transform: carla.Transform, *args):
        waypoint = self.town_map.get_waypoint(spawn_transform.location)
        route = get_reference_route_wrt_waypoint(waypoint, sampling_resolution=0.2, sampling_number=1000)
        return GlobalPath(route)






class ScenarioMultiAgent(ScenarioSingleAgent):
    num_agents = None

    @classmethod
    def set_parameters(cls, config):
        super().set_parameters(config)
        config.set('num_agents', cls.num_agents)
        return

