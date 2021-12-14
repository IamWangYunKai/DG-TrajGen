from carla_utils import carla

import time

from enum import Enum

from .agent_base import BaseAgent
from .tools import get_leading_agent_unsafe


class AgentState(Enum):
    """
    AGENT_STATE represents the possible states of a roaming agent
    """
    NAVIGATING = 1
    BLOCKED_BY_VEHICLE = 2
    BLOCKED_RED_LIGHT = 3


class NaiveAgent(BaseAgent):
    def __init__(self, config, vehicle, sensors_master, global_path):
        BaseAgent.__init__(self, config, vehicle, sensors_master, global_path)

        self.leading_range = 15.0
        assert self.leading_range < self.perception_range
    

    def get_target(self, reference):
        agents = reference
        hazard_detected = False

        # actor_list = self.world.get_actors()
        # lights_list = actor_list.filter("*traffic_light*")
        # '''get traffic light'''
        # light_state, traffic_light = self._is_light_red(lights_list)
        # if light_state: self._state = AgentState.BLOCKED_RED_LIGHT

        '''get leading agent'''
        current_transform = self.vehicle.get_transform()
        reference_waypoints, remaining_distance = self.global_path.remaining_waypoints(current_transform)

        ### disable currently
        if remaining_distance < self.leading_range:
            self.extend_route()
            reference_waypoints, remaining_distance = self.global_path.remaining_waypoints(current_transform)

        agent, _ = get_leading_agent_unsafe(self, agents, reference_waypoints, max_distance=self.leading_range)
        if agent is not None:
            self._state = AgentState.BLOCKED_BY_VEHICLE
            hazard_detected = True

        target = -1.0 if hazard_detected else self.max_velocity
        return target


