from carla_utils import carla

import time
import numpy as np

from .agent_base import BaseAgent

from .tools import get_leading_agent_unsafe
from .agent_naive import AgentState


class IdmAgent(BaseAgent):
    def __init__(self, config, vehicle, sensors_master, global_path):
        BaseAgent.__init__(self, config, vehicle, sensors_master, global_path)

        self.leading_range = 50.0
        assert self.leading_range <= self.perception_range

        self.desired_speed = self.max_velocity
        self.time_gap = 1.0  ## 1.6
        self.min_gap = 2.0
        self.delta = 4.0
        self.acceleration = 1.0
        self.comfortable_deceleration = 1.5  ## 1.7
    

    def get_target(self, reference):
        """
        
        Args:
            reference: list of BaseAgent
        
        Returns:
            
        """
        
        agents = reference
        self._state = AgentState.NAVIGATING

        # actor_list = self.world.get_actors()
        # lights_list = actor_list.filter("*traffic_light*")
        # '''get traffic light'''
        # light_state, traffic_light = self._is_light_red(lights_list)
        # if light_state: self._state = AgentState.BLOCKED_RED_LIGHT

        '''get leading agent'''
        current_transform = self.vehicle.get_transform()
        reference_waypoints, remaining_distance = self.global_path.remaining_waypoints(current_transform)
        agent, distance = get_leading_agent_unsafe(self, agents, reference_waypoints, max_distance=self.leading_range)

        if agent != None:
            self._state = AgentState.BLOCKED_BY_VEHICLE

        if self._state == AgentState.NAVIGATING:
            target_v = self.max_velocity
        elif self._state == AgentState.BLOCKED_RED_LIGHT:
            target_v = -1.0
        elif self._state == AgentState.BLOCKED_BY_VEHICLE:
            target_v = self.intelligent_driver_model(agent, distance)
        else: raise NotImplementedError
        return target_v


    def intelligent_driver_model(self, leading_agent: BaseAgent, leading_distance):
        distance_c2c = leading_distance
        length_two_half = leading_agent.vehicle.bounding_box.extent.x + self.vehicle.bounding_box.extent.x
        distance_b2b = distance_c2c - length_two_half - 0.3   # bumper-to-bumper distance
        distance_b2b_valid = max(0.001, distance_b2b)

        leading_v = leading_agent.get_current_v()
        current_v = self.get_current_v()
        delta_v = current_v - leading_v
        
        s = current_v*(self.time_gap+delta_v/(2*np.sqrt(self.acceleration*self.comfortable_deceleration)))
        distance_desired = self.min_gap + max(0, s)

        v_rational = (current_v / self.desired_speed)**self.delta
        s_rational = (distance_desired / distance_b2b_valid) ** 2
        acceleration_target = self.acceleration * (1 - v_rational - s_rational)
        target_v = current_v + acceleration_target / self.decision_frequency


        # a = (distance_b2b, (self.min_gap + current_v*self.time_gap) / (1-(current_v/self.desired_speed)**self.delta)*0.5)
        # print(a, (target_v, current_v), (acceleration_target, 1-v_rational, s_rational) )
        # print()

        return target_v
    
