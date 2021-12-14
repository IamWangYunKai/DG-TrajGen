from carla_utils import carla
ApplyVehicleControl = carla.command.ApplyVehicleControl
ApplyTransform = carla.command.ApplyTransform
DestroyActor = carla.command.DestroyActor

from abc import ABC, abstractmethod
from typing import List
import numpy as np

from ..system import Clock
from ..augment import CollisionCheck
from ..world_map import Core
from .agent_base import BaseAgent
from .agent_obstacle import BaseAgentObstacle


class AgentListMaster(ABC):
    def __init__(self, config):
        """
        
        
        Args:
        
        Returns:
        -------
        """
        
        self.config = config
        self.core: Core = config.get('core', None)
        self.client, self.world = self.core.client, self.core.world

        self.real = config.real
        self.expand = carla.Vector2D(0.0,0.0)

        self.decision_frequency = config.decision_frequency
        self.control_frequency = config.control_frequency
        self.skip_num = int(self.control_frequency // self.decision_frequency)
        assert self.control_frequency % self.decision_frequency == 0

        self.perception_range = float(config.perception_range)

        self.clock = Clock(self.control_frequency)

        self.agents: List[BaseAgent] = []
        self.agents_learnable: List[BaseAgent] = []
        self.obstacles: List[BaseAgentObstacle] = []
        self.run_step = self._run_step_real if config.mode == 'real' else self._run_step_fast
        self.stop_control = carla.VehicleControl(brake=0.87654)


    def register(self, agent: BaseAgent, learnable=True):
        self.agents.append(agent)
        if learnable: self.agents_learnable.append(agent)
    
    def register_obs(self, agent: BaseAgentObstacle):
        assert isinstance(agent, BaseAgentObstacle)
        self.obstacles.append(agent)

    
    def remove(self, agent):
        if agent in self.agents_learnable: self.agents_learnable.remove(agent)
        if agent in self.agents: self.agents.remove(agent)
        if agent in self.obstacles: self.obstacles.remove(agent)
    
    
    def destroy(self):
        ### v0
        # batch = []
        # for agent in self.agents + self.obstacles: batch.extend(agent.destroy_commands())
        # self.client.apply_batch_sync(batch)
        # self.core.tick()

        ### v1
        batch_vehicle = []
        batch_sensor = []
        for agent in self.agents + self.obstacles:
            batch_sensor.extend(agent.sensors_master.destroy_commands())
            batch_vehicle.append(DestroyActor(agent.vehicle))
        self.client.apply_batch_sync(batch_vehicle + batch_sensor)
        self.core.tick()

        ### ------
        self.agents, self.agents_learnable = [], []
        self.obstacles = []
        return
    

    def _run_step_fast(self, references):
        """
        Args:
            references: torch.Size([num_agents_learnable, dim_action])
        """

        references = [*references] + [self.agents] * (len(self.agents) - len(references))
        ### reference: torch.Size([dim_action])
        target_list = [agent.get_target(reference) for agent, reference in zip(self.agents, references)]
        for _ in range(self.skip_num):
            for agent, target in zip(self.agents, target_list):
                if agent.goal_reached(self.perception_range): agent.extend_route()
                control = agent.get_control(target)
                agent.forward(control)
            self.core.tick()
        return
    
    
    def _run_step_real(self, references):
        references = [*references] + [self.agents] * (len(self.agents) - len(references))
        self.targets = [agent.get_target(reference) for agent, reference in zip(self.agents, references)]
        return

    def run_control(self):
        if not hasattr(self, 'targets'):
            for agent in self.agents:
                agent.forward(self.stop_control)
            return
        for _ in range(self.skip_num):
            for agent, target in zip(self.agents, self.targets):
                if agent.goal_reached(self.perception_range): agent.extend_route()
                control = agent.get_control(target)
                agent.forward(control)
            self.core.tick()
        return



    def check_collision(self):
        num_agents = len(self.agents)
        collisions = [False] * num_agents
        for i, agent in enumerate(self.agents):
            if collisions[i] == True: continue
            for j, other_agent in enumerate(self.agents):
                if agent.id == other_agent.id: continue
                if CollisionCheck.d2(agent.vehicle, other_agent.vehicle, self.expand):
                    collisions[i] = True
                    collisions[j] = True
        return collisions

    @abstractmethod
    def perception(self): return

    @abstractmethod
    def get_agent_type(self): return
