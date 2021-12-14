
from carla_utils import carla

import numpy as np
from typing import List

from ..basic import Data
from ..augment import cvt
from ..agents import BaseAgent, BaseAgentObstacle


class GroundTruthVehicle(object):
    def __init__(self, config, max_vehicles, dim_state):
        """
        """

        self.core = config.core
        self.perception_range = config.perception_range
        self.max_velocity = config.max_velocity
        self.max_vehicles = max_vehicles
        self.dim_state = dim_state


    def run_step(self, agent: BaseAgent, obstacles: List[BaseAgentObstacle]):
        ego_state = agent.get_state()

        dist_haha = lambda l: np.hypot(l.x - ego_state.x, l.y - ego_state.y)
        obstacles = [(dist_haha(o.get_state()), o) for o in obstacles]
        obstacles = sorted(obstacles, key=lambda d:d[0])

        state_fixed = self._get_state_ego(ego_state)
        state_dynamic = np.ones((self.max_vehicles, self.dim_state), dtype=np.float32) * np.inf
        state_mask = np.zeros(self.max_vehicles, dtype=np.int64)
        for index, (d, o) in enumerate(obstacles[:self.max_vehicles]):
            if d > self.perception_range:
                break
            state_o = o.get_state().world2local(ego_state)

            state_dynamic[index] = self._get_state_obs(state_o)
            state_mask[index] = 1
        
        return Data(ego=state_fixed, obs=state_dynamic, mask=state_mask)


    def _get_state_ego(self, vehicle_state):
        state = np.array([
                vehicle_state.v /self.max_velocity /1.2,
                # vehicle_state.theta / np.pi,
            ], dtype=np.float32)
        return state

    def _get_state_obs(self, obs_state):
        state = np.array([
                # np.hypot(obs_state.x, obs_state.y) / self.perception_range,
                # np.arctan2(obs_state.y, obs_state.x) / np.pi,
                obs_state.x / self.perception_range, obs_state.y / self.perception_range, 
                obs_state.v /self.max_velocity /1.2,
                obs_state.theta / np.pi,
            ], dtype=np.float32)
        return state




class GroundTruthRoute(object):
    def __init__(self, config, dim_state, perception_range):
        self.core = config.core
        self.perception_range = perception_range
        self.dim_state = dim_state
        assert dim_state % 2 == 0


    def run_step(self, agent: BaseAgent):
        state0 = agent.get_state()
        current_transform = agent.get_transform()

        waypoints, _ = agent.global_path.remaining_waypoints(current_transform)

        states, xs, ys = [], [], []
        for wp in waypoints:
            if wp.transform.location.distance(current_transform.location) > self.perception_range:
                break
            state_local = cvt.CUAState.carla_transform(wp.transform).world2local(state0)
            states.append(state_local)
            xs.append(state_local.x)
            ys.append(state_local.y)

        xs = np.asarray(xs)
        ys = np.asarray(ys)

        t_origin = np.linspace(0,1, len(states))
        t_resample = np.linspace(0,1, int(self.dim_state /2))
        x_resample = np.interp(t_resample, t_origin, xs)
        y_resample = np.interp(t_resample, t_origin, ys)

        route = np.hstack([x_resample, y_resample]).astype(np.float32) / self.perception_range
        return route

        
    def viz(self, route):
        route = route.reshape(2,-1)
        x, y = route[0], route[1]

        import matplotlib.pyplot as plt
        plt.cla()
        plt.plot(x, y, 'og')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.pause(0.1)
        # plt.show()
