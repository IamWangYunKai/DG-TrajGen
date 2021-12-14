import rllib

from abc import ABC
import numpy as np
import time

import torch
import gym
from gym.spaces import Box

from .. import basic


class GymEnvSingleAgent(gym.Env, ABC):
    '''
        Only for ray.rllib.
    '''

    from .env_sa import EnvSingleAgent
    env_cls = EnvSingleAgent

    def __init__(self, config_dict):

        self.action_space = self.get_action_space()
        self.observation_space = self.get_observation_space()

        mode = 'train'
        env_index = config_dict.worker_index
        server = config_dict['servers'][env_index-1]
        seed = env_index + 1000
        config = basic.YamlConfig(
            description='pseudo'+str(config_dict.worker_index),
            host=server.host,
            port=server.port,
            timeout=2.0,
            real=False,
            seed=seed,
        )

        basic.create_dir(config, 'Pseudo')
        rllib.basic.setup_seed(seed)

        self.env = self.env_cls(config, mode, env_index)

        self.reset()


    def reset(self):
        t1 = time.time()
        self.env.reset()
        t2 = time.time()
        print('env {}:{} reset time: '.format(self.env.core.host, str(self.env.core.port)), t2-t1, 's')

        state = self.env.agents_master.perception()
        return state.squeeze(0).numpy()


    def step(self, action):
        select_action = lambda _: torch.from_numpy(action).unsqueeze(0)
        experience, epoch_done = self.env.step(select_action)
        return experience.next_state.squeeze(0).numpy(), experience.reward.item(), bool(epoch_done), {}


    def close(self):
        self.env.destroy()


    def get_action_space(self):
        return Box(low=-1, high=1, shape=(self.env_cls.dim_action,))

    def get_observation_space(self):
        return Box(low=-1, high=1, shape=(self.env_cls.dim_state,))

