import rllib
from carla_utils import carla

import time
from os.path import join
from abc import ABC, abstractmethod
import threading
import numpy as np

import torch

from ..system import mkdir, Clock
from ..world_map import Core
from ..world_map import default_settings
from ..agents import AgentListMaster

from ..basic import Data
from ..basic import Data as Experience
from ..basic import YamlConfig

from .scenario import ScenarioSingleAgent
from .recorder import Recorder


class EnvSingleAgent(ABC):
    """
    Suppose:
        1. Obstacles have no sensors.
        2. One env has its own config.

    """

    modes = ['train', 'evaluate', 'real']

    Scenario = ScenarioSingleAgent
    AgentListMaster = AgentListMaster
    sensors_params = []

    dim_state = None
    dim_action = None

    decision_frequency = None
    control_frequency = None

    perception_range = None


    def __init__(self, config: YamlConfig, writer: rllib.basic.Writer, mode, env_index=-100):
        ### set parameters
        self.set_parameters(config, mode, env_index)

        self.config, self.path_pack = config, config.path_pack
        self.writer = writer
        self.output_dir = join(self.path_pack.output_path, 'env')
        mkdir(self.output_dir)
        assert mode in self.modes
        self.mode = mode
        self.env_index = env_index

        self.clock = Clock(self.control_frequency)
        self.clock_decision = Clock(self.decision_frequency)
        self.step_reset = 0

        ### agent
        self.agents_master = None

        ### recorder
        self.recorder = Recorder(self.output_dir)
        
        ### client
        self.core = Core(config, self.Scenario.map_name, settings=self.settings)
        self.client, self.world, self.town_map = self.core.client, self.core.world, self.core.town_map
        self.traffic_manager = self.core.traffic_manager
        self.scenario: ScenarioSingleAgent = self.Scenario(config)

        self.init()
        return
    


    def reset(self):
        ### save record and destroy agents
        self._destroy_agents()

        ### env param
        self.step_reset += 1
        self.time_step = 0

        ### scenario param
        self.core.load_map(self.scenario.map_name)
        self.time_tolerance = self.scenario.time_tolerance
        self.map_name = self.scenario.map_name
        self.max_velocity = self.scenario.max_velocity
        self.num_vehicles = self.scenario.num_vehicles

        ### record scenario
        self.recorder.record_town_map(self.scenario)

        ### register agents
        self.agents_master: AgentListMaster = self.AgentListMaster(self.config)
        self.scenario.register_agents(self.agents_master, self.sensors_params)

        ### recorder
        self.recorder.record_scenario(self.config, self.scenario)
        # self.client.start_recorder(join(self.output_dir, 'recording_{}.log'.format(self.step_reset)), True)
        self.client.start_recorder(join(self.output_dir, 'recording_{}.log'.format(self.step_reset)))

        ### callback
        self.on_episode_start()
        return



    @torch.no_grad()
    def _step_train(self, method: rllib.template.MethodSingleAgent):
        timestamp = str(time.time())
        self.time_step += 1

        ### state
        state = self.agents_master.perception(self.step_reset, self.time_step)
        ### action
        action = method.select_action(state).cpu()
        ### reward
        reward, epoch_info = self._calculate_reward(state, action)
        epoch_done = epoch_info.done
        
        ### record
        self.recorder.record_agents(self.time_step, self.agents_master, epoch_info)
        self.recorder.record_experience(self.time_step, self.agents_master, action)
        
        ### callback
        self.on_episode_step(reward, epoch_info)

        ### step
        self.agents_master.run_step( action )

        ### next_state
        next_state = self.agents_master.perception(self.step_reset, timestamp=-1)

        ### experience
        reward = torch.tensor([reward], dtype=torch.float32)
        done = torch.tensor([epoch_done], dtype=torch.float32)
        experience = Experience(
            state=state, action=action, next_state=next_state, reward=reward,
            done=done, timestamp=timestamp,
        )
        return experience, epoch_done, epoch_info
    

    @torch.no_grad()
    def _step_evaluate(self, *args):
        return self._step_train(*args)
        # return EnvSingleAgent._step_train(self, *args)


    @torch.no_grad()
    def _step_real(self, method: rllib.template.MethodSingleAgent):
        self.clock_decision.tick_begin()
        timestamp = str(time.time())
        self.time_step += 1

        ### state
        state = self.agents_master.perception(self.step_reset, self.time_step)
        ### reward
        action = method.select_action(state).cpu()
        reward, epoch_info = self._calculate_reward(state, action)
        epoch_done = epoch_info.done

        ### step
        self.agents_master.run_step( action )
        
        ### next_state
        next_state = self.agents_master.perception(self.time_step, self.agents_master)

        ### experience
        reward = torch.tensor([reward], dtype=torch.float32)
        done = torch.tensor([epoch_done], dtype=torch.float32)
        experience = Experience(
            state=state, action=action, next_state=next_state, reward=reward,
            done=done, timestamp=timestamp,
        )
        self.clock_decision.tick_end()
        return experience, epoch_done, epoch_info



    @classmethod
    def set_parameters(cls, config: YamlConfig, mode, env_index=-100):
        print('[env] cls: ', cls)
        config.set('mode', mode)
        config.set('env_index', env_index)
        config.set('dim_state', cls.dim_state)
        config.set('dim_action', cls.dim_action)
        config.set('decision_frequency', cls.decision_frequency)
        config.set('control_frequency', cls.control_frequency)
        config.set('perception_range', cls.perception_range)

        if mode == 'real':
            real = True
        else:
            real = False
        config.set('real', real)
        return


    @property
    def settings(self):
        if self.mode == 'real':
            st = default_settings(sync=False, render=True, dt=0.0)
        else:
            st = default_settings(sync=True, render=False, dt=1/self.control_frequency)
        return st


    def init(self):
        if self.mode == 'real':
            self.step = self._step_real
            print('[{}] Env is in real mode.'.format(self.__class__.__name__))

            self.target = None
            self.thread_stoped = False
            self.thread_control = threading.Thread(target=self._run_control, args=(lambda: self.thread_stoped,))
            self.thread_control.start()

        elif self.mode == 'train':
            self.step = self._step_train
        elif self.mode == 'evaluate':
            self.step = self._step_evaluate
        else:
            raise NotImplementedError('Unknown Mode.')
        return


    def _destroy_agents(self):
        if self.agents_master != None:
            self.recorder.save_to_disk(self.step_reset)
            self.recorder.clear()
            self.client.stop_recorder()
            self.agents_master.destroy()
        self.agents_master = None
        return

    def destroy(self):
        if self.mode == 'real':
            self.thread_stoped = True
            self.thread_control.join()
        self._destroy_agents()
        self.core.kill()



    @abstractmethod
    def _calculate_reward(self): return

    def check_timeout(self):
        return self.time_step >= self.time_tolerance



    def on_episode_start(self):
        return
    def on_episode_step(self, *args, **kwargs):
        return
    def on_episode_end(self):
        return



    def _run_control(self, stop):
        while True:
            if self.agents_master == None:
                time.sleep(0.01)
                continue

            self.clock.tick_begin()

            self.agents_master.run_control()

            if stop(): break
            self.clock.tick_end()
        print('[carla_utils.rl_template.EnvSingleAgent-Real] Exit control thread.')
        return

