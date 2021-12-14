# @TODO

from carla_utils import carla

import random
import os
from os.path import join
from abc import abstractproperty

from ..system import parse_yaml_file_unsafe, Clock
from ..basic import setup_seed, YamlConfig
from ..sensor import createSensorListMaster, template
from ..world_map import Core, default_settings, connect_to_server, add_vehicle, Role
from ..agents import AgentABC

from .pygame_interaction import PyGameInteraction
from .tools import default_argparser



class EgoVehicle(object):
    def __init__(self, config: YamlConfig):
        scenario_cls = config.get('scenario_cls', None)  ## ! warning
        sensors_params = config.get('sensors_params', template.sensors_params)
        
        self.clock = Clock(config.decision_frequency)
        host, port, timeout, map_name = config.host, config.port, config.timeout, config.map_name

        # settings = default_settings(sync=True, render=True, dt=0.05)
        settings = default_settings(sync=False, render=True, dt=0.0)
        self.core = Core(config, map_name, settings, env_index=-1)
        self.world, self.town_map = self.core.world, self.core.town_map
        self.scenario = scenario_cls(self.core)

        self.global_frame_id, self.vehicle_frame_id = 'map', 'vehicle'
        role_name, type_id = config.role_name, config.type_id
        spawn_point = random.choice(self.scenario.spawn_points)
        self.vehicle = add_vehicle(self.core, True, spawn_point, type_id, role_name=Role(vi=0, name=role_name))
        self.sensor_manager = createSensorListMaster(self.core, self.vehicle, sensors_params)
        print('[ego_vehicle] ego_vehicle id: ', self.vehicle.id)
        self.agent = abstractproperty(AgentABC)

        self.pygame_interaction = PyGameInteraction(config, self.vehicle, self.sensor_manager)
        return
        

    def destroy(self):
        self.pygame_interaction.destroy()

        if self.sensor_manager is not None:
            self.sensor_manager.destroy()
            self.sensor_manager = None
        if self.vehicle is not None:
            self.vehicle.destroy()
            self.vehicle = None


    def run_step(self):
        """
            needs core.tick()
        """

        self.core.tick()
        return

    def run(self):
        while True:
            self.clock.tick_begin()
            self.run_step()
            self.pygame_interaction.tick()
            self.clock.tick_end()


if __name__ == '__main__':
    setup_seed(1998)

    try:
        config = parse_yaml_file_unsafe('./config/carla.yaml')
    except FileNotFoundError:
        print('[vehicle_visualizer] use default config.')
        file_dir = os.path.dirname(__file__)
        config = parse_yaml_file_unsafe(join(file_dir, './default_carla.yaml'))
    args = default_argparser().parse_args()
    config.update(args)

    ego_vehicle = EgoVehicle(config)
    try:
        ego_vehicle.run()
    except KeyboardInterrupt:
        print('canceled by user')
    finally:
        ego_vehicle.destroy()
        print('destroyed all relevant actors')

