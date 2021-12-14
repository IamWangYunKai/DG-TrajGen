'''
    server_fps
'''

from carla_utils import carla

import pygame

from ..system import parse_yaml_file_unsafe, Clock
from ..world_map import connect_to_server

from .tools import default_argparser


class ServerFps(object):
    def __init__(self, config):
        client, self.world, town_map = connect_to_server(config.host, config.port, config.timeout)
        self.clock = Clock(2)

        self.world.on_tick(self.on_world_tick)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._server_clock = pygame.time.Clock()


    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds
    
    def run_step(self):
        print(self.server_fps)

    def run(self):
        while True:
            self.clock.tick_begin()
            self.run_step()
            self.clock.tick_end()


if __name__ == "__main__":
    print(__doc__)

    import os
    from os.path import join

    try:
        config = parse_yaml_file_unsafe('./config/carla.yaml')
    except FileNotFoundError:
        print('[vehicle_visualizer] use default config.')
        file_dir = os.path.dirname(__file__)
        config = parse_yaml_file_unsafe(join(file_dir, './default_carla.yaml'))
    args = default_argparser().parse_args()
    config.update(args)
    
    server_fps = ServerFps(config)
    try:
        server_fps.run()
    except KeyboardInterrupt:
        print('canceled by user')

