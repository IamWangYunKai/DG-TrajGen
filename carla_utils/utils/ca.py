'''
    clear_actors
'''

from carla_utils import carla

from ..system import parse_yaml_file_unsafe
from ..world_map import Core

from .tools import default_argparser


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

    core = Core(config, use_tm=False)
    core.tick()
    actors = core.world.get_actors()
    vehicles = actors.filter('*vehicle*')
    sensors = actors.filter('*sensor*')

    for actor in sensors: print(actor)
    for actor in vehicles: print(actor)

    import pdb; pdb.set_trace()

    for actor in sensors: actor.destroy()
    for actor in vehicles: actor.destroy()

    core.tick()
