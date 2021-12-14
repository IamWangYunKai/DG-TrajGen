from carla_utils import carla

import os
from os.path import join
import random
import signal
import subprocess
import time
import psutil, pynvml

from carla_utils.basic import YamlConfig, Data
from carla_utils.system import is_used




class Core(object):
    '''
        Inspired by https://github.com/carla-simulator/rllib-integration/blob/main/rllib_integration/carla_core.py
    '''
    def __init__(self, config: YamlConfig, map_name=None, settings=None, use_tm=True):
        self.host, self.port = config.host, config.port
        self.timeout = config.get('timeout', 2.0)
        self.seed = config.get('seed', 0)
        self.mode = config.get('mode', None)
        
        self.connect_to_server()

        self.available_map_names = self.client.get_available_maps()
        if settings != None:
            self.settings = settings
        self.load_map(map_name)

        if use_tm:
            self.add_trafficmanager()

        config.set('core', self)


    def connect_to_server(self):
        """Connect to the client"""

        num_iter = 10
        for i in range(num_iter):
            try:
                self.client = carla.Client(self.host, self.port)
                self.client.set_timeout(self.timeout)
                self.world = self.client.get_world()
                self.town_map = self.world.get_map()
                self.map_name = self.town_map.name
                self.settings = self.world.get_settings()
                print('[Core] connected to server {}:{}'.format(self.host, self.port))
                return
            except Exception as e:
                print('Waiting for server to be ready: {}, attempt {} of {}'.format(e, i + 1, num_iter))
                time.sleep(2)
        raise Exception("Cannot connect to server. Try increasing 'timeout' or 'retries_on_error' at the carla configuration")


    def load_map(self, map_name=None, weather=carla.WeatherParameters.ClearNoon):
        ### map
        map_name = str(map_name)
        flag1 = self.map_name not in map_name
        flag2 = True in [map_name in available_map_name for available_map_name in self.available_map_names]
        if flag1 and flag2:
            self.client.load_world(map_name)
            self.world = self.client.get_world()
            self.town_map = self.world.get_map()
            self.map_name = self.town_map.name
            print('[Core] load map: ', self.map_name)

        ### weather
        self.world.set_weather(weather)   ## ! TODO

        ### settings
        current_settings = self.world.get_settings()
        if  self.settings.synchronous_mode != current_settings.synchronous_mode \
                or self.settings.no_rendering_mode != current_settings.no_rendering_mode \
                or self.settings.fixed_delta_seconds != current_settings.fixed_delta_seconds:
            self.world.apply_settings(self.settings)
            print('[Core] set settings: ', self.settings)

        return


    def add_trafficmanager(self):
        tm_port = self.port + 6000
        while is_used(tm_port):
            print("Traffic manager's port " + str(tm_port) + " is already being used. Checking the next one")
            tm_port += 1000

        traffic_manager = self.client.get_trafficmanager(tm_port)
        
        if hasattr(traffic_manager, 'set_random_device_seed'):
            traffic_manager.set_random_device_seed(self.seed)
        traffic_manager.set_synchronous_mode(self.settings.synchronous_mode)
        # traffic_manager.set_hybrid_physics_mode(True)  ## do not use this

        self.traffic_manager = traffic_manager
        self.tm_port = tm_port
        return
    

    def tick(self):
        if self.settings.synchronous_mode:
            return self.world.tick()


    def kill(self):
        if hasattr(self, 'server'):
            kill_server(self.server)
        return
    


# =============================================================================
# -- server  ------------------------------------------------------------------
# =============================================================================


def launch_server(env_index, host='127.0.0.1', sleep_time=5.0, low_quality=True, no_display=True):
    port = 2000 + env_index *2

    time.sleep(random.uniform(0, 1))

    port = get_port(port)

    cmd = generate_server_cmd(port, env_index, low_quality=low_quality, no_display=no_display)
    print('running: ', cmd)
    server_process = subprocess.Popen(cmd,
        shell=True,
        preexec_fn=os.setsid,
        stdout=open(os.devnull, 'w'),
    )

    time.sleep(sleep_time)
    server = Data(host=host, port=port, process=server_process)
    return server


def launch_servers(env_indices, sleep_time=20.0):
    host = '127.0.0.1'

    servers = []
    for index in env_indices:
        server = launch_server(index, host, sleep_time=0.0)
        servers.append(server)

    time.sleep(sleep_time)
    return servers


def kill_server(server):
    server.process.send_signal(signal.SIGKILL)
    os.killpg(os.getpgid(server.process.pid), signal.SIGKILL)
    print('killed server {}:{}'.format(server.host, server.port))
    return

def kill_servers(servers):
    for server in servers:
        kill_server(server)
    return

def kill_all_servers():
    '''Kill all PIDs that start with Carla'''
    processes = [p for p in psutil.process_iter() if "carla" in p.name().lower()]
    for process in processes:
        os.kill(process.pid, signal.SIGKILL)


def generate_server_cmd(port, env_index=-1, low_quality=True, use_opengl=True, no_display=True):
    assert port % 2 == 0

    if env_index == -1:
        env_index = 0
    pynvml.nvmlInit()
    gpu_index = env_index % pynvml.nvmlDeviceGetCount()

    cmd = join(os.environ['CARLAPATH'], 'CarlaUE4.sh')

    cmd += ' -carla-rpc-port=' + str(port)
    if low_quality:
        cmd += ' -quality-level=Low'
    if use_opengl:
        cmd += ' -opengl'
    if no_display:
        # cmd = 'DISPLAY= ' + cmd   ### deprecated
        cmd = 'SDL_VIDEODRIVER=offscreen SDL_HINT_CUDA_DEVICE={} '.format(str(gpu_index)) + cmd
    return cmd



def connect_to_server(host, port, timeout=2.0, map_name=None, **kwargs):
    client = carla.Client(host, port)
    client.set_timeout(timeout)
    available_map_names = client.get_available_maps()
    world = client.get_world()
    town_map = world.get_map()

    ### map
    map_name = str(map_name)
    flag1 = town_map.name not in map_name
    flag2 = True in [map_name in available_map_name for available_map_name in available_map_names]
    if flag1 and flag2:
        client.load_world(map_name)
        world = client.get_world()
        town_map = world.get_map()

    ### weather
    weather = kwargs.get('weather', carla.WeatherParameters.ClearNoon)
    world.set_weather(weather)

    ### settings
    current_settings = world.get_settings()

    settings = kwargs.get('settings', current_settings)

    if  settings.synchronous_mode != current_settings.synchronous_mode \
            or settings.no_rendering_mode != current_settings.no_rendering_mode \
            or settings.fixed_delta_seconds != current_settings.fixed_delta_seconds:
        world.apply_settings(settings)
    settings = world.get_settings()

    print('connected to server {}:{}'.format(host, port))
    return client, world, town_map


def get_port(port):
    while is_used(port) or is_used(port+1):
        port += 1000
    return port



# =============================================================================
# -- setting  -----------------------------------------------------------------
# =============================================================================



def default_settings(sync=False, render=True, dt=0.0):
    settings = carla.WorldSettings()
    settings.synchronous_mode = sync
    settings.no_rendering_mode = not render
    settings.fixed_delta_seconds = dt
    return settings



# =============================================================================
# -- tick  --------------------------------------------------------------------
# =============================================================================


# def tick_world(core: Core):
#     if core.settings.synchronous_mode:
#         return core.world.tick()

