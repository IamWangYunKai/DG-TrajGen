
import os, sys
from os.path import join
import glob
import json
import inspect
import psutil


def load_carla_standard():
    try:
        server_path = os.environ['CARLAPATH']
    except:
        print('run this in shell:\n    echo "export CARLAPATH=/your/carla/server/path" >> ~/.bashrc')
        exit(0)
    try:
        sys.path.append(server_path+'/PythonAPI/carla')
        sys.path.append(glob.glob(server_path+'/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
            sys.version_info.major,
            sys.version_info.minor,
            'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
    except:
        print('Fail to load carla library')
        print('run this in shell:\n    echo "export CARLAPATH=/your/carla/server/path" >> ~/.bashrc')
        exit(0)


def load_carla():
    basic_path = os.path.split(os.path.split(__file__)[0])[0]
    path = join(basic_path, 'carla_api')
    try:
        sys.path.append(path+'/carla')
        carla_path = glob.glob(path+'/carla/dist/carla-*%d.%d-%s.egg' % (
            sys.version_info.major,
            sys.version_info.minor,
            'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0]
        sys.path.append(carla_path)
    except:
        print('Fail to load carla library')


def get_carla_version():
    server_path = os.environ['CARLAPATH']
    carla_version = server_path.split('/')[-1].split('_')[-1]
    return carla_version



def printVariable(name, value):
    print(name + ': ' + str(value))


def parse_json_file(file_path):
    if not os.path.exists(file_path):
        raise RuntimeError("Could not read json file from {}".format(file_path))
    json_dict = None
    with open(file_path) as handle:
        json_dict = json.loads(handle.read())
    return json_dict



class Singleton(object):
    _instance = None

    def __new__(cls, *args, **kw):
        if not cls._instance:
            cls._instance = super(Singleton, cls).__new__(cls)  
        return cls._instance


def debug(info, info_type='debug'):
    if info_type == 'error':
        print('\033[1;31m ERROR:', info, '\033[0m')
    elif info_type == 'success':
        print('\033[1;32m SUCCESS:', info, '\033[0m')
    elif info_type == 'warning':
        print('\033[1;34m WARNING:', info, '\033[0m')
    elif info_type == 'debug':
        print('\033[1;35m DEBUG:', info, '\033[0m')
    else:
        print('\033[1;36m MESSAGE:', info, '\033[0m')


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var][0]



def is_used(port):
    '''Checks whether or not a port is used'''
    return port in [conn.laddr.port for conn in psutil.net_connections()]


def kill_process():
    import os, signal
    os.kill(os.getpid(), signal.SIGKILL)
