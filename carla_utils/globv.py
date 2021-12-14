
from copy import copy

from .system import nameddict



def init():
    GlobV = nameddict('GlobV', ())
    global _global_value_dict
    _global_value_dict = GlobV()


def set(key, value):
    if key in globals().keys():
        raise RuntimeError('key ' + str(key) + ' already setted: ', globals()[key])
        # print('[carla_utils.globv] Warning: ' + 'key ' + str(key) + ' already setted: ', globals()[key])
    globals()[key] = value
    _global_value_dict[key] = value

def get(key):
    return _global_value_dict.get(key, None)

def variable():
    return copy(_global_value_dict)
