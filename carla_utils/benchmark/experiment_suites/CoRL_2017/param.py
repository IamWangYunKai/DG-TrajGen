
'''
weather presets:

    ClearNoon, ClearSunset, CloudyNoon, CloudySunset, Default,
    HardRainNoon, HardRainSunset, MidRainSunset, MidRainyNoon,
    SoftRainNoon, SoftRainSunset, WetCloudyNoon, WetCloudySunset,
    WetNoon, WetSunset.

available maps:

    Town01, Town02, Town03, Town04, Town05.

'''


import os
dir_path = os.path.split(os.path.realpath(__file__))[0] + '/'

from .read import read


train_weathers = ['ClearNoon', 'ClearSunset', 'CloudyNoon', 'CloudySunset', 'Default']
test_weathers = ['SoftRainNoon', 'SoftRainSunset', 'WetCloudyNoon', 'WetCloudySunset']

class Town01(object):
    number_of_vehicles = 20
    number_of_pedestrians = 50

    @staticmethod
    def poses_straight(town_map):
        poses = read(dir_path + 'Town01/straight.csv', town_map)
        return poses
    @staticmethod
    def poses_one_turn(town_map):
        poses = read(dir_path + 'Town01/one_turn.csv', town_map)
        return poses
    @staticmethod
    def poses_navigation(town_map):
        poses = read(dir_path + 'Town01/navigation.csv', town_map)
        return poses
    @staticmethod
    def poses_navigation_with_dynamic_obstacles(town_map):
        poses = read(dir_path + 'Town01/navigation_with_dynamic_obstacles.csv', town_map)
        return poses



class Town02(object):
    number_of_vehicles = 15
    number_of_pedestrians = 50

    @staticmethod
    def poses_straight(town_map):
        poses = read(dir_path + 'Town02/straight.csv', town_map)
        return poses
    @staticmethod
    def poses_one_turn(town_map):
        poses = read(dir_path + 'Town02/one_turn.csv', town_map)
        return poses
    @staticmethod
    def poses_navigation(town_map):
        poses = read(dir_path + 'Town02/navigation.csv', town_map)
        return poses
    @staticmethod
    def poses_navigation_with_dynamic_obstacles(town_map):
        poses = read(dir_path + 'Town02/navigation_with_dynamic_obstacles.csv', town_map)
        return poses