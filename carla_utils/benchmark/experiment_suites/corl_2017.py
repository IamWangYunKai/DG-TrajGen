# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# CORL experiment set.

from __future__ import print_function

from carla_utils.benchmark.settings import CarlaSettings
from carla_utils.benchmark.experiment import Experiment

from .experiment_suite import ExperimentSuite
from .CoRL_2017 import param



class CoRL2017(ExperimentSuite):

    @property
    def train_weathers(self):
        return param.train_weathers
        return [1, 3, 6, 8]

    @property
    def test_weathers(self):
        return param.test_weathers
        return [4, 14]

    def _poses_town01(self):
        return [param.Town01.poses_straight(self.town_map),
                param.Town01.poses_one_turn(self.town_map),
                param.Town01.poses_navigation(self.town_map),
                param.Town01.poses_navigation_with_dynamic_obstacles(self.town_map)]

    def _poses_town02(self):
        return [param.Town02.poses_straight(self.town_map),
                param.Town02.poses_one_turn(self.town_map),
                param.Town02.poses_navigation(self.town_map),
                param.Town02.poses_navigation_with_dynamic_obstacles(self.town_map)]

    def build_experiments(self):
        """
        Creates the whole set of experiment objects,
        The experiments created depend on the selected Town.


        """

        # We set the camera
        # This single RGB camera is used on every experiment

        # camera = Camera('CameraRGB')
        # camera.set(FOV=100)
        # camera.set_image_size(800, 600)
        # camera.set_position(2.0, 0.0, 1.4)
        # camera.set_rotation(-15.0, 0, 0)
        camera = self.sensor_manager['camera']

        if self._city_name == 'Town01':
            poses_tasks = self._poses_town01()
            vehicles_tasks = [0, 0, 0, param.Town01.number_of_vehicles]
            pedestrians_tasks = [0, 0, 0, param.Town01.number_of_pedestrians]
        else:
            poses_tasks = self._poses_town02()
            vehicles_tasks = [0, 0, 0, param.Town02.number_of_vehicles]
            pedestrians_tasks = [0, 0, 0, param.Town02.number_of_pedestrians]

        experiment_list = []
        # print('poses_tasks: '+str(len(poses_tasks)))

        for weather in self.weathers:

            for iteration in range(len(poses_tasks)):
                poses = poses_tasks[iteration]
                vehicles = vehicles_tasks[iteration]
                pedestrians = pedestrians_tasks[iteration]

                settings = CarlaSettings()
                settings.set(
                    SendNonPlayerAgentsInfo=True,
                    NumberOfVehicles=vehicles,
                    NumberOfPedestrians=pedestrians,
                    WeatherId=weather
                )
                # Add all the cameras that were set for this experiments

                settings.add_sensor(camera)

                experiment = Experiment()
                experiment.set(
                    Conditions=settings,
                    Poses=poses,
                    Task=iteration,
                    Repetitions=1
                )
                experiment_list.append(experiment)

        return experiment_list
