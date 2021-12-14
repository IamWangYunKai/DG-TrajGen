from carla_utils import carla
DestroyActor = carla.command.DestroyActor

from abc import ABC
import threading
import numpy as np

from ..augment import GlobalPath
from ..sensor import CarlaSensorListMaster
from ..world_map import get_reference_route_wrt_waypoint, Core, Role
from ..system import Clock
from .controller import Controller
from .tools import vehicle_wheelbase
from .vehicle_model import RealModel, SteerModel


class AgentABC(ABC):
    def __init__(self, config, vehicle, sensors_master, global_path):
        """
        
        
        Args:
            config: contains decision_frequency, control_frequency,
                    max_velocity, max_acceleration, min_acceleration,
                    max_throttle, max_brake, max_steer
        
        Returns:
            None
        """
        
        '''config'''
        self.config = config
        self.real = config.real
        self.core: Core = config.get('core', None)

        self.decision_frequency = config.decision_frequency
        self.control_frequency = config.control_frequency
        self.skip_num = int(self.control_frequency // self.decision_frequency)
        assert self.control_frequency % self.decision_frequency == 0

        self.perception_range = float(config.perception_range)

        self.tick_time = 0

        self.clock = Clock(self.control_frequency)

        '''vehicle'''
        self.world, self.town_map, self.vehicle = self.core.world, self.core.town_map, vehicle
        self.sensors_master: CarlaSensorListMaster = sensors_master
        self.global_path: GlobalPath = global_path

        self.id = vehicle.id
        self.bounding_box = vehicle.bounding_box.extent

        self.role_name: Role = Role.loads(vehicle.attributes['role_name'])
        self.vi = self.role_name.vi

        self.decision_dt, self.control_dt = 1.0/self.decision_frequency, 1.0/self.control_frequency
        self.wheelbase = vehicle_wheelbase(self.vehicle)
        self.controller = Controller(config, self.control_dt, self.wheelbase)
        self.max_velocity = float(config.get('max_velocity', 8.34))
        self.max_acceleration = self.controller.max_acceleration
        self.min_acceleration = self.controller.min_acceleration
        self.max_throttle = self.controller.max_throttle
        self.max_brake = self.controller.max_brake
        self.max_steer = self.controller.max_steer
        self.max_curvature = np.tan(self.max_steer) / self.wheelbase

        self.steer_model = SteerModel(self.control_dt, alpha=0.0)
        self.vehicle_model = RealModel()
        self.init()
        return


    def extend_route(self):
        waypoint = self.global_path.carla_waypoints[-1]
        sr = self.global_path.sampling_resolution
        route = get_reference_route_wrt_waypoint(waypoint, sr, round(3*self.perception_range / sr))
        self.global_path.extend(route[1:])
        # self.global_path.draw(self.world, life_time=0)

    def goal_reached(self, preview_distance):
        return self.global_path.reached(preview_distance)

    def destroy(self):
        if self.real:
            self.thread_stoped = True
            self.thread_control.join()
        self.sensors_master.destroy()
        self.vehicle.destroy()
    
    def destroy_commands(self):
        '''
            deprecated
        '''
        cmds = self.sensors_master.destroy_commands()
        cmds.append(DestroyActor(self.vehicle))
        return cmds
    

    def check_collision(self):
        return self.sensors_master[('sensor.other.collision', 'default')].get_raw_data() != None

    def check_timeout(self, time_tolerance):
        self.tick_time += 1
        return self.tick_time >= time_tolerance


    def init(self):
        if self.real:
            self.run_step = self._run_step_real
            print('[{}] Agent is in real mode.'.format(self.__class__.__name__))
            
            self.target = None
            self.stop_control = carla.VehicleControl(brake=0.87654)
            self.thread_stoped = False
            self.thread_control = threading.Thread(target=self._run_control, args=(lambda: self.thread_stoped,))
            self.thread_control.start()
        else:
            self.run_step = self._run_step_fast


    def _run_step_fast(self, reference):
        target = self.get_target(reference)
        for _ in range(self.skip_num):
            if self.goal_reached(self.perception_range *1.2): self.extend_route()
            control = self.get_control(target)
            self.forward(control)
            self.core.tick()
        return


    def _run_step_real(self, reference):
        self.target = self.get_target(reference)
        return

    def _run_control(self, stop):
        while True:
            self.clock.tick_begin()
            if self.goal_reached(self.perception_range *1.2): self.extend_route()
            control = self.get_control(self.target) if self.target != None else self.stop_control
            self.forward(control)
            if stop(): break
            self.clock.tick_end()
        self.vehicle_model(self.vehicle, self.stop_control)
        print('[BaseAgent-Real] Exit control thread.')
        return
