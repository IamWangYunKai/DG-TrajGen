from carla_utils import carla
DestroyActor = carla.command.DestroyActor

from ..sensor import CarlaSensorListMaster
from ..world_map import Role
from .agent_base import BaseAgent


class BaseAgentObstacle(BaseAgent):
    def __init__(self, config, vehicle, sensors_master):
        """
        
        
        Args:
        
        Returns:
            None
        """
        
        '''config'''
        self.real = config.real
        self.core = config.get('core', None)

        '''vehicle'''
        self.world, self.town_map, self.vehicle = self.core.world, self.core.town_map, vehicle
        self.sensors_master: CarlaSensorListMaster = sensors_master

        self.id = vehicle.id
        self.bounding_box = vehicle.bounding_box.extent
        
        self.role_name: Role = Role.loads(vehicle.attributes['role_name'])
        self.vi = self.role_name.vi

        self.run_step = lambda _: RuntimeError('[BaseObstacle] Forbidden')
        return


    def get_target(self, reference):
        raise RuntimeError('[BaseObstacle] Forbidden')

    def get_control(self, target):
        raise RuntimeError('[BaseObstacle] Forbidden')
    
    def forward(self, control):
        raise RuntimeError('[BaseObstacle] Forbidden')


    def extend_route(self):
        raise RuntimeError('[BaseObstacle] Forbidden')

    def goal_reached(self, preview_distance):
        raise RuntimeError('[BaseObstacle] Forbidden')

    def destroy(self):
        raise RuntimeError('[BaseObstacle] Forbidden')
    

    def check_collision(self):
        raise RuntimeError('[BaseObstacle] Forbidden')

    def check_timeout(self, time_tolerance):
        raise RuntimeError('[BaseObstacle] Forbidden')


    def init(self):
        raise RuntimeError('[BaseObstacle] Forbidden')


    def _run_step_fast(self, reference):
        raise RuntimeError('[BaseObstacle] Forbidden')


    def _run_step_real(self, reference):
        raise RuntimeError('[BaseObstacle] Forbidden')

    def _run_control(self, stop):
        raise RuntimeError('[BaseObstacle] Forbidden')
