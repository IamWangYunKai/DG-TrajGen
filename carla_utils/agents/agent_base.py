from carla_utils import carla

from ..augment import cvt, vector3DNorm
from .agent_abc import AgentABC


class BaseAgent(AgentABC):
    def get_target(self, reference):
        return self.max_velocity
    
    def get_transform(self):
        return self.vehicle.get_transform()
    def get_current_v(self):
        return vector3DNorm(self.vehicle.get_velocity())
    def get_state(self):
        current_transform = self.get_transform()
        current_v = self.get_current_v()
        return cvt.CUAState.carla_transform(current_transform, v=current_v)


    def get_control(self, target):
        current_transform = self.get_transform()
        target_waypoint, curvature = self.global_path.target_waypoint(current_transform)
        # draw_arrow(self.world, target_waypoint.transform, life_time=0.1)

        current_v = self.get_current_v()
        current_state = cvt.CUAState.carla_transform(current_transform, v=current_v)
        target_state = cvt.CUAState.carla_transform(target_waypoint.transform, v=target, k=curvature)
        control = self.controller.run_step(current_state, target_state)
        return control
    
    def forward(self, control):
        control.steer = self.steer_model(control.steer)
        self.vehicle_model(self.vehicle, control)

