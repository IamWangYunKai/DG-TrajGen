from carla_utils import carla

import time
import numpy as np
from collections import deque

from ..augment import error_state


class Controller(object):
    def __init__(self, config, dt, wheelbase):
        '''parameter'''
        self.max_acceleration = config.get('max_acceleration', 5.0)
        self.min_acceleration = config.get('min_acceleration', -10.0)
        self.max_throttle = config.get('max_throttle', 1.0)
        self.max_brake = config.get('max_brake', 1.0)
        self.max_steer = config.get('max_steer', np.deg2rad(45))

        Kp, Ki, Kd = 1.00, 0.01, 0.05
        self.v_param = (Kp, Ki, Kd)
        # k_theta, k_e = 0.2, 0.05
        k_theta, k_e = 0.2, 0.8
        self.w_param = (k_theta, k_e)

        self.v_controller = LongPID(dt, -self.max_brake, self.max_throttle)
        self.w_controller = LatRWPF(wheelbase, self.max_steer)


    def run_step(self, current_state, target_state):
        target_state.v -= 0.01
        throttle, brake = self.v_controller.run_step(current_state, target_state, self.v_param)
        steer = self.w_controller.run_step(current_state, target_state, self.w_param)
        steer_normalized = steer / self.max_steer
        return carla.VehicleControl(throttle=throttle, brake=brake, steer=steer_normalized)
    
    def control_to_acceleration(self, control: carla.VehicleControl):
        throttle, brake = control.throttle, control.brake
        if throttle == 0 and brake == 0: acceleration = 0
        elif throttle > 0 and brake == 0: acceleration = throttle * self.max_acceleration / self.max_throttle
        elif throttle == 0 and brake > 0: acceleration = brake * self.min_acceleration / self.max_brake
        else: raise NotImplementedError
        return acceleration



# ==============================================================================
# -- longitudinal --------------------------------------------------------------
# ==============================================================================
class LongPID(object):
    def __init__(self, dt, min_a, max_a):
        self.dt = dt
        self.min_a, self.max_a = min_a, max_a
        self.last_error = 0
        self.sum_error = 0

    def run_step(self, current_state, target_state, param):
        Kp, Ki, Kd = param[0], param[1], param[2]

        v_current = current_state.v
        v_target = target_state.v
        error = v_target - v_current

        acceleration = Kp * error
        acceleration += Ki * self.sum_error * self.dt
        acceleration += Kd * (error - self.last_error) / self.dt

        self.last_error = error
        self.sum_error += error
        '''eliminate drift'''
        if abs(self.sum_error) > 10:
            self.sum_error = 0.0

        throttle = np.clip(acceleration, 0, self.max_a)
        brake = -np.clip(acceleration, self.min_a, 0)
        return throttle, brake


# ==============================================================================
# -- lateral -------------------------------------------------------------------
# ==============================================================================
class LatRWPF(object):
    """
        Paper: A Survey of Motion Planning and Control Techniques for Self-driving Urban Vehicles
    """
    def __init__(self, wheelbase, max_steer):
        self.wheelbase = wheelbase
        self.max_steer = max_steer
        self.max_curvature = np.tan(self.max_steer) / self.wheelbase
        self.curvature_factor = 1.0
        self.alpha = 1.8

    def run_step(self, current_state, target_state, param):
        k_theta, k_e = param[0], param[1]

        longitudinal_e, lateral_e, theta_e = error_state(current_state, target_state)

        vr, kr = target_state.v, target_state.k


        c1 = (kr*self.curvature_factor) *np.cos(theta_e)
        c2 = - k_theta *theta_e
        c3 = (k_e*np.exp(-theta_e**2/self.alpha))*lateral_e
        curvature = c1 + c2 + c3

        curvature = np.clip(curvature, -self.max_curvature, self.max_curvature)
        steer = np.arctan(curvature * self.wheelbase)
        return steer


class LatPID(object):
    """
        agents.navigation.controller.PIDLateralController
    """
    def __init__(self, dt, max_steer):
        self.dt = dt
        self.max_steer = max_steer
        self._e_buffer = deque(maxlen=10)

    def run_step(self, current_state, target_state, param):
        Kp, Ki, Kd = param[0], param[1], param[2]

        v_vec = np.array([np.cos(current_state.theta), np.sin(current_state.theta), 0.0])
        w_vec = np.array([target_state.x-current_state.x, target_state.y-current_state.y, 0.0])
        _dot = np.arccos(np.clip(np.dot(w_vec, v_vec) / (np.linalg.norm(w_vec) * np.linalg.norm(v_vec)), -1.0, 1.0))

        _cross = np.cross(v_vec, w_vec)

        if _cross[2] < 0:
            _dot *= -1.0

        self._e_buffer.append(_dot)
        if len(self._e_buffer) >= 2:
            _de = (self._e_buffer[-1] - self._e_buffer[-2]) / self.dt
            _ie = sum(self._e_buffer) * self.dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip((Kp * _dot) + (Kd * _de) + (Ki * _ie), -1.0, 1.0)

