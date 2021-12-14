
import time
import numpy as np
np.set_printoptions(precision=4, suppress=True)
from enum import IntEnum
import matplotlib.pyplot as plt

from ..trajectory import QuadraticSpiral, ConstantSpiral
from ..augment.state import State


class CurveType(IntEnum):
    CONSTANT_SPIRAL = 0
    QUADRATIC_SPIRAL = 2


class Curve(object):
    def __init__(self, frame_id, time_stamp, **kwargs):
        # default params
        self.frame_id = frame_id
        self.time_stamp = time_stamp
        self.is_parameterized = True
        self.curve_type = CurveType.QUADRATIC_SPIRAL
        self.piecewise_number = 0
        self.knot_states = []
        self.params = []
        self.knot_lengths = [0.0]
        self.curve_length = 0.0

        # parameters overload
        if 'is_parameterized' in kwargs:
            self.is_parameterized = kwargs['is_parameterized']
        if 'curve_type' in kwargs:
            self.curve_type = kwargs['curve_type']
        if 'vehicle_frame_id' in kwargs:
            self.vehicle_frame_id = kwargs['vehicle_frame_id']

        if self.curve_type == CurveType.QUADRATIC_SPIRAL:
            self.order = 2
        elif self.curve_type == CurveType.CONSTANT_SPIRAL:
            self.order = 0

        # delay
        self.cost, self.max_curvature = None, None


    def add_curve_param(self, knot_state, param, piecewise_length):
        '''
            knot_state: CUA/State
            param: list
            piecewise_length: float
        '''
        self.knot_states.append(knot_state)
        self.params.extend(param)
        self.knot_lengths.append(self.knot_lengths[-1] + piecewise_length);
        self.curve_length = self.knot_lengths[-1]
        self.piecewise_number += 1
        return True

    def add_curve(self, curve):
        if self.is_parameterized != curve.is_parameterized or self.curve_type != curve.curve_type:
            return False
        param_num = curve.order + 1
        for i in range(curve.piecewise_number):
            knot_state = curve.knot_states[i]
            param = curve.params[i*param_num : (i+1)*param_num]
            piecewise_length = curve.knot_lengths[i+1] - curve.knot_lengths[i]
            self.add_curve_param(knot_state, param, piecewise_length)
        return True


    def get_piecewise_index(self, s):
        station = np.clip(s, 0, self.curve_length)
        index = None
        for i in range(self.piecewise_number):
            if station <= self.knot_lengths[i+1]:
                index = i
                break
        return index

    def get_piecewise_param(self, s):
        station = np.clip(s, 0, self.curve_length)
        index = self.get_piecewise_index(station)
        param_num = self.order + 1
        s0, state0 = self.knot_lengths[index], self.knot_states[index]
        param = np.array(self.params[index*param_num : (index+1)*param_num]).reshape(param_num, 1)
        return param, s0, state0

    def get_state(self, s):
        station = np.clip(s, 0, self.curve_length)
        param, s0, state0 = self.get_piecewise_param(station)

        if self.curve_type == CurveType.QUADRATIC_SPIRAL:
            # QuadraticSpiral
            x_local = QuadraticSpiral.x_local(station-s0, param)
            y_local = QuadraticSpiral.y_local(station-s0, param)
            theta = QuadraticSpiral.theta(station-s0, param)
            k = QuadraticSpiral.curvature(station-s0, param)
            local_state = State(x=x_local,y=y_local,theta=theta,k=k)
            world_state = local_state.local2world(state0)
        elif self.curve_type == CurveType.CONSTANT_SPIRAL:
            # ConstantSpiral
            x_local = ConstantSpiral.x_local(station-s0, param)
            y_local = ConstantSpiral.y_local(station-s0, param)
            theta = ConstantSpiral.theta(station-s0, param)
            k = ConstantSpiral.curvature(station-s0, param)
            local_state = State(x=x_local,y=y_local,theta=theta,k=k)
            world_state = local_state.local2world(state0)
        else:
            raise NotImplementedError
        return world_state


    def discretizeToStates(self, start=0, end=-1, sample_number=40):
        s_array = np.linspace(start, end, sample_number)
        states = []
        for i, s in enumerate(s_array):
            state = self.get_state(s)
            states.append(state)
        return states

    def draw_plt(self, start=0, end=-1, linewidth=4, show_text=True, sample_number=100):
        states = self.discretizeToStates(start, end, sample_number)
        x_array = [state.x for state in states]
        y_array = [state.y for state in states]
        theta_array = [state.theta for state in states]
        k_array = [state.k for state in states]
        plt.plot(x_array, y_array, '-r', linewidth=linewidth)
        for i, (x,y) in enumerate(zip(x_array, y_array)):
            if i % 2 == 0 and show_text:
                plt.text(x,y, str(i))
        for state in self.knot_states:
            state.draw_plt(style='arrow', fmt='ob', color='deepskyblue')

        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlim((min(x_array)-0.5, max(x_array)+0.5))
        plt.ylim((min(y_array)-0.5, max(y_array)+0.5))
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')

        return x_array, y_array, theta_array, k_array



    ###############################################
    ############ Delayed gratification ############
    ###############################################
    def set_cost(self, cost):
        self.cost = float(cost)

    def set_max_curvature(self, max_curvature):
        self.max_curvature = float(max_curvature)



