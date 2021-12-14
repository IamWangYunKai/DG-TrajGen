
import time
from enum import IntEnum
import numpy as np


def getZeroStation(frame_id, time_stamp):
    station = Station(frame_id, time_stamp, station_type=StationType.CONSTANT_STATION)
    station.set_station_param(piecewise_number=1, params=[0.0], knot_time=[0.0, 10.0])
    return station


class StationType(IntEnum):
    CONSTANT_STATION = 0
    QUINTIC_STATION = 5


# base vector
s_vector = lambda t: np.array([[1, t, t**2, t**3, t**4, t**5]])
v_vector = lambda t: np.array([[0, 1, 2*t, 3*t**2, 4*t**3, 5*t**4]])
a_vector = lambda t: np.array([[0, 0, 2, 6*t, 12*t**2, 20*t**3]])


class Station(object):
    def __init__(self, frame_id, time_stamp, **kwargs):
        # default params
        self.frame_id = frame_id
        self.time_stamp = time_stamp
        self.is_parameterized = True
        self.station_type = StationType.QUINTIC_STATION
        self.piecewise_number = 0
        self.params = []
        self.knot_time = [0.0]
        self.total_time = 0.0

        # parameters overload
        if 'is_parameterized' in kwargs:
            self.is_parameterized = kwargs['is_parameterized']
        if 'station_type' in kwargs:
            self.station_type = kwargs['station_type']

        if self.station_type == StationType.QUINTIC_STATION:
            self.order = 5
        elif self.station_type == StationType.CONSTANT_STATION:
            self.order = 0


    def set_station_param(self, piecewise_number, params, knot_time):
        self.piecewise_number = piecewise_number
        self.params = params
        self.knot_time = knot_time
        self.total_time = knot_time[-1]
        return True


    def get_kinematics(self, t):
        t = np.clip(t, 0, self.total_time)

        index = None
        for i in range(self.piecewise_number):
            if t <= self.knot_time[i+1]:
                index = i
                break

        param_num = self.order + 1
        t0 = self.knot_time[index]
        param = np.array(self.params[index*param_num : (index+1)*param_num]).reshape(param_num, 1)

        # for CONSTANT_STATION and QUINTIC_STATION
        s = float(np.dot(s_vector(t)[:,:param_num], param))
        v = float(np.dot(v_vector(t)[:,:param_num], param))
        a = float(np.dot(a_vector(t)[:,:param_num], param))

        return s, v, a
