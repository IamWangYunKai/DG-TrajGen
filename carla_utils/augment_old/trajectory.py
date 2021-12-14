
import numpy as np

class Trajectory(object):
    def __init__(self, frame_id, time_stamp, curve, velocity_distribution):
        self.frame_id = frame_id
        self.time_stamp = time_stamp
        self.curve = curve
        self.velocity_distribution = velocity_distribution

        self.total_time = velocity_distribution.total_time
        self.curve_length = curve.curve_length


    def get_state(self, t):
        s, v, a = self.velocity_distribution.get_kinematics(t)
        state = self.curve.get_state(s)
        state.s, state.v, state.a, state.t = s, v, a, t
        return state


    def discretizeToStates(self, start=0, end=-1, sample_number=40):
        t_array = np.linspace(start, end, sample_number)
        states = []
        for i, t in enumerate(t_array):
            state = self.get_state(t)
            states.append(state)
        return states