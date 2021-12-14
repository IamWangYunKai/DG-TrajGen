
import time
import numpy as np
import copy

from .state import State, getActorState

def getObstacle(frame_id, time_stamp, obstacle, distance):
    state = getActorState(frame_id, time_stamp, obstacle)
    scale_factor = 5.0 if 'walker' in obstacle.type_id else 1.0
    return Obstacle(frame_id, time_stamp, obstacle.id, obstacle.bounding_box, state, distance, scale_factor)


class Obstacle(object):
    def __init__(self, frame_id, time_stamp, obstacle_id, bounding_box, state, distance, scale_factor=1.0, **kwargs):
        # default params
        self.frame_id = frame_id
        self.time_stamp = time_stamp
        self.id = obstacle_id

        self.length = bounding_box.extent.x*2 *scale_factor
        self.width = bounding_box.extent.y*2 *scale_factor
        self.height = bounding_box.extent.z*2
        self.radius = np.hypot(self.length/2, self.width/2)

        self.state = state
        self.distance = distance

        # delay
        self.predict_steps, self.predict_states = None, None
        self.predict_states = []




    ###############################################
    ############ Delayed gratification ############
    ###############################################
    def set_predict_states(self, time_steps, dt):
        self.predict_steps, self.predict_states = 0, []
        statef = self.state
        statef.t = 0
        self.predict_states.append(statef)
        for time_step in range(1, time_steps+1):
            statef = PredictionModel.constant_velocity(statef, dt)
            self.predict_states.append(statef)
            self.predict_steps += 1

    def contains_state(self, state, time_step=0, use='circle'):
        res = False
        predict_state = None
        if use == 'circle':
            predict_state = self.predict_states[time_step]
            dist = np.hypot(predict_state.x-state.x, predict_state.y-state.y)
            if dist <= self.radius:
                res = True
        return res, predict_state



class PredictionModel(object):
    @staticmethod
    def constant_velocity(state0, dt):
        if state0.t < 0:
            print('augment obstacle: wrong!!!')
        v, theta = state0.v, state0.theta
        x = state0.x + v*np.cos(theta)*dt
        y = state0.y + v*np.sin(theta)*dt
        statef = State(x=x, y=y, theta=theta, v=v, t=dt+state0.t)
        return statef

    @staticmethod
    def zero_velocity(state0, dt):
        return copy.copy(state0)