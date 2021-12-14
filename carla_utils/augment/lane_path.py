
import time
import numpy as np


class LanePath(object):
    def __init__(self, lane_id, **kwargs):
        # default params
        self.size = 0
        self.waypoints = []
        self.steps = []

        self.lane_id = lane_id


    def append(self, step, waypoint):
        self.steps.append(step)
        self.waypoints.append(waypoint)
        self.size += 1
