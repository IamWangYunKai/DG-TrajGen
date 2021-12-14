
import time
import numpy as np

from .state import getActorState
from .obstacle import getObstacle

def getObstacleArray(frame_id, time_stamp, world, vehicle, distance_range):
    current_state = getActorState(frame_id, time_stamp, vehicle)
    current_location = vehicle.get_location()
    vehicle_id = vehicle.id
    distance_func = lambda loc: loc.distance(current_location)

    vehicles = world.get_actors().filter('vehicle.*')
    obstacles = [(distance_func(o.get_location()), o) for o in vehicles if o.id != vehicle_id]
    walkers = world.get_actors().filter('walker.*')
    obstacles.extend( [(distance_func(o.get_location()), o) for o in walkers] )
    sorted_obstacles = sorted(obstacles, key=lambda x:x[0])

    obstacle_array = ObstacleArray(frame_id, time_stamp)
    for distance, obstacle_actor in sorted_obstacles:
        if distance > distance_range:
            break
        obstacle = getObstacle(frame_id, time_stamp, obstacle_actor, distance)
        obstacle_array.append(obstacle)
    return obstacle_array



class ObstacleArray(object):
    def __init__(self, frame_id, time_stamp, **kwargs):
        # default params
        self.frame_id = frame_id
        self.time_stamp = time_stamp
        self.number = 0
        self.obstacles = []


    def append(self, obstacle):
        self.obstacles.append(obstacle)
        self.number += 1




    ###############################################
    ############ Delayed gratification ############
    ###############################################
    def set_obstacle_predict_states(self, time_steps, dt):
        for obstacle in self.obstacles:
            obstacle.set_predict_states(time_steps, dt)