from carla_utils import carla

import numpy as np
from typing import List
import cv2
import copy
import time

from .. import basic
from ..augment import ActorVertices, cvt
from ..agents import BaseAgent, BaseAgentObstacle

from .town_map_image import TownMapImage
from .town_map_image import COLOR_CHAMELEON_0, COLOR_CHOCOLATE_1, COLOR_WHITE


def d2_local(obstacle: BaseAgentObstacle, agent: BaseAgent, expand=carla.Vector2D(0.0,0.0)):
    dx, dy = obstacle.bounding_box.x +expand.x, obstacle.bounding_box.y +expand.y

    state = obstacle.get_state().world2local(agent.get_state())
    center_x, center_y, theta = state.x, state.y, state.theta

    l, n = np.array([np.cos(theta), np.sin(theta)]), np.array([np.cos(theta+np.pi/2), np.sin(theta+np.pi/2)])
    vertices = np.expand_dims(np.array([center_x, center_y]), axis=0).repeat(4, axis=0)

    vertices[0] +=  l*dx + n*dy
    vertices[1] += -l*dx + n*dy
    vertices[2] += -l*dx - n*dy
    vertices[3] +=  l*dx - n*dy
    return vertices


def sub_image(image, center, theta, width, height):
	"""Extract a rectangle from the source image.
	
	image - source image
	center - (x,y) tuple for the centre point.
	theta - angle of rectangle.
	width, height - rectangle dimensions.
	"""
	
	if 45 < theta <= 90:
		theta = theta - 90
		width, height = height, width
		
	# theta *= np.pi / 180 # convert to rad
	v_x = (np.cos(theta), np.sin(theta))
	v_y = (-np.sin(theta), np.cos(theta))
	s_x = center[0] - v_x[0] * (width / 2) - v_y[0] * (height / 2)
	s_y = center[1] - v_x[1] * (width / 2) - v_y[1] * (height / 2)
	mapping = np.array([[v_x[0],v_y[0], s_x], [v_x[1],v_y[1], s_y]])

	return cv2.warpAffine(image, mapping, (width, height), flags=cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)



class BirdEyeView(object):
    def __init__(self, core, dim_state, range_min=None, range_max=None):
        '''
            range_min < 0, range_max > 0

        '''

        self.core = core
        self.dim_state = dim_state
        if range_min == None: range_min = carla.Vector2D(x=-25, y=-30)
        if range_max == None: range_max = carla.Vector2D(x=45, y=30)

        self.resolution = 0.1
        self.town_map_image = TownMapImage(core, self.resolution)

        assert range_max.x >= -range_min.x
        assert range_max.y + range_min.y == 0
        self.range_x_min, self.range_x_max = range_min.x, range_max.x
        self.range_y_max = range_max.y
        self.radius = 0.5 * (self.range_x_max - self.range_x_min) * np.sqrt(2)
        self.offset = np.array([self.range_x_min, -self.range_y_max])

        self.height, self.width = int(2*self.range_y_max / self.resolution), int((self.range_x_max - self.range_x_min) / self.resolution)
        self.channels = 3



    def run_step(self, agent: BaseAgent, obstacles: List[BaseAgentObstacle]):
        ### crop global image
        state_ego = agent.get_state()

        center = np.array([[state_ego.x, state_ego.y]])
        ##### find image center, maybe not center of vehicle
        center += 0.5*(self.range_x_max+self.range_x_min) *np.array([[np.cos(state_ego.theta), np.sin(state_ego.theta)]])
        center_pix = (center - self.town_map_image.world_offset) / self.town_map_image.resolution  ## world to image
        center_pix = center_pix[0].astype(np.int64)

        image_bev = sub_image(self.town_map_image.image_cv2, center_pix, state_ego.theta, self.width, self.height)

        ### route
        waypoints, _ = agent.global_path.remaining_waypoints(agent.get_transform())
        waypoints = waypoints[:150:15]   ## TODO change

        corners = []
        for p in waypoints:
            t = p.transform
            s = cvt.CUAState.carla_transform(t).world2local(state_ego)
            corners.append([s.x, s.y])

        pts = (np.array(corners) - self.offset) / self.resolution
        color = (0,0,255)
        cv2.polylines(image_bev, [pts.astype(np.int64)], False, color, 8)

        ### ego
        vertices_ego = d2_local(agent, agent)
        boundary_ego = (vertices_ego - self.offset) / self.resolution

        cv2.fillPoly(image_bev, pts=[boundary_ego.astype(np.int64)], color=COLOR_CHAMELEON_0)

        ### obs
        for obs in obstacles:
            state_obs = obs.get_state()
            if state_ego.distance_xy(state_obs) > self.radius:
                continue

            vertices_obs = d2_local(obs, agent)
            boundary_obs = (vertices_obs - self.offset) / self.resolution
            cv2.fillPoly(image_bev, pts=[boundary_obs.astype(np.int64)], color=COLOR_CHOCOLATE_1)
        
        return image_bev



class BridEyeViewChannelWise(BirdEyeView):
    def __init__(self, core, dim_state, range_min=None, range_max=None):
        super().__init__(core, dim_state, range_min, range_max)
        self.channels = 5

    def run_step(self, agent: BaseAgent, obstacles: List[BaseAgentObstacle]):

        ### crop global image
        state_ego = agent.get_state()

        center = np.array([[state_ego.x, state_ego.y]])
        ##### find image center, maybe not center of vehicle
        center += 0.5*(self.range_x_max+self.range_x_min) *np.array([[np.cos(state_ego.theta), np.sin(state_ego.theta)]])
        center_pix = (center - self.town_map_image.world_offset) / self.town_map_image.resolution  ## world to image
        center_pix = center_pix[0].astype(np.int64)

        image_background = sub_image(self.town_map_image.image_cv2, center_pix, state_ego.theta, self.width, self.height)
        image_background = np.expand_dims(cv2.cvtColor(image_background, cv2.COLOR_RGB2GRAY), 2)   ### reduce channel

        ### route
        image_route = np.zeros((self.height, self.width, 1), dtype=np.uint8)
        waypoints, _ = agent.global_path.remaining_waypoints(agent.get_transform())
        waypoints = waypoints[:150:15]   ## TODO change

        corners = []
        for p in waypoints:
            t = p.transform
            s = cvt.CUAState.carla_transform(t).world2local(state_ego)
            corners.append([s.x, s.y])

        pts = (np.array(corners) - self.offset) / self.resolution
        cv2.polylines(image_route, [pts.astype(np.int64)], False, COLOR_WHITE, 20)

        ### ego
        image_ego = np.zeros((self.height, self.width, 1), dtype=np.uint8)
        vertices_ego = d2_local(agent, agent)
        boundary_ego = (vertices_ego - self.offset) / self.resolution

        cv2.fillPoly(image_ego, pts=[boundary_ego.astype(np.int64)], color=COLOR_WHITE)

        ### obs
        image_obs = np.zeros((self.height, self.width, 1), dtype=np.uint8)
        image_obs_v = np.zeros((self.height, self.width, 1), dtype=np.uint8)
        for obs in obstacles:
            state_obs = obs.get_state()
            if state_ego.distance_xy(state_obs) > self.radius:
                continue

            vertices_obs = d2_local(obs, agent)
            boundary_obs = (vertices_obs - self.offset) / self.resolution
            cv2.fillPoly(image_obs, pts=[boundary_obs.astype(np.int64)], color=COLOR_WHITE)
            color_v = int((obs.get_current_v() / agent.max_velocity / 1.1) * 255)
            color_v = int(np.clip(color_v, 0, 255))
            # color_v = 0 ### ! warning
            cv2.fillPoly(image_obs_v, pts=[boundary_obs.astype(np.int64)], color=color_v)
        
        image = np.concatenate([image_background, image_route, image_ego, image_obs, image_obs_v], axis=2)
        image = cv2.resize(image, self.dim_state)
        return image


    def viz(self, image):
        image_background = image[:,:,0]
        image_route = image[:,:,1]
        image_ego = image[:,:,2]
        image_obs = image[:,:,3]

        iib = cv2.cvtColor(image_background, cv2.COLOR_GRAY2BGR)
        iii = np.hstack([iib, np.stack([image_route, image_ego, image_obs], axis=2)])
        cv2.imshow('haha', iii)
        cv2.waitKey(1)
        # cv2.imwrite('./results/tmp/' + str(time.time()) + '.png', iii)

