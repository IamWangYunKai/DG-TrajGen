from carla_utils import carla
tls = carla.TrafficLightState

import numpy as np
import pygame
import time
import copy
import cv2

from ..agents import BaseAgent
from ..examples import map_image as bv
from ..examples.tools import Util

from ..world_map import Core, Role



# class BirdView(object):
#     def __init__(self, core: Core, resolution, range: carla.Vector2D):
#         self.core = core
#         self.resolution, self.range = resolution, range
#         self.radius = np.hypot(range.x, range.y)

#         pygame.init()
#         pygame.font.init()
#         self.display = pygame.display.set_mode(
#             (1280, 720),
#             pygame.HWSURFACE | pygame.DOUBLEBUF)


#         self.map_image = bv.MapImage(
#             carla_world=core.world,
#             carla_map=core.town_map,
#             pixels_per_meter=bv.PIXELS_PER_METER,
#             show_triggers=False,
#             show_connections=False,
#             show_spawn_points=False)
        
#         image = pygame.surfarray.array3d(self.map_image.surface)
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#         cv2.imshow('haha', image)
#         print('here !!!')
#         cv2.waitKey(0)



def _split_actors(actors_with_transforms):
    """Splits the retrieved actors by type id"""
    vehicles = []
    traffic_lights = []
    speed_limits = []
    walkers = []

    for actor_with_transform in actors_with_transforms:
        actor = actor_with_transform[0]
        if 'vehicle' in actor.type_id:
            vehicles.append(actor_with_transform)
        elif 'traffic_light' in actor.type_id:
            traffic_lights.append(actor_with_transform)
        elif 'speed_limit' in actor.type_id:
            speed_limits.append(actor_with_transform)
        elif 'walker.pedestrian' in actor.type_id:
            walkers.append(actor_with_transform)

    return (vehicles, traffic_lights, speed_limits, walkers)



class BirdViewOrigin(object):
    def __init__(self, core: Core):
        self.core = core
        self.world = core.world
        
        pygame.init()
        pygame.font.init()
        self.display = pygame.display.set_mode(
            (1280, 720),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        self.show_triggers = False

        self.map_image = bv.MapImage(
            carla_world=core.world,
            carla_map=core.town_map,
            pixels_per_meter=bv.PIXELS_PER_METER,
            show_triggers=self.show_triggers,
            show_connections=False,
            show_spawn_points=False)


        self.surface_size = self.map_image.big_map_surface.get_width()

        self.actors_surface = pygame.Surface((self.map_image.surface.get_width(), self.map_image.surface.get_height()))
        self.actors_surface.set_colorkey(bv.COLOR_BLACK)

        self.result_surface = pygame.Surface((self.surface_size, self.surface_size)).convert()
        self.result_surface.set_colorkey(bv.COLOR_BLACK)

        self.vehicle_id_surface = pygame.Surface((self.surface_size, self.surface_size)).convert()
        self.vehicle_id_surface.set_colorkey(bv.COLOR_BLACK)

        self.traffic_light_surfaces = TrafficLightSurfaces()

        hud_dim = (1280, 720)
        self.hud_dim = hud_dim
        self.border_round_surface = pygame.Surface(hud_dim, pygame.SRCALPHA).convert()
        self.border_round_surface.set_colorkey(bv.COLOR_WHITE)
        self.border_round_surface.fill(bv.COLOR_BLACK)

        # Used for Hero Mode, draws the map contained in a circle with white border
        center_offset = (int(hud_dim[0] / 2), int(hud_dim[1] / 2))
        pygame.draw.circle(self.border_round_surface, bv.COLOR_ALUMINIUM_1, center_offset, int(hud_dim[1] / 2))
        pygame.draw.circle(self.border_round_surface, bv.COLOR_WHITE, center_offset, int((hud_dim[1] - 8) / 2))


        self.original_surface_size = min(*hud_dim)
        scaled_original_size = self.original_surface_size * (1.0 / 0.9)
        self.hero_surface = pygame.Surface((scaled_original_size, scaled_original_size)).convert()
        self.scaled_original_size = scaled_original_size

        self.game_clock = pygame.time.Clock()


    def run_step(self, agent: BaseAgent):
        self.game_clock.tick_busy_loop(0)

        self.display.fill(bv.COLOR_ALUMINIUM_4)

        tick1 = time.time()

        actors = self.world.get_actors()
        actors_with_transforms = [(actor, actor.get_transform()) for actor in actors]
        hero_actor = agent.vehicle
        hero_transform = agent.get_transform()
        self.hero_actor = hero_actor
        self.hero_transform = hero_transform

        self.result_surface.fill(bv.COLOR_BLACK)

        vehicles, traffic_lights, speed_limits, walkers = _split_actors(actors_with_transforms)


        self.actors_surface.fill(bv.COLOR_BLACK)
        self.render_actors(self.actors_surface,
            vehicles, traffic_lights, speed_limits, walkers)

        surfaces = ((self.map_image.surface, (0, 0)),
                    (self.actors_surface, (0, 0)),
                    (self.vehicle_id_surface, (0, 0)),
                    )

        angle = 0.0 if hero_actor is None else hero_transform.rotation.yaw + 90.0
        self.traffic_light_surfaces.rotozoom(-angle, self.map_image.scale)



        hero_location_screen = self.map_image.world_to_pixel(self.hero_transform.location)
        hero_front = self.hero_transform.get_forward_vector()
        translation_offset = (hero_location_screen[0] - self.hero_surface.get_width() / 2 + hero_front.x * bv.PIXELS_AHEAD_VEHICLE,
                                (hero_location_screen[1] - self.hero_surface.get_height() / 2 + hero_front.y * bv.PIXELS_AHEAD_VEHICLE))

        # Apply clipping rect
        clipping_rect = pygame.Rect(translation_offset[0],
                                    translation_offset[1],
                                    self.hero_surface.get_width(),
                                    self.hero_surface.get_height())
        self.clip_surfaces(clipping_rect)

        Util.blits(self.result_surface, surfaces)

        self.border_round_surface.set_clip(clipping_rect)

        self.hero_surface.fill(bv.COLOR_ALUMINIUM_4)
        self.hero_surface.blit(self.result_surface, (-translation_offset[0],
                                                        -translation_offset[1]))

        rotated_result_surface = pygame.transform.rotozoom(self.hero_surface, angle, 0.9).convert()

        center = (self.display.get_width() / 2, self.display.get_height() / 2)
        rotation_pivot = rotated_result_surface.get_rect(center=center)

        tick2 = time.time()
        print('time social: ', tick2 - tick1)

        # self.display.blit(rotated_result_surface, rotation_pivot)

        # self.display.blit(self.border_round_surface, (0, 0))


        clean_surface = pygame.Surface(self.hud_dim).convert()
        clean_surface.blit(rotated_result_surface, rotation_pivot)
        clean_surface.blit(self.border_round_surface, (0,0))
        self.display.blit(clean_surface, (0, 0))

        # imgdata = pygame.surfarray.array3d(self.hero_surface)
        print('scaled_original_size: ', self.scaled_original_size)
        print()

        # import cv2
        # cv2.imshow('haha', imgdata)
        # cv2.waitKey(0)
        # print(imgdata.shape)

        pygame.display.flip()

        pass

    def render_actors(self, surface, vehicles, traffic_lights, speed_limits, walkers):
        """Renders all the actors"""
        # Static actors
        self._render_traffic_lights(surface, [tl[0] for tl in traffic_lights], self.map_image.world_to_pixel)
        self._render_speed_limits(surface, [sl[0] for sl in speed_limits], self.map_image.world_to_pixel,
                                  self.map_image.world_to_pixel_width)

        # Dynamic actors
        self._render_vehicles(surface, vehicles, self.map_image.world_to_pixel)
        self._render_walkers(surface, walkers, self.map_image.world_to_pixel)




    def _render_traffic_lights(self, surface, list_tl, world_to_pixel):
        """Renders the traffic lights and shows its triggers and bounding boxes if flags are enabled"""
        self.affected_traffic_light = None

        for tl in list_tl:
            world_pos = tl.get_location()
            pos = world_to_pixel(world_pos)

            if self.show_triggers:
                corners = Util.get_bounding_box(tl)
                corners = [world_to_pixel(p) for p in corners]
                pygame.draw.lines(surface, bv.COLOR_BUTTER_1, True, corners, 2)

            if self.hero_actor is not None:
                corners = Util.get_bounding_box(tl)
                corners = [world_to_pixel(p) for p in corners]
                tl_t = tl.get_transform()

                transformed_tv = tl_t.transform(tl.trigger_volume.location)
                hero_location = self.hero_actor.get_location()
                d = hero_location.distance(transformed_tv)
                s = Util.length(tl.trigger_volume.extent) + Util.length(self.hero_actor.bounding_box.extent)
                if (d <= s):
                    # Highlight traffic light
                    self.affected_traffic_light = tl
                    srf = self.traffic_light_surfaces.surfaces['h']
                    surface.blit(srf, srf.get_rect(center=pos))

            srf = self.traffic_light_surfaces.surfaces[tl.state]
            surface.blit(srf, srf.get_rect(center=pos))

    def _render_speed_limits(self, surface, list_sl, world_to_pixel, world_to_pixel_width):
        """Renders the speed limits by drawing two concentric circles (outer is red and inner white) and a speed limit text"""

        font_size = world_to_pixel_width(2)
        radius = world_to_pixel_width(2)
        font = pygame.font.SysFont('Arial', font_size)

        for sl in list_sl:

            x, y = world_to_pixel(sl.get_location())

            # Render speed limit concentric circles
            white_circle_radius = int(radius * 0.75)

            pygame.draw.circle(surface, bv.COLOR_SCARLET_RED_1, (x, y), radius)
            pygame.draw.circle(surface, bv.COLOR_ALUMINIUM_0, (x, y), white_circle_radius)

            limit = sl.type_id.split('.')[2]
            font_surface = font.render(limit, True, bv.COLOR_ALUMINIUM_5)

            if self.show_triggers:
                corners = Util.get_bounding_box(sl)
                corners = [world_to_pixel(p) for p in corners]
                pygame.draw.lines(surface, bv.COLOR_PLUM_2, True, corners, 2)

            # Blit
            if self.hero_actor is not None:
                # In hero mode, Rotate font surface with respect to hero vehicle front
                angle = -self.hero_transform.rotation.yaw - 90.0
                font_surface = pygame.transform.rotate(font_surface, angle)
                offset = font_surface.get_rect(center=(x, y))
                surface.blit(font_surface, offset)

            else:
                # In map mode, there is no need to rotate the text of the speed limit
                surface.blit(font_surface, (x - radius / 2, y - radius / 2))

    def _render_walkers(self, surface, list_w, world_to_pixel):
        """Renders the walkers' bounding boxes"""
        for w in list_w:
            color = bv.COLOR_PLUM_0

            # Compute bounding box points
            bb = w[0].bounding_box.extent
            corners = [
                carla.Location(x=-bb.x, y=-bb.y),
                carla.Location(x=bb.x, y=-bb.y),
                carla.Location(x=bb.x, y=bb.y),
                carla.Location(x=-bb.x, y=bb.y)]

            w[1].transform(corners)
            corners = [world_to_pixel(p) for p in corners]
            pygame.draw.polygon(surface, color, corners)

    def _render_vehicles(self, surface, list_v, world_to_pixel):
        """Renders the vehicles' bounding boxes"""
        for v in list_v:
            color = bv.COLOR_SKY_BLUE_0
            if int(v[0].attributes['number_of_wheels']) == 2:
                color = bv.COLOR_CHOCOLATE_1
            if v[0].attributes['role_name'] == 'hero':
                color = bv.COLOR_CHAMELEON_0
            # Compute bounding box points
            bb = v[0].bounding_box.extent
            corners = [carla.Location(x=-bb.x, y=-bb.y),
                       carla.Location(x=bb.x - 0.8, y=-bb.y),
                       carla.Location(x=bb.x, y=0),
                       carla.Location(x=bb.x - 0.8, y=bb.y),
                       carla.Location(x=-bb.x, y=bb.y),
                       carla.Location(x=-bb.x, y=-bb.y)
                       ]
            v[1].transform(corners)
            corners = [world_to_pixel(p) for p in corners]
            # pygame.draw.lines(surface, color, False, corners, int(np.ceil(4.0 * self.map_image.scale)))
            pygame.draw.polygon(surface, color, corners)

    def clip_surfaces(self, clipping_rect):
        """Used to improve perfomance. Clips the surfaces in order to render only the part of the surfaces that are going to be visible"""
        self.actors_surface.set_clip(clipping_rect)
        self.vehicle_id_surface.set_clip(clipping_rect)
        self.result_surface.set_clip(clipping_rect)



class TrafficLightSurfaces(object):
    """Holds the surfaces (scaled and rotated) for painting traffic lights"""

    def __init__(self):
        def make_surface(tl):
            """Draws a traffic light, which is composed of a dark background surface with 3 circles that indicate its color depending on the state"""
            w = 40
            surface = pygame.Surface((w, 3 * w), pygame.SRCALPHA)
            surface.fill(bv.COLOR_ALUMINIUM_5 if tl != 'h' else bv.COLOR_ORANGE_2)
            if tl != 'h':
                hw = int(w / 2)
                off = bv.COLOR_ALUMINIUM_4
                red = bv.COLOR_SCARLET_RED_0
                yellow = bv.COLOR_BUTTER_0
                green = bv.COLOR_CHAMELEON_0

                # Draws the corresponding color if is on, otherwise it will be gray if its off
                pygame.draw.circle(surface, red if tl == tls.Red else off, (hw, hw), int(0.4 * w))
                pygame.draw.circle(surface, yellow if tl == tls.Yellow else off, (hw, w + hw), int(0.4 * w))
                pygame.draw.circle(surface, green if tl == tls.Green else off, (hw, 2 * w + hw), int(0.4 * w))

            return pygame.transform.smoothscale(surface, (15, 45) if tl != 'h' else (19, 49))

        self._original_surfaces = {
            'h': make_surface('h'),
            tls.Red: make_surface(tls.Red),
            tls.Yellow: make_surface(tls.Yellow),
            tls.Green: make_surface(tls.Green),
            tls.Off: make_surface(tls.Off),
            tls.Unknown: make_surface(tls.Unknown)
        }
        self.surfaces = dict(self._original_surfaces)

    def rotozoom(self, angle, scale):
        """Rotates and scales the traffic light surface"""
        for key, surface in self._original_surfaces.items():
            self.surfaces[key] = pygame.transform.rotozoom(surface, angle, scale)




class BirdViewSimple(BirdViewOrigin):
    '''
        embeded into env_sa
    '''

    def __init__(self, core: Core):
        self.core = core
        self.world = core.world 

        hud_dim = (720, 720)
        hud_dim = (1280, 1280)
        self.hud_dim = hud_dim

        self.inited = False
        if not self.inited:
            pygame.init()
            pygame.font.init()
            self.display = pygame.display.set_mode(hud_dim, pygame.HWSURFACE | pygame.DOUBLEBUF)
            self.inited = True


        self.show_triggers = False
        self.map_image = bv.MapImage(
            carla_world=core.world,
            carla_map=core.town_map,
            pixels_per_meter=bv.PIXELS_PER_METER,
            show_triggers=self.show_triggers,
            show_connections=False,
            show_spawn_points=False)


        self.surface_size = self.map_image.big_map_surface.get_width()

        self.actors_surface = pygame.Surface((self.map_image.surface.get_width(), self.map_image.surface.get_height()))
        self.actors_surface.set_colorkey(bv.COLOR_BLACK)

        self.waypoints_surface = pygame.Surface((self.map_image.surface.get_width(), self.map_image.surface.get_height()))
        self.waypoints_surface.set_colorkey(bv.COLOR_BLACK)

        self.result_surface = pygame.Surface((self.surface_size, self.surface_size)).convert()
        self.result_surface.set_colorkey(bv.COLOR_BLACK)

        self.vehicle_id_surface = pygame.Surface((self.surface_size, self.surface_size)).convert()
        self.vehicle_id_surface.set_colorkey(bv.COLOR_BLACK)

        self.border_round_surface = pygame.Surface(hud_dim, pygame.SRCALPHA).convert()
        self.border_round_surface.set_colorkey(bv.COLOR_WHITE)
        self.border_round_surface.fill(bv.COLOR_BLACK)

        self.clean_surface = pygame.Surface(hud_dim).convert()
        self.clean_surface.set_colorkey(bv.COLOR_BLACK)
        self.clean_surface.fill(bv.COLOR_BLACK)

        # Used for Hero Mode, draws the map contained in a circle with white border
        center_offset = (int(hud_dim[0] / 2), int(hud_dim[1] / 2))
        pygame.draw.circle(self.border_round_surface, bv.COLOR_ALUMINIUM_1, center_offset, int(hud_dim[1] / 2))
        pygame.draw.circle(self.border_round_surface, bv.COLOR_WHITE, center_offset, int((hud_dim[1] - 8) / 2))


        self.original_surface_size = min(*hud_dim)
        scaled_original_size = self.original_surface_size * (1.0 / 0.9)
        self.hero_surface = pygame.Surface((scaled_original_size, scaled_original_size)).convert()
        self.scaled_original_size = scaled_original_size

        self.game_clock = pygame.time.Clock()



    def run_step(self, agent: BaseAgent):
        self.game_clock.tick_busy_loop(0)

        actors = self.world.get_actors()
        actors_with_transforms = [(actor, actor.get_transform()) for actor in actors]
        hero_actor = agent.vehicle
        hero_transform = agent.get_transform()
        self.hero_actor = hero_actor
        self.hero_transform = hero_transform

        self.result_surface.fill(bv.COLOR_BLACK)

        vehicles, _, _, walkers = _split_actors(actors_with_transforms)


        self.actors_surface.fill(bv.COLOR_BLACK)
        self.render_actors(self.actors_surface, vehicles, walkers)

        waypoints, _ = agent.global_path.remaining_waypoints(agent.get_transform())
        waypoints = waypoints[:150:15]   ## TODO change
        self.waypoints_surface.fill(bv.COLOR_BLACK)
        self.render_waypoints(
            self.waypoints_surface, 
            waypoints,
            self.map_image.world_to_pixel)


        surfaces = ((self.map_image.surface, (0, 0)),
                    (self.waypoints_surface, (0, 0)),
                    (self.actors_surface, (0, 0)),
                    # (self.vehicle_id_surface, (0, 0)),
                    )

        angle = 0.0 if hero_actor is None else hero_transform.rotation.yaw + 90.0

        hero_location_screen = self.map_image.world_to_pixel(self.hero_transform.location)
        hero_front = self.hero_transform.get_forward_vector()
        translation_offset = (hero_location_screen[0] - self.hero_surface.get_width() / 2 + hero_front.x * bv.PIXELS_AHEAD_VEHICLE,
                                (hero_location_screen[1] - self.hero_surface.get_height() / 2 + hero_front.y * bv.PIXELS_AHEAD_VEHICLE))

        # Apply clipping rect
        clipping_rect = pygame.Rect(translation_offset[0],
                                    translation_offset[1],
                                    self.hero_surface.get_width(),
                                    self.hero_surface.get_height())
        self.clip_surfaces(clipping_rect)

        Util.blits(self.result_surface, surfaces)

        self.border_round_surface.set_clip(clipping_rect)

        self.hero_surface.fill(bv.COLOR_ALUMINIUM_4)
        self.hero_surface.blit(self.result_surface, (-translation_offset[0],
                                                        -translation_offset[1]))

        rotated_result_surface = pygame.transform.rotozoom(self.hero_surface, angle, 0.9).convert()

        center = (self.display.get_width() / 2, self.display.get_height() / 2)
        rotation_pivot = rotated_result_surface.get_rect(center=center)


        ### result
        self.clean_surface.fill(bv.COLOR_BLACK)
        self.clean_surface.blit(rotated_result_surface, rotation_pivot)
        self.clean_surface.blit(self.border_round_surface, (0,0))


        imgdata = pygame.surfarray.array3d(self.clean_surface)

        center_x, center_y = int(self.hud_dim[0]/2), int(self.hud_dim[1]/2)
        x_r = int(np.floor(self.hud_dim[0]/np.sqrt(8)))
        y_r = int(np.floor(self.hud_dim[1]/np.sqrt(8)))
        imgdata = imgdata[center_x-x_r:center_x+x_r, center_y-y_r:center_y+y_r]  ## TODO debug

        # imgdata = cv2.cvtColor(imgdata, cv2.COLOR_RGB2BGR)
        # cv2.imshow('haha', imgdata)
        # cv2.waitKey(1)
        return imgdata


    def render_actors(self, surface, vehicles, walkers):
        self._render_vehicles(surface, vehicles, self.map_image.world_to_pixel)
        self._render_walkers(surface, walkers, self.map_image.world_to_pixel)


    def _render_vehicles(self, surface, list_v, world_to_pixel):
        """Renders the vehicles' bounding boxes"""
        for v in list_v:
            color = bv.COLOR_SKY_BLUE_0
            if int(v[0].attributes['number_of_wheels']) == 2:
                color = bv.COLOR_CHOCOLATE_1
            if Role.loads(v[0].attributes['role_name']).atype == 'learnable':   # TODO
            # if False:
                color = bv.COLOR_CHAMELEON_0
            # Compute bounding box points
            bb = v[0].bounding_box.extent
            corners = [carla.Location(x=-bb.x, y=-bb.y),
                       carla.Location(x=bb.x - 0.8, y=-bb.y),
                       carla.Location(x=bb.x, y=0),
                       carla.Location(x=bb.x - 0.8, y=bb.y),
                       carla.Location(x=-bb.x, y=bb.y),
                       carla.Location(x=-bb.x, y=-bb.y)
                       ]
            v[1].transform(corners)
            corners = [world_to_pixel(p) for p in corners]
            # pygame.draw.lines(surface, color, False, corners, int(np.ceil(4.0 * self.map_image.scale)))
            pygame.draw.polygon(surface, color, corners)


    def render_waypoints(self, surface, waypoints, world_to_pixel):
        # if self.red_light:
        if False:
        # purple
            color = pygame.Color(np.floor(0.5*255), 0, np.floor(0.5*255))
        else:
        # blue
            color = pygame.Color(0,0,255)
        corners = []
        for p in waypoints:
            c = p.transform.location
            corners.append(carla.Location(x=c.x,y=c.y))
        corners = [world_to_pixel(p) for p in corners]
        pygame.draw.lines(surface, color, False, corners, 20)





#######################################################################
###  My Own  ##########################################################
#######################################################################



from typing import List



from .. import basic
from ..augment import ActorVertices, cvt
from ..agents import BaseAgent, BaseAgentObstacle



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
    

class PerceptionBirdEyeView(object):
    '''
        agent centric
    '''

    def __init__(self, core, resolution, range: carla.Location):
        self.core = core
        self.resolution, self.range = resolution, range
        self.radius = np.hypot(range.x, range.y)
        self.height, self.width = int(range.y / resolution), int(range.x / resolution)   ## TODO
        self.min, self.max = np.array([0,0], dtype=np.int64), np.array([self.width, self.height], dtype=np.int64)
        self.valid = lambda p: (p >= self.min) & (p < self.max)
        self.x_array = np.arange(self.width).reshape(1,self.width).repeat(self.height, axis=0)
        self.y_array = np.arange(self.height).reshape(self.height,1).repeat(self.width, axis=1)


    def run_step(self, agent: BaseAgent, obstacles: List[BaseAgentObstacle]):
        '''
        
        Returns:
            shape is (4, height, width)
        '''

        states = {agent.id: agent.get_state() for agent in [agent] + obstacles}

        # vehicles = [agent.vehicle for agent in self.agents]
        mask_indexes = self.get_mask_indexes(agent, obstacles)

        ### route
        route_mask_index = self.get_route_mask_index(agent)
        route_map = np.zeros((self.height, self.width), dtype=np.float32)
        route_map[route_mask_index[:,0], route_mask_index[:,1]] = 1

        ### other
        ego_map = np.zeros((self.height, self.width), dtype=np.float32)
        obs_map = np.zeros((self.height, self.width), dtype=np.float32)
        vel_map = np.zeros((self.height, self.width), dtype=np.float32)
        yaw_map = np.zeros((self.height, self.width), dtype=np.float32)

        state = states[agent.id]
        now_theta, now_speed = state.theta, state.v

        # local_mask_indexes = self.visualizer.crop(global_mask_indexes, agent.vehicle)

        ego_index = mask_indexes[agent.id]
        ego_map[ego_index[:,0], ego_index[:,1]] = 1

        for key, local_index in mask_indexes.items():
            if key == agent.id:
                continue
            
            other_state = states[key]
            other_theta, other_speed = other_state.theta, other_state.v

            height_index, width_index = local_index[:,0], local_index[:,1]

            obs_map[height_index, width_index] = 1
            vel_map[height_index, width_index] = np.clip((other_speed-now_speed) / (2*agent.max_velocity), -0.5, 0.5) + 0.5
            yaw_map[height_index, width_index] = basic.pi2pi(other_theta - now_theta) / (2*np.pi) + 0.5


        obsv_map = np.stack([ego_map, obs_map, vel_map, yaw_map, route_map])


        cv2.imshow('haha', obs_map)
        cv2.imshow('gaga', route_map)
        cv2.waitKey(0)

        


    def get_mask_indexes(self, agent: BaseAgent, obstacles: List[BaseAgentObstacle]):
        mask_indexes = dict()
        for vehicle in [agent] + obstacles:
            dist = np.hypot(vehicle.get_state().x - agent.get_state().x, vehicle.get_state().y - agent.get_state().y)
            if dist > self.radius:
                continue

            boundary = self._draw_vehicle(vehicle, agent)
            mask_index = self._mask_index(boundary)
            mask_indexes[vehicle.id] = mask_index
        
        return mask_indexes


    def _draw_vehicle(self, obstacle, agent, expand=carla.Vector2D(0,0)):
        vertices = d2_local(obstacle, agent, expand)
        vertices[:,1] *= -1
        boundary = vertices / self.resolution + np.array([[self.width/2, self.height/2]])
        return boundary

    def _mask_index(self, boundary):
        """
        Args:
            boundary: counter-clockwise
        
        Returns:
            
        """

        num = boundary.shape[0]
        condition = True
        for i in range(num):
            p1, p2 = boundary[i], boundary[(i+1)%num]
            x1, y1 = p1[0], p1[1]
            x2, y2 = p2[0], p2[1]
            condition &= ((y2-y1)*self.x_array - (x2-x1)*self.y_array -(y2-y1)*x1 + (x2-x1)*y1) > 0
        mask_index = np.argwhere(condition)
        ### Note that the shape of mask_index is (N,2), where (N,0) represents y/height axis. 
        return mask_index



    def get_route_mask_index(self, agent: BaseAgent):
        state0 = agent.get_state()
        t = agent.get_transform()
        global_path = agent.global_path

        waypoints, _ = global_path.remaining_waypoints(agent.get_transform())
        # waypoints = waypoints[:20]

        states = []
        for w in waypoints:
            if w.transform.location.distance(t.location) > self.radius:
                break
            state = cvt.CUAState.carla_transform(w.transform).world2local(state0)
            states.append(state)

        route_mask_index = []
        for i in range(len(states)-1):
            s1, s2 = states[i], states[i+1]

            dx, dy = s1.distance_xy(s2) /2, 2.0   # half lane width
            theta = np.arctan2(s2.y-s1.y, s2.x-s1.x)
            center_x, center_y = (s1.x + s2.x)*0.5, (s1.y + s2.y)*0.5

            l, n = np.array([np.cos(theta), np.sin(theta)]), np.array([np.cos(theta+np.pi/2), np.sin(theta+np.pi/2)])
            vertices = np.expand_dims(np.array([center_x, center_y]), axis=0).repeat(4, axis=0)

            vertices[0] +=  l*dx + n*dy
            vertices[1] += -l*dx + n*dy
            vertices[2] += -l*dx - n*dy
            vertices[3] +=  l*dx - n*dy

            vertices[:,1] *= -1
            boundary = vertices / self.resolution + np.array([[self.width/2, self.height/2]])
            mask_index = self._mask_index(boundary)

            route_mask_index.append(mask_index)

        route_mask_index = np.vstack(route_mask_index)
        return route_mask_index




class PerceptionBirdEyeViewMultiAgent(object):
    '''
        For multi-agent
        global and local
    '''
    def __init__(self, resolution, g_range, l_range):
        """
        
        
        Args:
            resolution: m/pix
        
        Returns:
            
        """
        self.resolution, self.g_range, self.l_range = resolution, g_range, l_range
        self.g_height, self.g_width = int(g_range.y / resolution), int(g_range.x / resolution)
        self.l_height, self.l_width = int(l_range.y / resolution), int(l_range.x / resolution)
        self.min, self.l_max = np.array([0,0], dtype=np.int64), np.array([self.l_width, self.l_height], dtype=np.int64)
        self.l_valid = lambda p: (p >= self.min) & (p < self.l_max)
        self.x_array = np.arange(self.g_width).reshape(1,self.g_width).repeat(self.g_height, axis=0)
        self.y_array = np.arange(self.g_height).reshape(self.g_height,1).repeat(self.g_width, axis=1)
    
    
    def global_mask_indexes(self, vehicles):
        mask_indexes = dict()
        for vehicle in vehicles:
            boundary = self._draw_vehicle(vehicle)
            mask_index = self._mask_index(boundary)
            mask_indexes[vehicle.id] = mask_index
        return mask_indexes

    def crop(self, global_mask_indexes, vehicle):
        bbx = vehicle.bounding_box.extent
        expand = carla.Vector2D((self.l_range.y-self.resolution)/2-bbx.y, (self.l_range.x-self.resolution)/2-bbx.x)
        boundary = self._draw_vehicle(vehicle, expand)

        theta_vehicle = np.deg2rad(vehicle.get_transform().rotation.yaw)
        htm = basic.HomogeneousMatrixInverse2D.xytheta([boundary[0][0], boundary[0][1], np.pi/2 - theta_vehicle])

        local_mask_indexes = dict()
        for key, global_index in global_mask_indexes.items():
            local_index = np.dot(htm, np.vstack((global_index.T[::-1,:], np.ones((1,global_index.shape[0])))))[:2].T
            local_index = np.round(local_index).astype(np.int64)
            local_mask = self.l_valid(local_index)
            local_index = local_index[local_mask[:,0] & local_mask[:,1]]
            local_mask_indexes[key] = local_index[:,::-1]
        
        return local_mask_indexes


    def _mask_index(self, boundary):
        """
        
        
        Args:
            boundary: counter-clockwise
        
        Returns:
            
        """

        num = boundary.shape[0]
        condition = True
        for i in range(num):
            p1, p2 = boundary[i], boundary[(i+1)%num]
            x1, y1 = p1[0], p1[1]
            x2, y2 = p2[0], p2[1]
            condition &= ((y2-y1)*self.x_array - (x2-x1)*self.y_array -(y2-y1)*x1 + (x2-x1)*y1) > 0
        mask_index = np.argwhere(condition)
        ### Note that the shape of mask_index is (N,2), where (N,0) represents y/height axis. 
        return mask_index
    
    def _draw_vehicle(self, vehicle, expand=carla.Vector2D(0,0)):
        vertices = ActorVertices.d2(vehicle, expand)
        vertices[:,1] *= -1
        boundary = vertices / self.resolution + np.array([[self.g_width/2, self.g_height/2]])
        return boundary