
import carla_utils as cu
from carla_utils import carla

import numpy as np
import open3d
import os
import glob
import time

from ..system import Clock, parse_yaml_file_unsafe
from ..basic import HomogeneousMatrix
from ..augment import ActorVertices
from ..world_map import connect_to_server



def get_fixed_boundary(color_open3d: np.ndarray):
    max_x, max_y = 80, 45
    z = 0
    line_set = open3d.geometry.LineSet()
    points = np.array([ [max_x,max_y,z], [-max_x,max_y,z], [-max_x,-max_y,z], [max_x,-max_y,z] ]).astype(np.float64)
    lines = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
    colors = np.expand_dims(color_open3d, axis=0).repeat(len(lines), axis=0)

    line_set.points = open3d.utility.Vector3dVector(points)
    line_set.lines = open3d.utility.Vector2iVector(lines)
    line_set.colors = open3d.utility.Vector3dVector(colors)
    return line_set


def get_road_geometry(town_map):
    waypoints = town_map.generate_waypoints(0.1)
    points = []
    for w in waypoints:
        l = w.transform.location
        points.append([l.x, l.y, l.z-0.5])
    points = np.asarray(points)

    color = np.array([100, 100, 100], np.float64) / 255

    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points)

    colors = np.expand_dims(color, axis=0).repeat(points.shape[0], axis=0)
    pcd.colors = open3d.utility.Vector3dVector(colors)
    pcd.paint_uniform_color(color)
    return pcd



def calculate_vis_bounding_box(vehicle: carla.Vehicle):
    color = vehicle.attributes.get('color', '190,190,190')
    color = np.array(eval(color)).astype(np.float64) / 255

    line_set = open3d.geometry.LineSet()

    vertices, lines = ActorVertices.d2arrow(vehicle)
    vertices = np.hstack((vertices, np.zeros((vertices.shape[0],1))))
    colors = np.expand_dims(color, axis=0).repeat(len(lines), axis=0)

    line_set.points = open3d.utility.Vector3dVector(vertices)
    line_set.lines = open3d.utility.Vector2iVector(lines)
    line_set.colors = open3d.utility.Vector3dVector(colors)
    return line_set



class Visualizer(object):
    def __init__(self, config, view_pose=None):

        '''parameter'''
        self.config = config
        self.connect_to_server()
        self.clock = Clock(100)
        self.max_vehicles = 1000  ## max number

        self.window_name = 'Vehicles Visualisation Example' + '   ' + config.host + ':' + str(config.port)
        self.vis = open3d.visualization.Visualizer()
        self.vis.create_window(window_name=self.window_name, width=1000, height=1000, left=0, top=0)
        self.view_pose = [0, 0, 60, 0, 0, -np.pi/2] if view_pose is None else view_pose

        render_option = self.vis.get_render_option()
        self.background_color = np.array([0.1529, 0.1569, 0.1333], np.float32)
        render_option.background_color = self.background_color
        render_option.point_color_option = open3d.visualization.PointColorOption.ZCoordinate
        # coordinate_frame = open3d.geometry.TriangleMesh.create_coordinate_frame()
        # self.vis.add_geometry(coordinate_frame)
        view_control = self.vis.get_view_control()
        params = view_control.convert_to_pinhole_camera_parameters()
        params.extrinsic = HomogeneousMatrix.xyzrpy(self.view_pose)
        view_control.convert_from_pinhole_camera_parameters(params)

        '''add geometry'''
        # self.vis.add_geometry(get_fixed_boundary(self.background_color))

        self.bounding_boxs = [open3d.geometry.LineSet() for _ in range(self.max_vehicles)]
        [self.vis.add_geometry(bounding_box) for bounding_box in self.bounding_boxs]

        self.vis.add_geometry( get_road_geometry(self.town_map) )


    def connect_to_server(self):
        host, port, timeout, map_name = self.config.host, self.config.port, self.config.timeout, self.config.map_name
        _, self.world, self.town_map = connect_to_server(host, port, timeout, map_name)



    def run_step(self, vehicles):
        number_min = min(len(vehicles), self.max_vehicles)
        number_max = max(len(vehicles), self.max_vehicles)

        for i in range(number_min):
            vehicle, bounding_box = vehicles[i], self.bounding_boxs[i]
            new_bounding_box = calculate_vis_bounding_box(vehicle)
            bounding_box.points = new_bounding_box.points
            bounding_box.lines = new_bounding_box.lines
            bounding_box.colors = new_bounding_box.colors

        for i in range(number_min, number_max): self.bounding_boxs[i].clear()

        return
        
    def update_vis(self):
        self.vis.update_geometry()
        self.vis.poll_events()
        self.vis.update_renderer()

    
    def run(self):
        while True:
            self.clock.tick_begin()

            actors = self.world.get_actors()
            vehicles = actors.filter('*vehicle*')
            self.run_step(list(vehicles))
            self.update_vis()

            self.clock.tick_end()



if __name__ == "__main__":
    config = cu.basic.YamlConfig()
    args = cu.utils.default_argparser().parse_args()
    config.update(args)

    vehicles_visualizer = Visualizer(config, )
    try:
        vehicles_visualizer.run()
    except KeyboardInterrupt:
        print('canceled by user')

