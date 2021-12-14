'''
    open3d visualizer
'''

import carla_utils as cu
from carla_utils import carla

import numpy as np
import open3d
import open3d
import os
import glob

from ..system import Clock, parse_yaml_file_unsafe
from ..basic import HomogeneousMatrix

from ..augment import ActorVertices
from ..world_map import connect_to_server

from .tools import default_argparser


def calculate_vis_bounding_box(vehicle : carla.Vehicle):
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


def get_fixed_boundary(color_open3d : np.ndarray):
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



def calculate_perception_range(vehicle, line_set, perception_range):
    current_transform = vehicle.get_transform()
    x, y, z = current_transform.location.x, current_transform.location.y, current_transform.location.z

    resolution = np.deg2rad(2)
    rads = np.linspace(-np.pi, np.pi, int(np.pi / resolution))
    points, lines = [], []
    for i, rad in enumerate(rads):
        point = [x + perception_range*np.cos(rad), y + perception_range*np.sin(rad), z]
        points.append(point)
        lines.append([i, (i+1)%len(rads)])
    
    points = np.array(points, dtype=np.float64)
    lines = np.array(lines)

    color = vehicle.attributes.get('color', '190,190,190')
    color = np.array(eval(color)).astype(np.float64) / 255
    colors = np.expand_dims(color, axis=0).repeat(len(lines), axis=0)

    line_set.points = open3d.utility.Vector3dVector(points)
    line_set.lines = open3d.utility.Vector2iVector(lines)
    line_set.colors = open3d.utility.Vector3dVector(colors)
    return line_set


def get_road_geometry_center(global_paths):
    color = np.array([100, 100, 100], np.float64) / 255

    line_sets = []
    for global_path in global_paths:
        centers = np.stack([global_path.x, global_path.y]).T
        lines = np.vstack([np.arange(0, len(global_path)-1), np.arange(1, len(global_path))]).T
        colors = np.expand_dims(color, axis=0).repeat(len(lines), axis=0)

        line_set = open3d.geometry.LineSet()
        line_set.points = open3d.utility.Vector3dVector(np.hstack((centers, np.zeros((len(global_path),1)))))
        line_set.lines = open3d.utility.Vector2iVector(lines)
        line_set.colors = open3d.utility.Vector3dVector(colors)
        line_sets.append(line_set)
    return line_sets

def get_road_geometry_side(town_map):
    color = np.array([150, 150, 150], np.float64) / 255

    files = glob.glob(os.path.split(os.path.abspath(__file__))[0] + '/../utils/global_path/*')
    global_paths = [cu.GlobalPath.read_from_disk(town_map, i) for i in files]

    line_sets = []
    for global_path in global_paths:
        lane_widths = np.array([w.lane_width for w in global_path.carla_waypoints]).reshape(len(global_path),1)
        thetas = np.array(global_path.theta).reshape(len(global_path),1)
        directions =  np.hstack([np.cos(thetas + np.pi/2), np.sin(thetas + np.pi/2)])
        centers = np.stack([global_path.x, global_path.y]).T
        lines = np.vstack([np.arange(0, len(global_path)-1), np.arange(1, len(global_path))]).T
        colors = np.expand_dims(color, axis=0).repeat(len(lines), axis=0)

        ### left
        line_set = open3d.geometry.LineSet()
        left = centers + lane_widths/2 * directions
        
        line_set.points = open3d.utility.Vector3dVector(np.hstack((left, np.zeros((len(global_path),1)))))
        line_set.lines = open3d.utility.Vector2iVector(lines)
        line_set.colors = open3d.utility.Vector3dVector(colors)
        line_sets.append(line_set)

        ### right
        line_set = open3d.geometry.LineSet()
        right = centers - lane_widths/2 * directions
        
        line_set.points = open3d.utility.Vector3dVector(np.hstack((right, np.zeros((len(global_path),1)))))
        line_set.lines = open3d.utility.Vector2iVector(lines)
        line_set.colors = open3d.utility.Vector3dVector(colors)
        line_sets.append(line_set)
    return line_sets



class VehiclesVisualizer(object):
    def __init__(self, config, view_pose=None):
        '''parameter'''
        self.config = config
        host, port, timeout, map_name = config.host, config.port, config.timeout, config.map_name
        self.client, self.world, self.town_map = connect_to_server(host, port, timeout, map_name)
        self.clock = Clock(10)
        self.max_vehicles = config.max_vehicles  ## max number

        self.window_name = 'Vehicles Visualisation Example' + '   ' + host + ':' + str(port)
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
        self.vis.add_geometry(get_fixed_boundary(self.background_color))

        self.bounding_boxs = [open3d.geometry.LineSet() for _ in range(self.max_vehicles)]
        [self.vis.add_geometry(bounding_box) for bounding_box in self.bounding_boxs]


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
    print(__doc__)
    
    import os
    from os.path import join

    try:
        config = parse_yaml_file_unsafe('./config/carla.yaml')
    except FileNotFoundError:
        print('[vehicle_visualizer] use default config.')
        file_dir = os.path.dirname(__file__)
        config = parse_yaml_file_unsafe(join(file_dir, './default_carla.yaml'))
    args = default_argparser().parse_args()
    config.update(args)
    
    vehicles_visualizer = VehiclesVisualizer(config, )
    try:
        vehicles_visualizer.run()
    except KeyboardInterrupt:
        print('canceled by user')

