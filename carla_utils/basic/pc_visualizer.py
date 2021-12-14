
import numpy as np
import open3d

from .coordinate_transformation import HomogeneousMatrix

class PointCloud3DVisualizer(object):
    def __init__(self, view_pose=None):
        self.title = "Velodyne Visualisation Example"
        self.vis = None
        self.view_pose = [0, 3, 32, np.pi, -np.pi/6, -np.pi/2] if view_pose is None else view_pose

    def run_step(self, pointcloud):
        '''
        Args:
            pointcloud: numpy.ndarray (x,y,z,I) × N
        '''
        if self.vis is None:
            self.vis = open3d.visualization.Visualizer()
            self.vis.create_window(window_name=self.title, width=800, height=800)
            self.pcd = open3d.geometry.PointCloud()
            # initialise the geometry pre loop
            self.pcd.points = open3d.utility.Vector3dVector(pointcloud[:3].transpose().astype(np.float64))
            self.pcd.colors = open3d.utility.Vector3dVector(np.tile(pointcloud[3:].transpose(), (1, 3)).astype(np.float64))
            # Rotate pointcloud to align displayed coordinate frame colouring
            self.pcd.transform(HomogeneousMatrix.xyzrpy([0, 0, 0, np.pi, 0, -np.pi / 2]))
            self.vis.add_geometry(self.pcd)
            render_option = self.vis.get_render_option()
            render_option.background_color = np.array([0.1529, 0.1569, 0.1333], np.float32)
            render_option.point_color_option = open3d.visualization.PointColorOption.ZCoordinate
            coordinate_frame = open3d.geometry.TriangleMesh.create_coordinate_frame()
            self.vis.add_geometry(coordinate_frame)
            view_control = self.vis.get_view_control()
            params = view_control.convert_to_pinhole_camera_parameters()
            params.extrinsic = HomogeneousMatrix.xyzrpy(self.view_pose)
            view_control.convert_from_pinhole_camera_parameters(params)

        self.pcd.points = open3d.utility.Vector3dVector(pointcloud[:3].transpose().astype(np.float64))
        self.pcd.colors = open3d.utility.Vector3dVector(np.tile(pointcloud[3:].transpose(), (1, 3)).astype(np.float64))
        self.vis.update_geometry()
        self.vis.poll_events()
        self.vis.update_renderer()
    
    def run(self, pointcloud):
        self.run_step(pointcloud)
        self.vis.run()
