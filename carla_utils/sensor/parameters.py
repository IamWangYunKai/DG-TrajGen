
import numpy as np

from ..basic import np_dot
from ..basic import RotationMatrix, HomogeneousMatrix, HomogeneousMatrixInverse, Reverse



def intrinsicMatrix(fx, fy, u0, v0):
    K = np.array([  [fx, 0, u0],
                    [0, fy, v0],
                    [0,  0,  1] ])
    return K


class IntrinsicParams(object):
    def __init__(self, sensor):
        '''
        https://github.com/carla-simulator/carla/issues/56
        Args:
            sensor: carla.Sensor
        '''
        image_size_x = float(sensor.attributes['image_size_x'])
        image_size_y = float(sensor.attributes['image_size_y'])
        fov = eval(sensor.attributes['fov'])
        f = image_size_x /(2 * np.tan(fov * np.pi / 360))

        # [px]
        fx = f
        fy = f
        u0 = image_size_x / 2
        v0 = image_size_y / 2

        self.K = intrinsicMatrix(fx, fy, u0, v0)
        

class ExtrinsicParams(object):
    def __init__(self, sensor):
        '''
        Args:
            sensor: carla.Sensor
        '''

        # camera coordinate in imu coordinate
        transform = sensor.get_transform()

        # [m]
        x = transform.location.x
        y = transform.location.y
        z = transform.location.z

        # [rad]
        roll = np.deg2rad(transform.rotation.roll)
        pitch = np.deg2rad(transform.rotation.pitch)
        yaw = np.deg2rad(transform.rotation.yaw)

        # (coordinate) t: camera in imu, R: camera to imu
        self.t = np.array([[x, y, z]]).T
        self.R = RotationMatrix.rpy(roll, pitch, yaw)


class CameraParams(object):
    I = np_dot(Reverse.x(), Reverse.y(), RotationMatrix.ypr(np.pi/2, 0, -np.pi/2))
    # I = np_dot(Reverse.x(), RotationMatrix.ypr(np.pi/2, 0, -np.pi/2))
    def __init__(self, sensor):
        '''
        https://github.com/carla-simulator/carla/issues/553
        Args:
            sensor: carla.Sensor
            intrinsic_params: IntrinsicParams
            extrinsic_params: ExtrinsicParams
        '''
        intrinsic_params = IntrinsicParams(sensor)
        extrinsic_params = ExtrinsicParams(sensor)
        self.K = intrinsic_params.K
        self.t = extrinsic_params.t
        self.R = extrinsic_params.R

        self.K_augment = np.hstack((self.K, np.zeros((3,1))))
        self.T_img_imu = np_dot(HomogeneousMatrix.rotation(CameraParams.I), HomogeneousMatrixInverse.rotation_translation(self.R, self.t))
        