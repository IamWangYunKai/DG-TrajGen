
import rospy

from ..world_map import Core
from .pub_sub import ROSPublish


class PublishWrapper(object):
    pub_dict = dict()
    def __init__(self, config, node_name='carla_env'):
        core: Core = config.core

        rospy.init_node('{}_{}_{}'.format(node_name, core.host.replace('.', '_'), str(core.port)), disable_signals=True)
        self.global_frame_id = 'map'
        self.ros_pubish = ROSPublish(self.pub_dict)


    def run_once(self, *args, **kwargs):
        return

    def run_step(self, *args, **kwargs):
        return


    def kill(self):
        rospy.signal_shutdown('[ROS] kill myself!')
