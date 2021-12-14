import carla_utils as cu
from carla_utils import carla
import rospy

import time
import subprocess
import os
import signal
import multiprocessing as mp

from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import Marker, MarkerArray

from ..world_map import Core

from carla_utils.ros import PublishWrapper
from carla_utils.ros.pub_sub import PubFormat, basic_publish
from carla_utils.ros import create_message as cm
from carla_utils.ros import convert as cvt



class RosViz(PublishWrapper):
    pub_dict = {
        '~town_map': PubFormat(MarkerArray, basic_publish, True, 1),
        '/tf': PubFormat(TFMessage, basic_publish, False, 1),
        '~vehicle_bbx': PubFormat(MarkerArray, basic_publish, False, 1),
    }

    def __init__(self, config: cu.basic.YamlConfig):
        node_name = 'carla'
        core = Core(config, use_tm=False)
        self.client, self.world, self.town_map = core.client, core.world, core.town_map
        self.town_map_name = self.town_map.name

        super().__init__(config, node_name)

        self.ros_clock = rospy.Rate(100)
        # self.cnt = 1



    def run_once(self):
        timestamp = time.time()
        # topic = '/tf_static'
        # self.ros_pubish.publish(topic, (self.sensor_manager, self.vehicle_frame_id))

        topic = '~town_map'
        self.map_viz = cm.Map(self.global_frame_id, time.time(), self.town_map)
        self.map_viz.markers[0].header = cvt.header(self.global_frame_id, timestamp)
        self.map_viz.markers[1].header = cvt.header(self.global_frame_id, timestamp)
        self.ros_pubish.publish(topic, self.map_viz)

        if self.world.get_map().name != self.town_map_name:
            self.world = self.client.get_world()
            self.town_map = self.world.get_map()
            self.town_map_name = self.town_map.name
        return


    def run_step(self):
        timestamp = time.time()
        actors = self.world.get_actors()
        vehicles = actors.filter('*vehicle*')
        static_vehicles = None
        if hasattr(self.world, 'get_environment_objects'):
            static_vehicles = self.world.get_environment_objects(carla.CityObjectLabel.Vehicles)

        topic = '/tf'
        tfmsg = cm.VehiclesTransform(self.global_frame_id, timestamp, vehicles)
        if static_vehicles != None:
            tfmsg.transforms.extend(cm.StaticVehiclesTransform(self.global_frame_id, timestamp, static_vehicles).transforms)
        self.ros_pubish.publish(topic, tfmsg)

        topic = '~vehicle_bbx'
        bbx = cm.BoundingBoxes(None, timestamp, vehicles)
        if static_vehicles != None:
            bbx.markers.extend(cm.StaticBoundingBoxes(None, timestamp, static_vehicles).markers)
        self.ros_pubish.publish(topic, bbx)

        # if self.cnt % 50 == 0:
        #     self.run_once()
        # self.cnt += 1
        return


    def run(self):
        self.run_once()
        while not rospy.is_shutdown():
            t1 = time.time()

            self.run_step()
            t2 = time.time()
            # print('time: ', t2-t1, 1/(t2-t1))

            self.ros_clock.sleep()


def start_rviz(config):
    try:
        rospy.get_master().getSystemState()
    except:
        subprocess.Popen('roscore')
        time.sleep(1)
    
    cmd_str = 'rosrun rviz rviz -d utils/carla_{}_{}.rviz'.format(config.host.replace('.', '_'), str(config.port))
    print('run cmd:\n    ', cmd_str)
    rviz = subprocess.Popen(cmd_str, shell=True)
    return rviz

def start_repub():
    #     <arg name="input_topic" value="env_info"/>
    # <arg name="output_fields" value="road_path obstacle_array"/>
    # <node pkg="carla_msgs" type="RepubField" name="repub_field"
    #     args="$(arg input_topic) $(arg output_fields)"
    #     ns="ego_vehicle" output="screen">
    pass



if __name__ == "__main__":
    config = cu.basic.YamlConfig()
    args = cu.utils.default_argparser().parse_args()
    config.update(args)

    rviz = start_rviz(config)
    time.sleep(0.5)

    try:
        ros_viz = RosViz(config)
        ros_viz.run()
    # except Exception as e:
    #     import traceback
    #     traceback.print_exc()
    finally:
        ros_viz.kill()
        rviz.send_signal(signal.SIGKILL)
        os.killpg(os.getpgid(rviz.pid), signal.SIGKILL)

