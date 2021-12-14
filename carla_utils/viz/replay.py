import carla_utils as cu
from carla_utils import carla

import numpy as np
import os, sys
from os.path import join
import threading


from ..basic import YamlConfig
from .playback import generate_args


class Replay(object):
    def __init__(self, config):

        self.server = cu.launch_server(env_index=10, low_quality=False, no_display=False)
        config_server = YamlConfig(
            host=self.server.host,
            port=self.server.port,
            
        )
        self.core = cu.Core(config_server)
        self.client = self.core.client

        file_path = join(os.path.abspath(config.dir), 'recording_{}.log'.format(str(config.index)))
        bbb = self.client.show_recorder_file_info(file_path, show_all=False)
        aaa = self.client.replay_file(file_path, 0.0, 0.0, 0)


    def destroy(self):
        cu.kill_server(self.server)




if __name__ == "__main__":
    config = YamlConfig()
    args = generate_args()
    config.update(args)

    replay = Replay(config)
    try:
        import time
        time.sleep(10)
        replay.client.stop_replayer(True)
    finally:
        replay.destroy()

