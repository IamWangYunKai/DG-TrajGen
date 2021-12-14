import carla_utils as cu
from carla_utils import carla

import numpy as np
import os, sys
from os.path import join
import threading
from tqdm import tqdm


from carla_utils.basic import YamlConfig
from carla_utils.rl_template.recorder import PicklableTownMap
from carla_utils.viz import plt
from carla_utils.viz import open3d


class Vopen3d(open3d.Visualizer):
    def connect_to_server(self):
        self.world = None
        self.town_map = self.config.town_map

class Vplt(plt.Visualizer):
    def connect_to_server(self):
        self.world = None
        self.town_map: PicklableTownMap = self.config.town_map

    def get_topology(self):
        return self.town_map.topology


class ReplayRecord(object):
    def __init__(self, config):
        file_path = join(config.dir, str(config.index) + '.txt')
        self._record = cu.rl_template.Recorder.load_from_disk(file_path)
        
        scenario = self._record.pop('scenario')

        map_path = join(config.dir, scenario['map_name'] + '.txt')
        town_map = cu.rl_template.Recorder.load_from_disk(map_path)
        self.town_map = town_map
        config.set('town_map', town_map)

        self.vv = config.viz_type(config)
        self.clock = cu.system.Clock(scenario['frequency'] *config.speed)
        print(scenario['frequency'])
        self.clock_viz = cu.system.Clock(100)

        self.agents = []

        self.num_replay = config.num_replay
        self.thread_replay = threading.Thread(target=self.replay)
        self.thread_replay.start()
        return
    
    def replay(self):
        for _ in tqdm(range(self.num_replay)):
            self.replay_once()
        print('[ReplayRecord] finish replay')
        return

    def replay_once(self):
        agent_keys = [key for key in self._record.keys() if key.startswith('agent')]
        obstacle_keys = [key for key in self._record.keys() if key.startswith('obstacle')]
        timestamps = set()
        for agent_key in agent_keys:
            timestamps.update( set(self._record[agent_key].keys()) )
        timestamps = sorted(list(timestamps))

        print('[replay_once] len of timestamps: ', len(timestamps))
        for timestamp in timestamps:
            self.clock.tick_begin()

            agents = [self._record[agent_key].get(timestamp, None) for agent_key in agent_keys]
            obstacles = [self._record[obstacle_key].get(timestamp, None) for obstacle_key in obstacle_keys]
            self.agents = [i.agent for i in agents + obstacles if i != None]

            self.clock.tick_end()
        return


    def run(self):
        while True:
            self.clock_viz.tick_begin()
            self.vv.run_step(self.agents)
            self.vv.update_vis()
            self.clock_viz.tick_end()



class ReplayRecords(ReplayRecord):
    """
        Only for one scenario.
    """
    def __init__(self, config):
        indices = list(range(*config.indices))
        self.records = {}
        for index in indices:
            file_path = join(config.dir, str(index) + '.txt')
            if os.path.isfile(file_path):
                self.records[index] = cu.rl_template.Recorder.load_from_disk(file_path)
        self.record_indices = list(self.records.keys())

        scenario = self.records[self.record_indices[0]].pop('scenario')

        map_path = join(config.dir, scenario['map_name'] + '.txt')
        town_map = cu.rl_template.Recorder.load_from_disk(map_path)
        self.town_map = town_map
        config.set('town_map', town_map)

        self.vv = config.viz_type(config)
        self.clock = cu.system.Clock(scenario['frequency'] *config.speed)
        print(scenario['frequency'])
        self.clock_viz = cu.system.Clock(100)

        self.agents = []

        self.num_replay = config.num_replay
        self.thread_replay = threading.Thread(target=self.replay)
        self.thread_replay.start()
        return

    def replay(self):
        for index, record in self.records.items():
            self._record = record
            for _ in tqdm(range(self.num_replay)):
                self.replay_once()
            print('[ReplayRecord] finish replay {}'.format(str(index)))
        return



def generate_args():
    from carla_utils.utils import default_argparser
    argparser = default_argparser()

    argparser.add_argument('-d', dest='description', default='Nothing', help='[Method] description.')

    argparser.add_argument('-n', dest='num_replay', default=1, type=int, help='num_replay.')

    argparser.add_argument('-s', '--speed', dest='speed', default=1.0, type=float, help='play speed.')

    argparser.add_argument('--dir', default='./', type=str, help='')
    argparser.add_argument('--index', default=-1, type=int, help='')
    argparser.add_argument('--indices', default=[-1,100], type=int, nargs='+', help='')
    argparser.add_argument('-t', '--type', type=int, choices=[0, 1], default=0, help='0: open3d, 1: plt')

    args = argparser.parse_args()
    return args



if __name__ == "__main__":
    config = YamlConfig()
    args = generate_args()
    config.update(args)

    ### mode 1
    if config.index > 0:
        RR = ReplayRecord
    elif config.indices[0] > 0:
        RR = ReplayRecords
    else:
        raise NotImplementedError

    ### mode 2
    if config.type == 0:
        config.set('viz_type', Vopen3d)
    elif config.type == 1:
        config.set('viz_type', Vplt)
    else:
        raise NotImplementedError

    rr = RR(config)
    rr.run()


