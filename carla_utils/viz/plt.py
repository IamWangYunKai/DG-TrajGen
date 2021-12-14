import carla_utils as cu
from carla_utils import carla

import numpy as np
import matplotlib.pyplot as plt
import time

from ..system import Clock
from ..world_map import connect_to_server
from ..augment import ActorVertices




class Visualizer(object):
    def __init__(self, config):
        self.config = config
        self.connect_to_server()
        self.clock = Clock(100)

        self.window_name = 'Vehicles Visualisation Example' + '   ' + config.host + ':' + str(config.port)

        ### visualization boundary
        waypoints = self.town_map.generate_waypoints(2)
        margin = 50
        max_x = max(waypoints, key=lambda x: x.transform.location.x).transform.location.x + margin
        max_y = max(waypoints, key=lambda x: x.transform.location.y).transform.location.y + margin
        min_x = min(waypoints, key=lambda x: x.transform.location.x).transform.location.x - margin
        min_y = min(waypoints, key=lambda x: x.transform.location.y).transform.location.y - margin

        fig, ax = plt.subplots()
        fig.canvas.set_window_title(self.window_name)
        ax.set_facecolor(np.array([180, 180, 180], np.float64) / 255)

        plt.xlim([min_x, max_x])
        plt.ylim([min_y, max_y])
        plt.gca().set_aspect('equal', adjustable='box')

        topology = self.get_topology()
        self.background = []
        for t in topology:
            line = plt.plot(t.x, t.y)[0]
            self.background.append(line)

        plt.pause(0.00001)
        self.foreground = []
        return


    def get_topology(self):
        topology = cu.get_topology(self.town_map, sampling_resolution=2.0)
        topology = [t.info for t in topology]
        return topology


    def connect_to_server(self):
        host, port, timeout, map_name = self.config.host, self.config.port, self.config.timeout, self.config.map_name
        _, self.world, self.town_map = connect_to_server(host, port, timeout, map_name)


    def update_vis(self):
        plt.pause(0.00001)
        return


    def run_step(self, vehicles):
        [i.remove() for i in self.foreground]
        self.foreground = []

        for vehicle in vehicles:
            vertices, lines = ActorVertices.d2arrow(vehicle)
            vertices = np.vstack([vertices, vertices[[4]], vertices[[0]]])

            color = vehicle.attributes.get('color', '190,190,190')
            color = np.array(eval(color)).astype(np.float64) / 255

            line = plt.plot(vertices[:,0], vertices[:,1], '-', color=color)[0]
            self.foreground.append(line)
        return

    
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

    visualizer = Visualizer(config, )
    try:
        visualizer.run()
    except KeyboardInterrupt:
        print('canceled by user')


