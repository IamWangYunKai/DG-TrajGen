


class EnvInfo(object):
    def __init__(self, frame_id, time_stamp, road_path, obstacle_array):
        # default params
        self.frame_id = frame_id
        self.time_stamp = time_stamp
        self.road_path = road_path
        self.obstacle_array = obstacle_array