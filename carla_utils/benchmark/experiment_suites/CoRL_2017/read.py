
import carla


def read(file_path, town_map):
    poses = []
    file = open(file_path)
    while True:
        line = file.readline()
        if not line:
            break
        if line == '\n':
            continue

        line_list = line.split(',')
        start_x, start_y = eval(line_list[0]), eval(line_list[1])
        end_x, end_y = eval(line_list[2]), eval(line_list[3])

        start_transform = town_map.get_waypoint(carla.Location(x=start_x, y=start_y)).transform
        end_transform = town_map.get_waypoint(carla.Location(x=end_x, y=end_y)).transform
        poses.append([start_transform, end_transform])
    file.close()
    return poses
