from carla_utils import carla

import os, hashlib, glob
import numpy as np
import cv2
from PIL import Image


COLOR_BUTTER_0 = (252, 233, 79)
COLOR_BUTTER_1 = (237, 212, 0)
COLOR_BUTTER_2 = (196, 160, 0)

COLOR_ORANGE_0 = (252, 175, 62)
COLOR_ORANGE_1 = (245, 121, 0)
COLOR_ORANGE_2 = (209, 92, 0)

COLOR_CHOCOLATE_0 = (233, 185, 110)
COLOR_CHOCOLATE_1 = (193, 125, 17)
COLOR_CHOCOLATE_2 = (143, 89, 2)

COLOR_CHAMELEON_0 = (138, 226, 52)
COLOR_CHAMELEON_1 = (115, 210, 22)
COLOR_CHAMELEON_2 = (78, 154, 6)

COLOR_SKY_BLUE_0 = (114, 159, 207)
COLOR_SKY_BLUE_1 = (52, 101, 164)
COLOR_SKY_BLUE_2 = (32, 74, 135)

COLOR_PLUM_0 = (173, 127, 168)
COLOR_PLUM_1 = (117, 80, 123)
COLOR_PLUM_2 = (92, 53, 102)

COLOR_SCARLET_RED_0 = (239, 41, 41)
COLOR_SCARLET_RED_1 = (204, 0, 0)
COLOR_SCARLET_RED_2 = (164, 0, 0)

COLOR_ALUMINIUM_0 = (238, 238, 236)
COLOR_ALUMINIUM_1 = (211, 215, 207)
COLOR_ALUMINIUM_2 = (186, 189, 182)
COLOR_ALUMINIUM_3 = (136, 138, 133)
COLOR_ALUMINIUM_4 = (85, 87, 83)
COLOR_ALUMINIUM_4_5 = (66, 62, 64)
COLOR_ALUMINIUM_5 = (46, 52, 54)


COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)


class TownMapImage(object):
    def __init__(self, core, resolution):
        """
        
        
        Args:
            resolution: m/pix
        
        Returns:
            
        """

        self.core = core
        self.world, self.town_map = self.core.world, self.core.town_map

        self.resolution = resolution

        waypoints = self.town_map.generate_waypoints(2)
        margin = 50
        max_x = max(waypoints, key=lambda x: x.transform.location.x).transform.location.x + margin
        max_y = max(waypoints, key=lambda x: x.transform.location.y).transform.location.y + margin
        min_x = min(waypoints, key=lambda x: x.transform.location.x).transform.location.x - margin
        min_y = min(waypoints, key=lambda x: x.transform.location.y).transform.location.y - margin

        range_x = max_x - min_x
        range_y = max_y - min_y

        self.height, self.width = int(range_y / resolution), int(range_x / resolution)
        self._world_offset = (min_x, min_y)
        self.world_offset = np.array([[min_x, min_y]])

        # print('height is {}, width is {}'.format(str(self.height), str(self.width)))

        ### paramters
        self.precision = 0.05
        self._pixels_per_meter = int(1/resolution)


        # Get hash based on content
        opendrive_content = self.town_map.to_opendrive()
        hash_func = hashlib.sha1()
        hash_func.update(opendrive_content.encode("UTF-8"))
        opendrive_hash = str(hash_func.hexdigest())

        # Build path for saving or loading the cached rendered map
        filename = self.town_map.name + '_' + opendrive_hash + '.png'
        dirname = 'cache'
        full_path = str(os.path.join(dirname, filename))

        if os.path.isfile(full_path):
            # Load Image
            self.load(full_path)
        else:
            # Render map
            self.draw()

            # If folders path does not exist, create it
            if not os.path.exists(dirname):
                os.makedirs(dirname)

            # Remove files if selected town had a previous version saved
            list_filenames = glob.glob(os.path.join(dirname, self.town_map.name) + '*')
            for town_filename in list_filenames:
                os.remove(town_filename)

            # Save rendered map for next executions of same map
            self.save(full_path)
        return


    def draw(self):
        self.image_cv2 = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.image_cv2[:,:,] = COLOR_ALUMINIUM_4

        self.draw_topology()

        self.image_cv2 = cv2.cvtColor(self.image_cv2, cv2.COLOR_RGB2BGR)
        self.image_pil = Image.fromarray(self.image_cv2)
        return


    def load(self, full_path):
        self.image_cv2 = cv2.imread(full_path)
        self.image_pil = Image.fromarray(self.image_cv2)
        

    def save(self, full_path):
        cv2.imwrite(full_path, self.image_cv2)
    

    def world2image(self, location, offset=(0, 0)):
        """Converts the world coordinates to pixel coordinates"""
        x = self._pixels_per_meter * (location.x - self._world_offset[0])
        y = self._pixels_per_meter * (location.y - self._world_offset[1])
        return [int(x - offset[0]), int(y - offset[1])]


    def draw_topology(self):
        topology = self.town_map.get_topology()
        topology = [x[0] for x in topology]
        topology = sorted(topology, key=lambda w: w.transform.location.z)
        set_waypoints = []
        for waypoint in topology:
            waypoints = [waypoint]

            # Generate waypoints of a road id. Stop when road id differs
            nxt = waypoint.next(self.precision)
            if len(nxt) > 0:
                nxt = nxt[0]
                while nxt.road_id == waypoint.road_id:
                    waypoints.append(nxt)
                    nxt = nxt.next(self.precision)
                    if len(nxt) > 0:
                        nxt = nxt[0]
                    else:
                        break
            set_waypoints.append(waypoints)

            # Draw Shoulders, Parkings and Sidewalks
            PARKING_COLOR = COLOR_ALUMINIUM_4_5  ## COLOR_ALUMINIUM_4_5 = pygame.Color(66, 62, 64)
            SHOULDER_COLOR = COLOR_ALUMINIUM_5   ## COLOR_ALUMINIUM_5 = pygame.Color(46, 52, 54)
            SIDEWALK_COLOR = COLOR_ALUMINIUM_3   ## COLOR_ALUMINIUM_3 = pygame.Color(136, 138, 133)

            shoulder = [[], []]
            parking = [[], []]
            sidewalk = [[], []]

            for w in waypoints:
                # Classify lane types until there are no waypoints by going left
                l = w.get_left_lane()
                while l and l.lane_type != carla.LaneType.Driving:
                    if l.lane_type == carla.LaneType.Shoulder:
                        shoulder[0].append(l)
                    if l.lane_type == carla.LaneType.Parking:
                        parking[0].append(l)
                    if l.lane_type == carla.LaneType.Sidewalk:
                        sidewalk[0].append(l)
                    l = l.get_left_lane()

                # Classify lane types until there are no waypoints by going right
                r = w.get_right_lane()
                while r and r.lane_type != carla.LaneType.Driving:
                    if r.lane_type == carla.LaneType.Shoulder:
                        shoulder[1].append(r)
                    if r.lane_type == carla.LaneType.Parking:
                        parking[1].append(r)
                    if r.lane_type == carla.LaneType.Sidewalk:
                        sidewalk[1].append(r)
                    r = r.get_right_lane()

            # Draw classified lane types
            self.draw_lane(shoulder, SHOULDER_COLOR)
            self.draw_lane(parking, PARKING_COLOR)
            self.draw_lane(sidewalk, SIDEWALK_COLOR)


        ## Draw Roads
        for waypoints in set_waypoints:
            waypoint = waypoints[0]
            road_left_side = [lateral_shift(w.transform, -w.lane_width * 0.5) for w in waypoints]
            road_right_side = [lateral_shift(w.transform, w.lane_width * 0.5) for w in waypoints]

            polygon = road_left_side + [x for x in reversed(road_right_side)]
            polygon = [self.world2image(x) for x in polygon]

            if len(polygon) > 2:
                cv2.fillPoly(self.image_cv2, pts=[np.array(polygon, np.int32)], color=COLOR_ALUMINIUM_5)

            # Draw Lane Markings and Arrows
            if not waypoint.is_junction:
                self.draw_lane_marking([waypoints, waypoints])
                for n, wp in enumerate(waypoints):
                    if ((n + 1) % 400) == 0:
                        self.draw_arrow(wp.transform)

        return


    def draw_lane(self, lane, color):
        """Renders a single lane in a surface and with a specified color"""
        for side in lane:
            lane_left_side = [lateral_shift(w.transform, -w.lane_width * 0.5) for w in side]
            lane_right_side = [lateral_shift(w.transform, w.lane_width * 0.5) for w in side]

            polygon = lane_left_side + [x for x in reversed(lane_right_side)]
            polygon = [self.world2image(x) for x in polygon]

            if len(polygon) > 2:
                # import pdb; pdb.set_trace()
                # cv2.polylines(self.image_cv2, [np.array(polygon, np.int32)], True, color, 5)
                # cv2.polylines(self.image_cv2, [np.array(polygon, np.int32)], True, color)
                cv2.fillPoly(self.image_cv2, pts=[np.array(polygon, np.int32)], color=color)
                # pygame.draw.polygon(surface, color, polygon, 5)
                # pygame.draw.polygon(surface, color, polygon)


    def draw_lane_marking(self, waypoints):
        """Draws the left and right side of lane markings"""
        # Left Side
        self.draw_lane_marking_single_side(waypoints[0], -1)

        # Right Side
        self.draw_lane_marking_single_side(waypoints[1], 1)

    def draw_lane_marking_single_side(self, waypoints, sign):
        """Draws the lane marking given a set of waypoints and decides whether drawing the right or left side of
        the waypoint based on the sign parameter"""
        lane_marking = None

        marking_type = carla.LaneMarkingType.NONE
        previous_marking_type = carla.LaneMarkingType.NONE

        marking_color = carla.LaneMarkingColor.Other
        previous_marking_color = carla.LaneMarkingColor.Other

        markings_list = []
        temp_waypoints = []
        current_lane_marking = carla.LaneMarkingType.NONE
        for sample in waypoints:
            lane_marking = sample.left_lane_marking if sign < 0 else sample.right_lane_marking

            if lane_marking is None:
                continue

            marking_type = lane_marking.type
            marking_color = lane_marking.color

            if current_lane_marking != marking_type:
                # Get the list of lane markings to draw
                markings = self.get_lane_markings(
                    previous_marking_type,
                    lane_marking_color_to_tango(previous_marking_color),
                    temp_waypoints,
                    sign)
                current_lane_marking = marking_type

                # Append each lane marking in the list
                for marking in markings:
                    markings_list.append(marking)

                temp_waypoints = temp_waypoints[-1:]

            else:
                temp_waypoints.append((sample))
                previous_marking_type = marking_type
                previous_marking_color = marking_color

        # Add last marking
        last_markings = self.get_lane_markings(
            previous_marking_type,
            lane_marking_color_to_tango(previous_marking_color),
            temp_waypoints,
            sign)
        for marking in last_markings:
            markings_list.append(marking)

        # Once the lane markings have been simplified to Solid or Broken lines, we draw them

        for markings in markings_list:
            if markings[0] == carla.LaneMarkingType.Solid:
                self.draw_solid_line(markings[1], False, markings[2], 2)
            elif markings[0] == carla.LaneMarkingType.Broken:
                self.draw_broken_line(markings[1], False, markings[2], 2)
        return

    def draw_arrow(self, transform, color=COLOR_ALUMINIUM_2):
        """ Draws an arrow with a specified color given a transform"""
        transform.rotation.yaw += 180
        forward = transform.get_forward_vector()
        transform.rotation.yaw += 90
        right_dir = transform.get_forward_vector()
        end = transform.location
        start = end - 2.0 * forward
        right = start + 0.8 * forward + 0.4 * right_dir
        left = start + 0.8 * forward - 0.4 * right_dir

        # Draw lines
        # pygame.draw.lines(surface, color, False, [self.world2image(x) for x in [start, end]], 4)
        # pygame.draw.lines(surface, color, False, [self.world2image(x) for x in [left, start, right]], 4)
        ## !warning
        cv2.polylines(self.image_cv2, [np.array([self.world2image(x) for x in [start, end]], np.int32)], False, color, 4)
        cv2.polylines(self.image_cv2, [np.array([self.world2image(x) for x in [left, start, right]], np.int32)], False, color, 4)


    def draw_solid_line(self, color, closed, points, width):
        """Draws solid lines in a surface given a set of points, width and color"""
        if len(points) >= 2:
            # import pdb; pdb.set_trace()
            # pygame.draw.lines(surface, color, closed, points, width)
             cv2.polylines(self.image_cv2, [np.array(points, np.int32)], closed, color, width)

    def draw_broken_line(self, color, closed, points, width):
        """Draws broken lines in a surface given a set of points, width and color"""
        # Select which lines are going to be rendered from the set of lines
        broken_lines = [x for n, x in enumerate(zip(*(iter(points),) * 20)) if n % 3 == 0]

        # Draw selected lines
        width = 1   ## TODO
        for line in broken_lines:
            cv2.polylines(self.image_cv2, [np.array(line, np.int32)], closed, color, width)
            # for start, end in zip(line[0::2], line[1::2]):
            #     import pdb; pdb.set_trace()
            #     cv2.line(self.image_cv2, start, end, color, width)



    def get_lane_markings(self, lane_marking_type, lane_marking_color, waypoints, sign):
        """For multiple lane marking types (SolidSolid, BrokenSolid, SolidBroken and BrokenBroken), it converts them
            as a combination of Broken and Solid lines"""
        margin = 0.25
        marking_1 = [self.world2image(lateral_shift(w.transform, sign * w.lane_width * 0.5)) for w in waypoints]
        if lane_marking_type == carla.LaneMarkingType.Broken or (lane_marking_type == carla.LaneMarkingType.Solid):
            return [(lane_marking_type, lane_marking_color, marking_1)]
        else:
            marking_2 = [self.world2image(lateral_shift(w.transform,
                                                        sign * (w.lane_width * 0.5 + margin * 2))) for w in waypoints]
            if lane_marking_type == carla.LaneMarkingType.SolidBroken:
                return [(carla.LaneMarkingType.Broken, lane_marking_color, marking_1),
                        (carla.LaneMarkingType.Solid, lane_marking_color, marking_2)]
            elif lane_marking_type == carla.LaneMarkingType.BrokenSolid:
                return [(carla.LaneMarkingType.Solid, lane_marking_color, marking_1),
                        (carla.LaneMarkingType.Broken, lane_marking_color, marking_2)]
            elif lane_marking_type == carla.LaneMarkingType.BrokenBroken:
                return [(carla.LaneMarkingType.Broken, lane_marking_color, marking_1),
                        (carla.LaneMarkingType.Broken, lane_marking_color, marking_2)]
            elif lane_marking_type == carla.LaneMarkingType.SolidSolid:
                return [(carla.LaneMarkingType.Solid, lane_marking_color, marking_1),
                        (carla.LaneMarkingType.Solid, lane_marking_color, marking_2)]

        return [(carla.LaneMarkingType.NONE, carla.LaneMarkingColor.Other, [])]




def lateral_shift(transform, shift):
    """Makes a lateral shift of the forward vector of a transform"""
    transform.rotation.yaw += 90
    return transform.location + shift * transform.get_forward_vector()


def lane_marking_color_to_tango(lane_marking_color):
    """Maps the lane marking color enum specified in PythonAPI to a Tango Color"""
    tango_color = COLOR_BLACK

    if lane_marking_color == carla.LaneMarkingColor.White:
        tango_color = COLOR_ALUMINIUM_2

    elif lane_marking_color == carla.LaneMarkingColor.Blue:
        tango_color = COLOR_SKY_BLUE_0

    elif lane_marking_color == carla.LaneMarkingColor.Green:
        tango_color = COLOR_CHAMELEON_0

    elif lane_marking_color == carla.LaneMarkingColor.Red:
        tango_color = COLOR_SCARLET_RED_0

    elif lane_marking_color == carla.LaneMarkingColor.Yellow:
        tango_color = COLOR_ORANGE_0

    return tango_color

