
import numpy as np
import datetime
import pygame


from .tools import find_weather_presets, make_surface, parse_collision_history
from .tools import get_actor_display_name, FadingText


class HUD(object):
    def __init__(self, core, vehicle, sensors_master, display):
        self.client, self.world, self.town_map = core.client, core.world, core.town_map
        self.vehicle, self.sensors_master = vehicle, sensors_master
        self.display = display

        dim = (self.display.get_width(), self.display.get_height())
        self.dim = dim
        self.origin_dim = dim
        width, height = dim[0], dim[1]
        self.desired_aspect_ratio = float(width) / float(height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))

        self.settings = self.world.get_settings()
        self.world.on_tick(self.on_world_tick)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._server_clock = pygame.time.Clock()

        self._show_info = True
        self._info_text = []

        self._weather_presets = find_weather_presets()
        self._weather_index = 0


    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, clock):
        self._notifications.tick(clock)

        if self.sensors_master.get_camera() != None:
            carla_image = self.sensors_master.get_camera().get_raw_data()
            if carla_image == None: print('[pygame_interaction] tick: warning'); return
            self._surface = make_surface(carla_image)

        if not self._show_info:
            self._info_text = []
            return
        t = self.vehicle.get_transform()
        v = self.vehicle.get_velocity()
        c = self.vehicle.get_control()
        a = self.vehicle.get_acceleration()

        # self.notification('Collision with %r' % data.other_actor.type_id)
        colhist = parse_collision_history(self.sensors_master[('sensor.other.collision', 'default')].get_data())
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = self.world.get_actors().filter('vehicle.*')
        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(self.vehicle, truncate=20),
            'Map:     % 20s' % self.town_map.name,
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * np.sqrt(v.x**2 + v.y**2 + v.z**2)),
            'Accelero: %14.1f m/s2' % (np.sqrt(a.x**2 + a.y**2 + a.z**2)),
            'Location:% 20s' % ('(% 5.1f, % 5.1f, % 3.1f)' % (t.location.x, t.location.y, t.location.z)),
            'Height:  % 18.0f m' % t.location.z,
            '']
        self._info_text += [
            ('Throttle:', c.throttle, 0.0, 1.0),
            ('Steer:', c.steer, -1.0, 1.0),
            ('Brake:', c.brake, 0.0, 1.0),
            ('Reverse:', c.reverse),
            ('Hand brake:', c.hand_brake),
            ('Manual:', c.manual_gear_shift),
            'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear)]
        self._info_text += [
            '', 'Collision:', collision, '', 'Number of vehicles: % 8d' % (len(vehicles)-1)]
        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']
            distance = lambda l: np.sqrt((l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
            vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != self.vehicle.id]
            for d, vehicle in sorted(vehicles):
                if d > 800.0:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append('% 4dm %s' % (d, vehicle_type))
        self.render(self.display)
        
    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.notification('Weather: %s' % preset[1])
        self.world.set_weather(preset[0])

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        display.fill((255,255,255))

        if hasattr(self, '_surface'):
            size = self._surface.get_size()
            desired_aspect_ratio = float(size[0]) / float(size[1])
            dim1 = (self.dim[0], round(self.dim[0]/desired_aspect_ratio))
            dim2 = (round(self.dim[1]*desired_aspect_ratio), self.dim[1])
            scaled_size = min(dim1, dim2)
            display.blit(pygame.transform.scale(self._surface, scaled_size), (0, 0))

        info_surface = pygame.Surface((220, self.dim[1]))
        alpha = 100 if self._show_info else 0
        info_surface.set_alpha(alpha)
        display.blit(info_surface, (0, 0))
        v_offset = 4
        bar_h_offset = 100
        bar_width = 106
        for item in self._info_text:
            if v_offset + 18 > self.dim[1]:
                break
            if isinstance(item, list):
                if len(item) > 1:
                    points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                    pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                item = None
                v_offset += 18
            elif isinstance(item, tuple):
                if isinstance(item[1], bool):
                    rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                    pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                else:
                    rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                    pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                    f = (item[1] - item[2]) / (item[3] - item[2])
                    if item[2] < 0.0:
                        rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                    else:
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                    pygame.draw.rect(display, (255, 255, 255), rect)
                item = item[0]
            if item:  # At this point has to be a str.
                surface = self._font_mono.render(item, True, (255, 255, 255))
                display.blit(surface, (8, v_offset))
            v_offset += 18

        self._notifications.render(display)
