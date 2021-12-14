from carla_utils import carla

import pygame
from pygame.locals import KMOD_CTRL
from pygame.locals import KMOD_SHIFT
from pygame.locals import K_0
from pygame.locals import K_9
from pygame.locals import K_BACKQUOTE
from pygame.locals import K_BACKSPACE
from pygame.locals import K_COMMA
from pygame.locals import K_DOWN
from pygame.locals import K_ESCAPE
from pygame.locals import K_F1
from pygame.locals import K_LEFT
from pygame.locals import K_PERIOD
from pygame.locals import K_RIGHT
from pygame.locals import K_SLASH
from pygame.locals import K_SPACE
from pygame.locals import K_TAB
from pygame.locals import K_UP
from pygame.locals import K_a
from pygame.locals import K_c
from pygame.locals import K_g
from pygame.locals import K_d
from pygame.locals import K_h
from pygame.locals import K_m
from pygame.locals import K_n
from pygame.locals import K_p
from pygame.locals import K_q
from pygame.locals import K_r
from pygame.locals import K_s
from pygame.locals import K_w
from pygame.locals import K_l
from pygame.locals import K_i
from pygame.locals import K_z
from pygame.locals import K_x
from pygame.locals import K_MINUS
from pygame.locals import K_EQUALS



class KeyboardControl(object):
    def __init__(self, hud, display):
        self.hud = hud
        self._autopilot_enabled = False

        self._control = carla.VehicleControl()
        self._lights = carla.VehicleLightState.NONE
        self.hud.vehicle.set_autopilot(self._autopilot_enabled)
        self.hud.vehicle.set_light_state(self._lights)
        self._steer_cache = 0.0

        self._desired_aspect_ratio = self.hud.desired_aspect_ratio


    def parse_events(self, display, clock):
        if isinstance(self._control, carla.VehicleControl):
            current_lights = self._lights
        else:
            raise NotImplementedError
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt('quit')
            elif event.type == pygame.VIDEORESIZE:
                width = event.w
                height = event.h
                dim1 = (width, round(width/self._desired_aspect_ratio))
                dim2 = (round(height*self._desired_aspect_ratio), height)
                self._screen_size = min(dim1, dim2)
                # display = pygame.display.set_mode(self.hud.dim, pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE)

            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    raise KeyboardInterrupt('quit')
                elif event.key == K_F1:
                    self.hud.toggle_info()
                elif event.key == K_TAB:
                    self.hud.sensors_master.toggle_camera()
                elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                    self.hud.next_weather(reverse=True)
                elif event.key == K_c:
                    self.hud.next_weather()
                elif event.key == K_BACKQUOTE:
                    pass
                elif event.key == K_n:
                    pass
                elif event.key > K_0 and event.key <= K_9:
                    pass

                elif event.key == K_r and not (pygame.key.get_mods() & KMOD_CTRL):
                    pass
                elif event.key == K_r and (pygame.key.get_mods() & KMOD_CTRL):
                    pass
                elif event.key == K_p and (pygame.key.get_mods() & KMOD_CTRL):
                    pass
                elif event.key == K_MINUS and (pygame.key.get_mods() & KMOD_CTRL):
                    pass
                elif event.key == K_EQUALS and (pygame.key.get_mods() & KMOD_CTRL):
                    pass

                if event.key == K_q:
                    self._control.gear = 1 if self._control.reverse else -1
                elif event.key == K_m:
                    self._control.manual_gear_shift = not self._control.manual_gear_shift
                    self._control.gear = self.hud.vehicle.get_control().gear
                    self.hud.notification('%s Transmission' %
                                           ('Manual' if self._control.manual_gear_shift else 'Automatic'))
                elif self._control.manual_gear_shift and event.key == K_COMMA:
                    self._control.gear = max(-1, self._control.gear - 1)
                elif self._control.manual_gear_shift and event.key == K_PERIOD:
                    self._control.gear = min(5, self._control.gear + 1)
                elif event.key == K_p and not pygame.key.get_mods() & KMOD_CTRL:
                    self._autopilot_enabled = not self._autopilot_enabled
                    self.hud.vehicle.set_autopilot(self._autopilot_enabled)
                    self.hud.notification('Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))
                elif event.key == K_l and pygame.key.get_mods() & KMOD_CTRL:
                    current_lights ^= carla.VehicleLightState.Special1
                elif event.key == K_l and pygame.key.get_mods() & KMOD_SHIFT:
                    current_lights ^= carla.VehicleLightState.HighBeam
                elif event.key == K_l:
                    # Use 'L' key to switch between lights:
                    # closed -> position -> low beam -> fog
                    if not self._lights & carla.VehicleLightState.Position:
                        self.hud.notification("Position lights")
                        current_lights |= carla.VehicleLightState.Position
                    else:
                        self.hud.notification("Low beam lights")
                        current_lights |= carla.VehicleLightState.LowBeam
                    if self._lights & carla.VehicleLightState.LowBeam:
                        self.hud.notification("Fog lights")
                        current_lights |= carla.VehicleLightState.Fog
                    if self._lights & carla.VehicleLightState.Fog:
                        self.hud.notification("Lights off")
                        current_lights ^= carla.VehicleLightState.Position
                        current_lights ^= carla.VehicleLightState.LowBeam
                        current_lights ^= carla.VehicleLightState.Fog
                elif event.key == K_i:
                    current_lights ^= carla.VehicleLightState.Interior
                elif event.key == K_z:
                    current_lights ^= carla.VehicleLightState.LeftBlinker
                elif event.key == K_x:
                    current_lights ^= carla.VehicleLightState.RightBlinker

        if not self._autopilot_enabled:
            pressed = self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())
            self._control.reverse = self._control.gear < 0
            # Set automatic control-related vehicle lights
            if self._control.brake:
                current_lights |= carla.VehicleLightState.Brake
            else: # Remove the Brake flag
                current_lights &= ~carla.VehicleLightState.Brake
            if self._control.reverse:
                current_lights |= carla.VehicleLightState.Reverse
            else: # Remove the Reverse flag
                current_lights &= ~carla.VehicleLightState.Reverse
            if current_lights != self._lights: # Change the light state only if necessary
                self._lights = current_lights
                self.hud.vehicle.set_light_state(carla.VehicleLightState(self._lights))

            if pressed:
                self.hud.vehicle.apply_control(self._control)


    def _parse_vehicle_keys(self, keys, milliseconds):
        flag = False
        if keys[K_UP] or keys[K_w]:
            flag = True
            self._control.throttle = min(self._control.throttle + 0.01, 1)
        else:
            self._control.throttle = 0.0

        if keys[K_DOWN] or keys[K_s]:
            flag = True
            self._control.brake = min(self._control.brake + 0.2, 1)
        else:
            self._control.brake = 0

        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            flag = True
            if self._steer_cache > 0:
                self._steer_cache = 0
            else:
                self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            flag = True
            if self._steer_cache < 0:
                self._steer_cache = 0
            else:
                self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self._control.hand_brake = keys[K_SPACE]
        return flag

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


