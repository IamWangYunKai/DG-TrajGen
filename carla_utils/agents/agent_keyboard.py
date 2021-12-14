'''
Use ARROWS or WASD keys for control.

    W            : throttle
    S            : brake
    A/D          : steer left/right
    Q            : toggle reverse
    Space        : hand-brake
    P            : toggle autopilot
    M            : toggle manual transmission
    ,/.          : gear up/down

    L            : toggle next light type
    SHIFT + L    : toggle high beam
    Z/X          : toggle right/left blinker
    I            : toggle interior light

    TAB          : change sensor position
    ` or N       : next sensor
    [1-9]        : change to sensor [1-9]
    G            : toggle radar visualization
    C            : change weather (Shift+C reverse)

    R            : toggle recording images to disk

    CTRL + R     : toggle recording of simulation (replacing any previous)
    CTRL + P     : start replaying last recorded simulation
    CTRL + +     : increments the start time of the replay by 1 second (+SHIFT = 10 seconds)
    CTRL + -     : decrements the start time of the replay by 1 second (+SHIFT = 10 seconds)

    F1           : toggle HUD
    H/?          : toggle help
    ESC          : quit
'''

from carla_utils import carla

import pygame
import pygame.locals as pgl
import time

from ..sensor import CarlaSensorListMaster, create_sensor, DefaultCallback
from ..examples import HUD
from .agent_base import BaseAgent


## TODO real

class KeyboardAgent(BaseAgent):
    def __init__(self, config, vehicle, sensors_master, global_path):

        if config.real:
            self.get_control = self.get_control_real

        super().__init__(config, vehicle, sensors_master, global_path)

        create_view_camera(config, vehicle, sensors_master)

        width, height = config.get('width', 1000), config.get('height', 600)
        self.use_kb_control = config.get('use_kb_control', True)
        
        pygame.init()
        pygame.font.init()
        self.display = pygame.display.set_mode((width, height), pygame.HWSURFACE | pygame.DOUBLEBUF)

        self.pygame_clock = pygame.time.Clock()
        self.client = self.core.client
        self.hud = HUD(self.core, vehicle, sensors_master, self.display)
        self.kb_control = KeyboardControl(self.hud)

        print(__doc__)


    def get_target(self, _):
        self.pygame_clock.tick_busy_loop(0)
        control = self.kb_control.parse_events(self.pygame_clock)
        self.hud.tick(self.pygame_clock)
        pygame.display.flip()
        target = control
        return target


    def get_control(self, target):
        control = target
        return control   

    def get_control_real(self, target):
        control = target
        return control


    def destroy(self):
        pygame.quit()
        return super().destroy()



def create_view_camera(config, vehicle, sensors_master: CarlaSensorListMaster):
    core = config.core
    sensor_param = {
        'type_id': 'sensor.camera.rgb',
        'role_name': 'view',
        'image_size_x': 640 *2,
        'image_size_y': 360 *2,
        'fov': 120,
        'sensor_tick': 1/ config.control_frequency,
        'transform': carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
        'callback': DefaultCallback.sensor_camera_rgb,
    }

    sensor = create_sensor(core, vehicle, core.world.get_blueprint_library(), sensor_param)
    sensors_master.append(sensor, sensor_param['transform'], sensor_param['callback'])
    return



# =============================================================================
# ---- KeyboardControl --------------------------------------------------------
# =============================================================================



def _is_quit_shortcut(key):
    return (key == pgl.K_ESCAPE) or (key == pgl.K_q and pygame.key.get_mods() & pgl.KMOD_CTRL)



class KeyboardControl(object):
    def __init__(self, hud):
        self.hud = hud
        self._autopilot_enabled = False

        self._control = carla.VehicleControl()
        self._lights = carla.VehicleLightState.NONE
        self.hud.vehicle.set_autopilot(self._autopilot_enabled)
        self.hud.vehicle.set_light_state(self._lights)
        self._steer_cache = 0.0

        self._desired_aspect_ratio = self.hud.desired_aspect_ratio


    def parse_events(self, clock):
        while True:
            current_lights = self._lights
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt('quit')
                
                elif event.type == pygame.KEYUP:
                    if _is_quit_shortcut(event.key):
                        raise KeyboardInterrupt('quit')
                    elif event.key == pgl.K_c and pygame.key.get_mods() & pgl.KMOD_SHIFT:
                        self.hud.next_weather(reverse=True)
                    elif event.key == pgl.K_c:
                        self.hud.next_weather()

                    if event.key == pgl.K_q:
                        self._control.gear = 1 if self._control.reverse else -1
                    elif event.key == pgl.K_m:
                        self._control.manual_gear_shift = not self._control.manual_gear_shift
                        self._control.gear = self.hud.vehicle.get_control().gear
                        self.hud.notification('%s Transmission' %
                                            ('Manual' if self._control.manual_gear_shift else 'Automatic'))
                    elif self._control.manual_gear_shift and event.key == pgl.K_COMMA:
                        self._control.gear = max(-1, self._control.gear - 1)
                    elif self._control.manual_gear_shift and event.key == pgl.K_PERIOD:
                        self._control.gear = min(5, self._control.gear + 1)
                    elif event.key == pgl.K_p and not pygame.key.get_mods() & pgl.KMOD_CTRL:
                        self._autopilot_enabled = not self._autopilot_enabled
                        self.hud.vehicle.set_autopilot(self._autopilot_enabled)
                        self.hud.notification('Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))
                    elif event.key == pgl.K_l and pygame.key.get_mods() & pgl.KMOD_CTRL:
                        current_lights ^= carla.VehicleLightState.Special1
                    elif event.key == pgl.K_l and pygame.key.get_mods() & pgl.KMOD_SHIFT:
                        current_lights ^= carla.VehicleLightState.HighBeam
                    elif event.key == pgl.K_l:
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
                    elif event.key == pgl.K_i:
                        current_lights ^= carla.VehicleLightState.Interior
                    elif event.key == pgl.K_z:
                        current_lights ^= carla.VehicleLightState.LeftBlinker
                    elif event.key == pgl.K_x:
                        current_lights ^= carla.VehicleLightState.RightBlinker

                else:
                    continue

            pressed = self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())
            if not pressed:
                continue

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

            return self._control



    def _parse_vehicle_keys(self, keys, milliseconds):
        flag = False
        if keys[pgl.K_UP] or keys[pgl.K_w]:
            flag = True
            self._control.throttle = min(self._control.throttle + 0.1, 1)
        else:
            self._control.throttle = 0.0

        if keys[pgl.K_DOWN] or keys[pgl.K_s]:
            flag = True
            self._control.brake = min(self._control.brake + 0.2, 1)
        else:
            self._control.brake = 0

        steer_increment = 5e-4 * milliseconds
        if keys[pgl.K_LEFT] or keys[pgl.K_a]:
            flag = True
            if self._steer_cache > 0:
                self._steer_cache = 0
            else:
                self._steer_cache -= steer_increment
        elif keys[pgl.K_RIGHT] or keys[pgl.K_d]:
            flag = True
            if self._steer_cache < 0:
                self._steer_cache = 0
            else:
                self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self._control.hand_brake = keys[pgl.K_SPACE]
        return flag
