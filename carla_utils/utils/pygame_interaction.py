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


from ..examples import HUD, KeyboardControl

class PyGameInteraction(object):
    def __init__(self, config, vehicle, sensors_master):
        '''
            Args:
            config: need to contain:
                config.width
                config.height
                config.use_kb_control
        '''

        core = config.get('core', None)

        width, height = config.get('width', 1000), config.get('height', 600)
        self.use_kb_control = config.get('use_kb_control', True)
        print(__doc__)
        
        pygame.init()
        pygame.font.init()
        self.display = pygame.display.set_mode((width, height), pygame.HWSURFACE | pygame.DOUBLEBUF)

        self.clock = pygame.time.Clock()
        self.client = core.client
        self.hud = HUD(core, vehicle, sensors_master, self.display)
        self.kb_control = KeyboardControl(self.hud, self.display)


    def tick(self):
        self.clock.tick_busy_loop(0)
        if self.use_kb_control: self.kb_control.parse_events(self.display, self.clock)
        self.hud.tick(self.clock)
        pygame.display.flip()


    def destroy(self):
        pygame.quit()
