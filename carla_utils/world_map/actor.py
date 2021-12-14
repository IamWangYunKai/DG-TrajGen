from carla_utils import carla
DestroyActor = carla.command.DestroyActor

import random
import pickle
from enum import Enum


def get_actor(world, type_id, role_name):
    """
        ! warning: not suitable for multi-agent
        ! deprecated
    """
    actor_list = world.get_actors()
    for actor in actor_list:
        if actor.type_id == type_id and actor.attributes['role_name'] == role_name:
            return actor
    return None

def get_attached_actor(actor_list, actor):
    for target_actor in actor_list:
        print(target_actor.id, target_actor.parent)
    print()



def destroy_actors(core, actors):
    client = core.client
    client.apply_batch([DestroyActor(x) for x in actors])
    return



from ..basic import Data
class Role(Data):
    def dumps(self):
        return pickle.dumps(self, 0).decode()
    
    @staticmethod
    def loads(role_str):
        return pickle.loads(bytes(role_str, encoding='utf-8'))



class ScenarioRole(Enum):
    learnable = -1
    obstacle = 1
    agent = 2
    static = 3


