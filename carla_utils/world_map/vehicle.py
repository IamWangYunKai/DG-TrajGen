from carla_utils import carla
SpawnActor = carla.command.SpawnActor
SetAutopilot = carla.command.SetAutopilot
FutureActor = carla.command.FutureActor

import random

from .core import Core
from .tools import get_spawn_transform, get_random_spawn_transform, get_waypoint
from .actor import Role, ScenarioRole


def vehicle_frame_id(vehicle):
    vfid = vehicle.type_id.replace('.', '_') + '_'
    try:
        role = Role.loads(vehicle.attributes['role_name'])
        vfid += role.atype + '_' + str(role.vi)
    except:
        vfid += vehicle.attributes['role_name'] + '_' + str(vehicle.id)
    return vfid


def add_vehicle(core: Core, enable_physics, spawn_point, type_id='vehicle.bmw.grandtourer', **attributes):
    """
    
    
    Args:
        attributes: contains role_name, color
    
    Returns:
        carla.Vehicle
    """

    world, town_map = core.world, core.town_map

    delta_z = get_spawn_delta_z(town_map)

    bp = create_blueprint(core, type_id, **attributes)

    spawn_transform = get_random_spawn_transform(town_map)
    if spawn_point:
        spawn_transform = get_spawn_transform(core, spawn_point, height=delta_z)

    vehicle = None
    while vehicle == None:
        vehicle = world.try_spawn_actor(bp, spawn_transform)

        if core.mode == 'real':  ## a bug in real mode
            import time
            time.sleep(0.1)

        waypoint = get_waypoint(town_map, spawn_transform)
        spawn_transform = random.choice(waypoint.next(2.0)).transform
        spawn_transform.location.z += delta_z

    vehicle.set_simulate_physics(enable_physics)
    core.tick()

    # print('spawn_point: x={}, y={}'.format(vehicle.get_location().x, vehicle.get_location().y))
    return vehicle


def add_vehicles(core: Core, enable_physics, spawn_points, type_ids, **attributes):
    """
    
    
    Args:
        attributes: contains role_names, colors
    
    Returns:
        list of carla.Vehicle
    """

    client, world, traffic_manager = core.client, core.world, core.traffic_manager

    delta_z = get_spawn_delta_z(core.town_map)

    number = len(spawn_points)
    role_names = attributes.get('role_names', [Role(name='hero', atype='obstacle')]*number)
    colors = attributes.get('colors', [(255,255,255)]*number)
    bps = [create_blueprint(core, type_ids[i], role_name=role_names[i], color=colors[i]) for i in range(number)]

    batch = []
    for i, (bp, spawn_point) in enumerate(zip(bps, spawn_points)):
        spawn_transform = get_spawn_transform(core, spawn_point, height=delta_z)
        if role_names[i].atype == ScenarioRole.obstacle:
            cmd = SpawnActor(bp, spawn_transform).then(SetAutopilot(FutureActor, True, traffic_manager.get_port()))
        else:
            cmd = SpawnActor(bp, spawn_transform)
        batch.append(cmd)
    
    actor_ids = []
    for response in client.apply_batch_sync(batch):
        # if response.error: raise RuntimeError('spawn vehicles failed: ' + response.error)
        if response.error: pass  # print('Warning: spawn vehicles failed: ' + response.error)
        else: actor_ids.append(response.actor_id)
    vehicles = world.get_actors(actor_ids)
    for vehicle in vehicles: vehicle.set_simulate_physics(enable_physics)

    core.tick()
    return [v for v in vehicles]


def get_spawn_delta_z(town_map):
    delta_z = 0.2
    if town_map.name == 'Town02':
        delta_z += 0.3
    return delta_z




def create_blueprint(core, type_id, **attributes):
    world = core.world
    blueprint_lib = world.get_blueprint_library()

    blueprint_lib = blueprint_lib.filter(type_id)
    blueprint_lib = [x for x in blueprint_lib if int(x.get_attribute('number_of_wheels')) == 4]
    bp = random.choice(blueprint_lib)

    role_name: Role = attributes.get('role_name', Role(name='hero'))
    bp.set_attribute('role_name', role_name.dumps())

    if bp.has_attribute('color'):
        color = attributes.get('color', None)
        if color == None:
            color = random.choice(bp.get_attribute('color').recommended_values)
        else:
            color = str(color[0]) + ',' + str(color[1]) + ',' + str(color[2])
        bp.set_attribute('color', color)
    if bp.has_attribute('driver_id'):
        driver_id = random.choice(bp.get_attribute('driver_id').recommended_values)
        bp.set_attribute('driver_id', driver_id)
    if bp.has_attribute('is_invincible'):
        bp.set_attribute('is_invincible', 'true')
    return bp


