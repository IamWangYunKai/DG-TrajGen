from carla_utils import carla
DestroyActor = carla.command.DestroyActor

import weakref
from typing import Dict

from ..system import debug
from ..basic import flatten_list
from .sensor_callback import RawCallback, DefaultCallback
from .sensor_create import create_sensor, create_sensor_command


def createSensorListMaster(core, vehicle, sensors_param_list):
    client, world = core.client, core.world
    blueprint_library = world.get_blueprint_library()
    batch = [create_sensor_command(core, vehicle, blueprint_library, config) for config in sensors_param_list]
    sensor_ids = []
    for response in client.apply_batch_sync(batch):
        if response.error: raise RuntimeError('spawn sensor failed: ' + response.error)
        else: sensor_ids.append(response.actor_id)
    sensors = world.get_actors(sensor_ids)

    sensors_master = CarlaSensorListMaster(core, vehicle)
    for sensor, config in zip(sensors, sensors_param_list):
        transform, callback = config['transform'], config.get('callback', None)
        sensors_master.append(sensor, transform, callback)
    return sensors_master


def createSensorListMasters(core, vehicles, sensors_param_lists):
    client, world = core.client, core.world
    blueprint_library = world.get_blueprint_library()
    sensors_master_dict = {vehicle.id: CarlaSensorListMaster(core, vehicle) for vehicle in vehicles}

    batch = []
    for vehicle, sensors_param_list in zip(vehicles, sensors_param_lists):
        batch.extend([create_sensor_command(core, vehicle, blueprint_library, config) for config in sensors_param_list])
    
    sensor_ids = []
    for response in client.apply_batch_sync(batch):
        if response.error: raise RuntimeError('spawn sensor failed: ' + response.error)
        else: sensor_ids.append(response.actor_id)
    sensors = world.get_actors(sensor_ids)

    sensors_param_list = flatten_list(sensors_param_lists)
    for sensor, config in zip(sensors, sensors_param_list):
        transform, callback = config['transform'], config.get('callback', None)
        sensors_master_dict[sensor.parent.id].append(sensor, transform, callback)
    return list(sensors_master_dict.values())



class CarlaSensorListMaster(object):
    def __init__(self, core, vehicle):
        self.world, self.vehicle = core.world, vehicle

        self.sensor_list = []
        self.sensor_dict: Dict[tuple, CarlaSensorMaster] = dict()

        '''camera'''
        self._cameras = []
        self.current_camera_index = 0

    def append(self, sensor, transform, callback):
        sensor_master = CarlaSensorMaster(sensor, transform, callback)
        self.sensor_list.append(sensor_master)
        self.sensor_dict[(sensor.type_id, sensor.attributes['role_name'])] = sensor_master

        if sensor_master.type_id == 'sensor.camera.rgb' \
                or sensor_master.type_id == 'sensor.camera.semantic_segmentation':
            self._cameras.append(sensor_master)

    # ## disable currently
    # def reset(self):
    #     [sensor_master.reset() for sensor_master in self.sensor_dict.values()]

    def get_camera(self):
        sensor_master = None
        try:
            sensor_master = self._cameras[self.current_camera_index]
        except IndexError:
            pass
        return sensor_master
    def toggle_camera(self):
        self.current_camera_index = (self.current_camera_index + 1) % len(self._cameras)


    def destroy(self):
        for sensor_master in self.sensor_dict.values():
            sensor_master.destroy()
    def destroy_commands(self):
        """
            Note: do not destroy vehicle in this class.
        """
        return [sensor_master.destroy_command() for sensor_master in self.sensor_dict.values()]


    def __del__(self):
        self.destroy()

    def __iter__(self):
        for sensor_master in self.sensor_dict.values():
            yield sensor_master

    def get(self, key):
        return self.__getitem__(key)

    def __getitem__(self, key):
        if key in self.sensor_dict:
            return self.sensor_dict[key]
        else:
            raise RuntimeError('[{}] No sensor called '.format(self.__class__.__name__) + str(key))
            debug(info='No sensor called '+ str(key), info_type='error')
            return None
    
    def __setitem__(self, key, value):
        if key in self.sensor_dict:
            self.sensor_dict[key] = value
            return True
        else:
            debug(info='No sensor called '+ str(key), info_type='error')
            return None


class CarlaSensorMaster(object):
    def __init__(self, sensor, transform, callback):
        self.sensor = sensor
        self.transform = transform
        self.raw_data, self.data = None, None

        self.type_id = sensor.type_id
        self.attributes = sensor.attributes

        self.frame_id = sensor.type_id.replace('.', '_') + '/' + sensor.attributes['role_name']

        weak_self = weakref.ref(self)
        if callback != None:
            self.callback = lambda data: callback(weak_self, data)
        else:
            '''default callback'''
            func_name = sensor.type_id.replace('.', '_')
            # print('[CarlaSensorMaster] ', func_name)
            func = getattr(DefaultCallback, func_name)
            self.callback = lambda data: func(weak_self, data)

        if hasattr(sensor, 'listen'):
            self.sensor.listen(self.callback)

        return

    # ## disable currently
    # def reset(self):
    #     if self.sensor.is_listening: self.sensor.stop()
    #     self.raw_data, self.data = None, None
    #     self.sensor.listen(self.callback)


    def get_transform(self):
        '''
            transform relative to parent actor
        '''
        return self.transform
    def get_world_transform(self):
        return self.sensor.get_transform()

    def get_raw_data(self):
        return self.raw_data
    def get_data(self):
        return self.data


    def destroy(self):
        if self.sensor.is_listening: self.sensor.stop()
        self.sensor.destroy()
    def destroy_command(self):
        if self.sensor.is_listening: self.sensor.stop()
        return DestroyActor(self.sensor)
