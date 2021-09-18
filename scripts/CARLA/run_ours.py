#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from os.path import join, dirname
sys.path.insert(0, join(dirname(__file__), '../../'))

import simulator
simulator.load('/home/wang/CARLA_0.9.9.4')
import carla
sys.path.append('/home/wang/CARLA_0.9.9.4/PythonAPI/carla')
from agents.navigation.basic_agent import BasicAgent

from simulator import config, set_weather, add_vehicle
from simulator.sensor_manager import SensorManager
from utils.navigator_sim import get_map, get_nav, replan, close2dest
from utils import add_alpha_channel

from controller import CapacController, getActorState

import os
import cv2
import time
import copy
import threading
import random
import argparse
import numpy as np
from PIL import Image
from datetime import datetime
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
plt.ion()

import torch
from torch.autograd import grad
import torchvision.transforms as transforms

from learning.model import Generator, EncoderWithV

global_img = None
global_nav = None
global_v0 = 0.
global_vel = 0.
global_plan_time = 0.
global_trajectory = None
start_control = False
global_vehicle = None
global_plan_map = None

global_transform = None
max_steer_angle = 0.
global_view_img = None
state0 = None

global_collision = False
global_cnt = 0

MAX_SPEED = 30
img_height = 200
img_width = 400

speed_list = []

random.seed(datetime.now())
torch.manual_seed(999)
torch.cuda.manual_seed(999)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Params')
parser.add_argument('-d', '--data', type=int, default=1, help='data index')
parser.add_argument('-s', '--save', type=bool, default=False, help='save result')
parser.add_argument('--width', type=int, default=400, help='image width')
parser.add_argument('--height', type=int, default=200, help='image height')
parser.add_argument('--max_dist', type=float, default=25., help='max distance')
parser.add_argument('--max_t', type=float, default=3., help='max time')
parser.add_argument('--vector_dim', type=int, default=64, help='vector dim')
parser.add_argument('--max_speed', type=float, default=10., help='max speed')
parser.add_argument('--scale', type=float, default=25., help='longitudinal length')
parser.add_argument('--dt', type=float, default=0.05, help='discretization minimum time interval')
parser.add_argument('--rnn_steps', type=int, default=10, help='rnn readout steps')
args = parser.parse_args()

data_index = args.data
save_path = '/media/wang/DATASET/CARLA/town01/'+str(data_index)+'/'

encoder = EncoderWithV(input_dim=6, out_dim=args.vector_dim).to(device)
encoder.load_state_dict(torch.load('encoder.pth'))
encoder.eval()
generator = Generator(input_dim=1+1+args.vector_dim, output=2).to(device)
generator.load_state_dict(torch.load('generator.pth'))
generator.eval()


img_transforms = [
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
img_trans = transforms.Compose(img_transforms)


def mkdir(path):
    os.makedirs(save_path+path, exist_ok=True)

def image_callback(data):
    global state0, global_img, global_plan_time, global_vehicle, global_plan_map,global_nav, global_transform, global_v0
    global_plan_time = time.time()
    global_transform = global_vehicle.get_transform()

    state0 = getActorState('odom', global_plan_time, global_vehicle)
    state0.x = global_transform.location.x
    state0.y = global_transform.location.y
    state0.z = global_transform.location.z
    state0.theta = np.deg2rad(global_transform.rotation.yaw)

    array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8")) 
    array = np.reshape(array, (data.height, data.width, 4)) # RGBA format
    global_img = array

    global_nav = get_nav(global_vehicle, global_plan_map)

    v = global_vehicle.get_velocity()
    global_v0 = np.sqrt(v.x**2+v.y**2)


def view_image_callback(data):
    global global_view_img
    array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8")) 
    array = np.reshape(array, (data.height, data.width, 4)) # RGBA format
    global_view_img = array

def collision_callback(data):
    global global_collision
    global_collision = True


def visualize(input_img, nav):
    global global_vel, global_cnt
    global_cnt += 1
    img = copy.deepcopy(input_img)
    text = "speed: "+str(round(3.6*global_vel, 1))+' km/h'
    cv2.putText(img, text, (20, 30), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255), 2)
    new_nav = add_alpha_channel(nav)
    new_nav = cv2.flip(new_nav, 1)
    img[:nav.shape[0],-nav.shape[1]:] = new_nav
    
    # if global_cnt % 2 == 0: cv2.imwrite('video/ours/WetCloudySunset/'+str(global_cnt)+'.png', copy.deepcopy(img))
    cv2.imshow('Visualization', img)
    cv2.waitKey(5)

def get_traj(plan_time, global_img, global_nav):
    global global_v0, draw_cost_map, state0, global_vehicle    
    t = torch.arange(0, 0.99, args.dt).unsqueeze(1).to(device)
    t.requires_grad = True
    points_num = len(t)

    v = global_v0 if global_v0 > 4 else 4
    v_0 = torch.FloatTensor([v/args.max_speed]).unsqueeze(1)
    v_0 = v_0.to(device)
    condition = torch.FloatTensor([v/args.max_speed]*points_num).view(-1, 1)
    condition = condition.to(device)

    img = Image.fromarray(cv2.cvtColor(global_img,cv2.COLOR_BGR2RGB))
    nav = Image.fromarray(cv2.cvtColor(global_nav,cv2.COLOR_BGR2RGB))
    img = img_trans(img)
    nav = img_trans(nav)
    input_img = torch.cat((img, nav), 0).unsqueeze(0).to(device)

    single_latent = encoder(input_img, v_0)
    single_latent = single_latent.unsqueeze(1)
    latent = single_latent.expand(1, points_num, single_latent.shape[-1])
    latent = latent.reshape(1 * points_num, single_latent.shape[-1])

    output = generator(condition, latent, t)

    vx = grad(output[:,0].sum(), t, create_graph=True)[0][:,0]*(args.max_dist/args.max_t)
    vy = grad(output[:,1].sum(), t, create_graph=True)[0][:,0]*(args.max_dist/args.max_t)
    
    ax = grad(vx.sum(), t, create_graph=True)[0][:,0]/args.max_t
    ay = grad(vy.sum(), t, create_graph=True)[0][:,0]/args.max_t

    output_axy = torch.cat([ax.unsqueeze(1), ay.unsqueeze(1)], dim=1)

    x = output[:,0]*args.max_dist
    y = output[:,1]*args.max_dist

    theta_a = torch.atan2(ay, ax)
    theta_v = torch.atan2(vy, vx)
    sign = torch.sign(torch.cos(theta_a-theta_v))
    a = torch.mul(torch.norm(output_axy, dim=1), sign.flatten()).unsqueeze(1)

    
    vx = vx.data.cpu().numpy()
    vy = vy.data.cpu().numpy()
    x = x.data.cpu().numpy()
    y = y.data.cpu().numpy()
    ax = ax.data.cpu().numpy()
    ay = ay.data.cpu().numpy()
    a = a.data.cpu().numpy()

    trajectory = {'time':plan_time, 'x':x, 'y':y, 'vx':vx, 'vy':vy, 'ax':ax, 'ay':ay, 'a':a}
    return trajectory

def make_plan():
    global global_img, global_nav, global_pcd, global_plan_time, global_trajectory,start_control
    while True:
        global_trajectory = get_traj(global_plan_time, global_img, global_nav)
        if not start_control:
            start_control = True
    
def main():
    global global_nav, global_vel, start_control, global_plan_map, global_vehicle, global_transform, max_steer_angle, global_a, state0, global_collision, global_view_img
    client = carla.Client(config['host'], config['port'])
    client.set_timeout(config['timeout'])
    
    world = client.load_world('Town01')

    weather = carla.WeatherParameters(
        cloudiness= 0,
        precipitation=0,
        sun_altitude_angle= 45,
        fog_density = 100,
        fog_distance = 0,
        fog_falloff = 0,
    )
    set_weather(world, weather)
    # world.set_weather(carla.WeatherParameters.HardRainSunset)
    # world.set_weather(carla.WeatherParameters.WetCloudySunset)
    # world.set_weather(carla.WeatherParameters.ClearNoon)
    
    blueprint = world.get_blueprint_library()
    world_map = world.get_map()
    
    vehicle = add_vehicle(world, blueprint, vehicle_type='vehicle.audi.a2')
    global_vehicle = vehicle
    # Enables or disables the simulation of physics on this actor.
    vehicle.set_simulate_physics(True)
    physics_control = vehicle.get_physics_control()
    max_steer_angle = np.deg2rad(physics_control.wheels[0].max_steer_angle)


    spawn_points = world_map.get_spawn_points()
    waypoint_tuple_list = world_map.get_topology()
    origin_map = get_map(waypoint_tuple_list)

    agent = BasicAgent(vehicle, target_speed=MAX_SPEED)

    # prepare map
    destination = carla.Transform()
    destination.location = world.get_random_location_from_navigation()
    global_plan_map, destination = replan(agent, destination, copy.deepcopy(origin_map), spawn_points)

    sensor_dict = {
        'camera':{
            'transform':carla.Transform(carla.Location(x=0.5, y=0.0, z=2.5)),
            'callback':image_callback,
            },
        'camera:view':{
            'transform':carla.Transform(carla.Location(x=-3.0, y=0.0, z=6.0), carla.Rotation(pitch=-45)),
            'callback':view_image_callback,
            },
        'collision':{
            'transform':carla.Transform(carla.Location(x=0.5, y=0.0, z=2.5)),
            'callback':collision_callback,
            },
        }

    sm = SensorManager(world, blueprint, vehicle, sensor_dict)
    sm.init_all()
    
    
    # start to plan
    plan_thread = threading.Thread(target = make_plan, args=())

    while True:
        if (global_img is not None) and (global_nav is not None):
            plan_thread.start()
            break
        else:
            time.sleep(0.001)
    
    # wait for the first plan result
    while not start_control:
        time.sleep(0.001)

    print('Start to control')
    
    success_cnt = 0
    fail_cnt = 0
    ctrller = CapacController(world, vehicle, MAX_SPEED)
    while True:
        # change destination
        if close2dest(vehicle, destination):
            success_cnt += 1
            print('Success:', success_cnt, '\tFail:', fail_cnt, '\t', 100*(success_cnt)/(success_cnt+fail_cnt))
            print('Avg speed', sum(speed_list)/len(speed_list))
            #destination = get_random_destination(spawn_points)
            print('get destination !', time.time())
            destination = carla.Transform()
            destination.location = world.get_random_location_from_navigation()
            global_plan_map, destination = replan(agent, destination, copy.deepcopy(origin_map), spawn_points)

        if global_collision:
            fail_cnt += 1
            print('Success:', success_cnt, '\tFail:', fail_cnt, '\t', 100*(success_cnt)/(success_cnt+fail_cnt))
            print('Avg speed', sum(speed_list)/len(speed_list))
            cv2.imwrite('img_log/'+str(time.time())+'.png', copy.deepcopy(global_view_img))
            
            start_point = random.choice(spawn_points)
            vehicle.set_transform(start_point)
            global_plan_map, destination = replan(agent, destination, copy.deepcopy(origin_map), spawn_points)

            start_waypoint = agent._map.get_waypoint(agent._vehicle.get_location())
            end_waypoint = agent._map.get_waypoint(destination.location)

            route_trace = agent._trace_route(start_waypoint, end_waypoint)
            start_point.rotation = route_trace[0][0].transform.rotation
            vehicle.set_transform(start_point)
            time.sleep(0.1)
            global_collision = False

        v = global_vehicle.get_velocity()
        a = global_vehicle.get_acceleration()
        global_vel = np.sqrt(v.x**2+v.y**2+v.z**2)
        global_a = np.sqrt(a.x**2+a.y**2+a.z**2)

        control_time = time.time()
        dt = control_time - global_trajectory['time']
        index = int((dt/args.max_t)//args.dt) + 2
        if index > 0.99/args.dt:
            continue
    
        control = ctrller.run_step(global_trajectory, index, state0)
        vehicle.apply_control(control)

        speed_list.append(global_v0)
        visualize(global_view_img, global_nav
        #time.sleep(1/60.)

    cv2.destroyAllWindows()
    sm.close_all()
    vehicle.destroy()
        
if __name__ == '__main__':
    main()