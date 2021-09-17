#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from os.path import join, dirname
sys.path.insert(0, join(dirname(__file__), './../../'))

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import os
import time
import random
import argparse
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})

from carla_utils import parse_yaml_file_unsafe
from robo_utils.oxford.oxford_dataset import DIMDataset
from robo_utils.kitti.torch_dataset import BoostDatasetDomain
from robo_utils.kitti.torch_dataset import BoostDataset as KittiDataset
from learning.dataset import CARLADatasetDomain

from utils import write_params, check_shape, to_device, set_mute
from dim_model import ImitativeModel

random.seed(datetime.now())
torch.manual_seed(666)
torch.cuda.manual_seed(666)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--test_mode', type=bool, default=False, help='test model switch')
parser.add_argument('--dataset_name', type=str, default="train-RIP-01", help='name of the dataset')
parser.add_argument('--width', type=int, default=400, help='image width')
parser.add_argument('--height', type=int, default=200, help='image height')
parser.add_argument('--scale', type=float, default=25., help='longitudinal length')
parser.add_argument('--batch_size', type=int, default=32, help='size of the batches')
parser.add_argument('--points_num', type=int, default=4, help='points number')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='adam: weight_decay')
parser.add_argument('--lr', type=float, default=3e-3, help='adam: learning rate')
parser.add_argument('--n_cpu', type=int, default=16, help='number of cpu threads to use during batch generation')
parser.add_argument('--checkpoint_interval', type=int, default=1000, help='interval between model checkpoints')
parser.add_argument('--test_interval', type=int, default=50, help='interval between model test')
parser.add_argument('--max_dist', type=float, default=25., help='max distance')
parser.add_argument('--max_speed', type=float, default=10., help='max distance')
parser.add_argument('--max_t', type=float, default=3., help='max time')
parser.add_argument('--model_num', type=int, default=5, help='model_num')
opt = parser.parse_args()

description = 'train RIP'
log_path = 'result/log/'+opt.dataset_name+'/'+str(round(time.time()))+'/'
os.makedirs(log_path, exist_ok=True)
os.makedirs('result/saved_models/%s' % opt.dataset_name, exist_ok=True)
os.makedirs('result/output/%s' % opt.dataset_name, exist_ok=True)

write_params(log_path, parser, description)

def load_models(dataset_name, total_steps):
    for model_index in range(opt.model_num):
        model = model_cluster[model_index]
        model.load_state_dict(torch.load('result/saved_models/%s/%s/model_%d.pth'%(dataset_name, total_steps, model_index)))
        
model_cluster = [ImitativeModel(output_shape=(opt.points_num, 2)).to(device) for _ in range(opt.model_num)]
# load_models('train-kitti-01', 100000)

param = parse_yaml_file_unsafe('../../params/param_kitti.yaml')
train_loader_cluster = [
    iter(DataLoader(BoostDatasetDomain(param, mode='train', opt=opt, data_index=i), batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)) \
        for i in range(opt.model_num)
]

param = parse_yaml_file_unsafe('../../params/param_oxford.yaml')
eval_loader = DataLoader(DIMDataset(param, mode='eval', opt=opt), batch_size=1, shuffle=False, num_workers=1)
eval_samples_robotcar = iter(eval_loader)

param = parse_yaml_file_unsafe('../../params/param_kitti.yaml')
eval_loader2 = DataLoader(KittiDataset(param, 'eval', opt), batch_size=1, shuffle=False, num_workers=1)
eval_samples_kitti = iter(eval_loader2)

eval_loader3 = DataLoader(CARLADatasetDomain(data_index=list(range(1,11)), opt=opt, dataset_path='/media/wang/DATASET/CARLA_HUMAN/town01/'), batch_size=1, shuffle=False, num_workers=1)
eval_samples_carla = iter(eval_loader3)

logger_cluster = [
    SummaryWriter(log_dir=log_path+'model_'+str(i)) for i in range(opt.model_num)
]
optimizer_cluster = [torch.optim.Adam(model_cluster[i].parameters(), lr=opt.lr, weight_decay=opt.weight_decay) for i in range(opt.model_num)]

logger_robotcar = SummaryWriter(log_dir=log_path+'/robotcar')
logger_kitti = SummaryWriter(log_dir=log_path+'/kitti')
logger_carla = SummaryWriter(log_dir=log_path+'/carla')

def interpolation(time_list, x_list, y_list, t):
    if t >= time_list[-1]:
        dt = time_list[-1] - time_list[-2]
        dx = x_list[-1] - x_list[-2]
        dy = y_list[-1] - y_list[-2]
        x = x_list[-1] + (dx/dt)*(t - time_list[-1])
        y = y_list[-1] + (dy/dt)*(t - time_list[-1])
    else:
        index = np.argmin(np.abs(time_list-t))
        if t >= time_list[index]:
            dt = time_list[index+1] - time_list[index]
            dx = x_list[index+1] - x_list[index]
            dy = y_list[index+1] - y_list[index]
            x = x_list[index] + (dx/dt)*(t - time_list[index])
            y = y_list[index] + (dy/dt)*(t - time_list[index])
        else:
            dt = time_list[index] - time_list[index-1]
            dx = x_list[index] - x_list[index-1]
            dy = y_list[index] - y_list[index-1]
            x = x_list[index-1] + (dx/dt)*(t - time_list[index-1])
            y = y_list[index-1] + (dy/dt)*(t - time_list[index-1])
    return x, y

def eval_decision(logger, total_steps, eval_samples, algorithm='WCM'):
    batch = next(eval_samples)
    to_device(batch, device)
    check_shape(batch)
    set_mute(True)

    x = model_cluster[0]._decoder._base_dist.mean.clone().detach().view(-1, opt.points_num, 2)
    x.requires_grad = True
    zs = [model._params(
        velocity=batch['v_0'].view(-1,1),
        visual_features=batch['img'],
    ) for model in model_cluster]

    optimizer = torch.optim.Adam(params=[x], lr=0.5)

    x_best = x.clone()
    loss_best = torch.ones(()).to(x.device) * 1000.0
    for _ in range(50):
        optimizer.zero_grad()
        y, _ = model_cluster[0]._decoder._forward(x=x, z=zs[0])

        imitation_posteriors = list()
        for model, z in zip(model_cluster, zs):
            _, log_prob, logabsdet = model._decoder._inverse(y=y, z=z)
            imitation_prior = torch.mean(log_prob - logabsdet)
            imitation_posteriors.append(imitation_prior)

        imitation_posteriors = torch.stack(imitation_posteriors, dim=0)

        if algorithm == "WCM":
            loss, _ = torch.min(-imitation_posteriors, dim=0)
        elif algorithm == "BCM":
            loss, _ = torch.max(-imitation_posteriors, dim=0)
        else:
            loss = torch.mean(-imitation_posteriors, dim=0)

        loss.backward(retain_graph=True)
        optimizer.step()
        if loss < loss_best:
            x_best = x.clone()
            loss_best = loss.clone()

    plan, _ = model_cluster[0]._decoder._forward(x=x_best, z=zs[0])
    xy = plan.detach().cpu().numpy()[0]*opt.max_dist
    real_xy = batch['xy'].view(-1, 2).data.cpu().numpy()*opt.max_dist

    fake_x = xy[:,0]
    fake_y = xy[:,1]

    real_x = real_xy[:,0]
    real_y = real_xy[:,1]

    time = batch['t'].data.cpu().numpy()[0]*opt.max_t
    # time_list = [0.0, 0.75, 1.5, 2.25] #oxford training time
    time_list = [0.7300, 1.4500, 2.1801, 2.9100] #kitti training time
    xs = []
    ys = []
    for t in time:
        x, y = interpolation(time_list, fake_x, fake_y, t)
        xs.append(x)
        ys.append(y)
    fake_x = np.array(xs)
    fake_y = np.array(ys)

    ex = np.mean(np.abs(fake_x-real_x))
    ey = np.mean(np.abs(fake_y-real_y))

    fde = np.hypot(fake_x - real_x, fake_y - real_y)[-1]
    ade = np.mean(np.hypot(fake_x - real_x, fake_y - real_y))

    logger.add_scalar('eval/ex',  ex.item(),  total_steps)
    logger.add_scalar('eval/ey',  ey.item(),  total_steps)
    logger.add_scalar('eval/fde', fde.item(), total_steps)
    logger.add_scalar('eval/ade', ade.item(), total_steps)

for total_steps in range(10000000):
    for model_index in range(opt.model_num):
        model = model_cluster[model_index]
        train_loader = train_loader_cluster[model_index]
        optimizer = optimizer_cluster[model_index]
        logger = logger_cluster[model_index]

        batch = next(train_loader)
        check_shape(batch)
        to_device(batch, device)

        y = batch['xy']

        z = model._params(
            velocity=batch['v_0'].view(-1,1),
            visual_features=batch['img'],
        )
        _, log_prob, logabsdet = model._decoder._inverse(y=y, z=z)
        check_shape(log_prob, 'log_prob')
        check_shape(logabsdet, 'logabsdet')
        loss = -torch.mean(log_prob - logabsdet, dim=0)
        optimizer.zero_grad()
        loss.backward()
        logger.add_scalar('train/loss',  loss.item(),  total_steps)
        torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
        optimizer.step()
        set_mute(True)

    if total_steps > 0 and total_steps % opt.test_interval == 0:
        eval_decision(logger_robotcar, total_steps, eval_samples_robotcar, algorithm='WCM')
        eval_decision(logger_kitti,    total_steps, eval_samples_kitti,    algorithm='WCM')
        eval_decision(logger_carla,    total_steps, eval_samples_carla,    algorithm='WCM')

    if total_steps > 0 and total_steps % opt.checkpoint_interval == 0:
        for model_index in range(opt.model_num):
            model = model_cluster[model_index]
            os.makedirs('result/saved_models/%s/%s'%(opt.dataset_name, total_steps), exist_ok=True)
            torch.save(model.state_dict(), 'result/saved_models/%s/%s/model_%d.pth'%(opt.dataset_name, total_steps, model_index))