#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from os.path import join, dirname
sys.path.insert(0, join(dirname(__file__), './../../'))

import os
import time
import random
import argparse
import numpy as np
from datetime import datetime
import cv2
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from learning.model import ModelMixStyleWithV
from carla_utils import parse_yaml_file_unsafe
from learning.dataset import CARLADataset
from robo_utils.oxford.oxford_dataset import DIVADataset
from robo_utils.kitti.torch_dataset import BoostDataset as KittiDataset
from utils import write_params, to_device

random.seed(datetime.now())
torch.manual_seed(666)
torch.cuda.manual_seed(666)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--test_mode', type=bool, default=False, help='test model switch')
parser.add_argument('--dataset_name', type=str, default="train-mixstyle-01", help='name of the dataset')
parser.add_argument('--width', type=int, default=400, help='image width')
parser.add_argument('--height', type=int, default=200, help='image height')
parser.add_argument('--scale', type=float, default=25., help='longitudinal length')
parser.add_argument('--batch_size', type=int, default=32, help='size of the batches')
parser.add_argument('--points_num', type=int, default=16, help='points number')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='adam: weight_decay')
parser.add_argument('--lr', type=float, default=3e-4, help='adam: learning rate')
parser.add_argument('--n_cpu', type=int, default=16, help='number of cpu threads to use during batch generation')
parser.add_argument('--checkpoint_interval', type=int, default=500, help='interval between model checkpoints')
parser.add_argument('--test_interval', type=int, default=5, help='interval between model test')
parser.add_argument('--max_dist', type=float, default=25., help='max distance')
parser.add_argument('--max_speed', type=float, default=10., help='max distance')
parser.add_argument('--max_t', type=float, default=3., help='max time')

opt = parser.parse_args()
if opt.test_mode: opt.batch_size = 1
    
description = 'train cluster, with v0 loss'
log_path = 'result/log/'+opt.dataset_name+'/'+str(round(time.time()))+'/'
os.makedirs(log_path, exist_ok=True)
os.makedirs('result/saved_models/%s' % opt.dataset_name, exist_ok=True)
os.makedirs('result/output/%s' % opt.dataset_name, exist_ok=True)

write_params(log_path, parser, description)

logger = SummaryWriter(log_dir=log_path+'/train')
logger_carla = SummaryWriter(log_dir=log_path+'/carla')
logger_kitti = SummaryWriter(log_dir=log_path+'/kitti')
logger_robotcar = SummaryWriter(log_dir=log_path+'/robotcar')


model = ModelMixStyleWithV(latent_dim=64).to(device)
# model.load_state_dict(torch.load('result/saved_models/train-mix-style-01/model_1000.pth'))
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=opt.weight_decay)


param = parse_yaml_file_unsafe('../../params/param_kitti.yaml')
train_loader = DataLoader(KittiDataset(param, 'train', opt), batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)
train_samples = iter(train_loader)


param = parse_yaml_file_unsafe('../../params/param_oxford.yaml')
eval_loader = DataLoader(DIVADataset(param, mode='eval', opt=opt), batch_size=1, shuffle=False, num_workers=1)
eval_samples_robotcar = iter(eval_loader)

param = parse_yaml_file_unsafe('../../params/param_kitti.yaml')
eval_loader2 = DataLoader(KittiDataset(param, 'eval', opt), batch_size=1, shuffle=False, num_workers=1)
eval_samples_kitti = iter(eval_loader2)

eval_loader3 = DataLoader(CARLADataset(data_index=list(range(1,11)), opt=opt, dataset_path='/media/wang/DATASET/CARLA_HUMAN/town01/'), batch_size=1, shuffle=False, num_workers=1)
eval_samples_carla = iter(eval_loader3)

trajectory_criterion = torch.nn.MSELoss().to(device)


def eval_error_kitti(total_step):
    model.eval()

    batch = next(eval_samples_kitti)
    batch['t'] = batch['t'].view(-1,1)
    batch['v0_array'] = batch['v0_array'].view(-1,1)
    batch['xy'] = batch['xy'].view(-1,2)
    to_device(batch, device)

    feature = model.get_encoder(batch['img'], batch['v_0'])
    feature_dim = feature.shape[-1]
    feature = feature.unsqueeze(1)
    feature = feature.expand(1, batch['t'].shape[0], feature_dim)
    feature = feature.reshape(1 * batch['t'].shape[0], feature_dim)
    
    output = model.get_trajectory(feature, batch['v0_array'], batch['t'])

    output_xy = output[:,:2]

    real_traj = batch['xy'].data.cpu().numpy()*opt.max_dist
    fake_traj = output_xy.data.cpu().numpy()*opt.max_dist

    real_x = real_traj[:,0]
    real_y = real_traj[:,1]
    fake_x = fake_traj[:,0]
    fake_y = fake_traj[:,1]

    ex = np.mean(np.abs(fake_x-real_x))
    ey = np.mean(np.abs(fake_y-real_y))
    fde = np.hypot(fake_x - real_x, fake_y - real_y)[-1]
    ade = np.mean(np.hypot(fake_x - real_x, fake_y - real_y))
    
    logger_kitti.add_scalar('eval/ex',  ex.item(),  total_step)
    logger_kitti.add_scalar('eval/ey',  ey.item(),  total_step)
    logger_kitti.add_scalar('eval/fde', fde.item(), total_step)
    logger_kitti.add_scalar('eval/ade', ade.item(), total_step)

    model.train()

def eval_error_carla(total_step):
    model.eval()

    batch = next(eval_samples_carla)
    batch['t'] = batch['t'].view(-1,1)
    batch['v0_array'] = batch['v0_array'].view(-1,1)
    batch['xy'] = batch['xy'].view(-1,2)
    to_device(batch, device)

    feature = model.get_encoder(batch['img'], batch['v_0'])
    feature_dim = feature.shape[-1]
    feature = feature.unsqueeze(1)
    feature = feature.expand(1, batch['t'].shape[0], feature_dim)
    feature = feature.reshape(1 * batch['t'].shape[0], feature_dim)
    
    output = model.get_trajectory(feature, batch['v0_array'], batch['t'])

    output_xy = output[:,:2]

    real_traj = batch['xy'].data.cpu().numpy()*opt.max_dist
    fake_traj = output_xy.data.cpu().numpy()*opt.max_dist

    real_x = real_traj[:,0]
    real_y = real_traj[:,1]
    fake_x = fake_traj[:,0]
    fake_y = fake_traj[:,1]

    ex = np.mean(np.abs(fake_x-real_x))
    ey = np.mean(np.abs(fake_y-real_y))
    fde = np.hypot(fake_x - real_x, fake_y - real_y)[-1]
    ade = np.mean(np.hypot(fake_x - real_x, fake_y - real_y))
    
    logger_carla.add_scalar('eval/ex',  ex.item(),  total_step)
    logger_carla.add_scalar('eval/ey',  ey.item(),  total_step)
    logger_carla.add_scalar('eval/fde', fde.item(), total_step)
    logger_carla.add_scalar('eval/ade', ade.item(), total_step)

    model.train()

def eval_error_robotcar(total_step):
    model.eval()

    batch = next(eval_samples_robotcar)
    batch['t'] = batch['t'].view(-1,1)
    batch['v0_array'] = batch['v0_array'].view(-1,1)
    batch['xy'] = batch['xy'].view(-1,2)
    to_device(batch, device)

    feature = model.get_encoder(batch['img'], batch['v_0'])
    feature_dim = feature.shape[-1]
    feature = feature.unsqueeze(1)
    feature = feature.expand(1, batch['t'].shape[0], feature_dim)
    feature = feature.reshape(1 * batch['t'].shape[0], feature_dim)
    
    output = model.get_trajectory(feature, batch['v0_array'], batch['t'])

    output_xy = output[:,:2]

    real_traj = batch['xy'].data.cpu().numpy()*opt.max_dist
    fake_traj = output_xy.data.cpu().numpy()*opt.max_dist

    real_x = real_traj[:,0]
    real_y = real_traj[:,1]
    fake_x = fake_traj[:,0]
    fake_y = fake_traj[:,1]

    ex = np.mean(np.abs(fake_x-real_x))
    ey = np.mean(np.abs(fake_y-real_y))
    fde = np.hypot(fake_x - real_x, fake_y - real_y)[-1]
    ade = np.mean(np.hypot(fake_x - real_x, fake_y - real_y))
    
    logger_robotcar.add_scalar('eval/ex',  ex.item(),  total_step)
    logger_robotcar.add_scalar('eval/ey',  ey.item(),  total_step)
    logger_robotcar.add_scalar('eval/fde', fde.item(), total_step)
    logger_robotcar.add_scalar('eval/ade', ade.item(), total_step)

    model.train()


for total_steps in range(100000000):
    batch = next(train_samples)

    batch['t'] = batch['t'].view(-1,1)
    batch['t'].requires_grad = True
    batch['v0_array'] = batch['v0_array'].view(-1,1)
    to_device(batch, device)

    feature = model.get_encoder(batch['img'], batch['v_0'])
    feature_dim = feature.shape[-1]

    feature = feature.unsqueeze(1)
    feature = feature.expand(opt.batch_size, opt.points_num, feature_dim)
    feature = feature.reshape(opt.batch_size * opt.points_num, feature_dim)

    output = model.get_trajectory(feature, batch['v0_array'], batch['t'])

    output_xy = output[:,:2]

    fake_traj = output_xy.reshape(-1,opt.points_num*2)
    real_traj = batch['xy'].view(-1,opt.points_num*2)

    loss_trajectory_x = trajectory_criterion(real_traj.view(opt.batch_size, opt.points_num,2)[:,:,0]*opt.max_dist, fake_traj.view(opt.batch_size, opt.points_num,2)[:,:,0]*opt.max_dist)
    loss_trajectory_y = trajectory_criterion(real_traj.view(opt.batch_size, opt.points_num,2)[:,:,1]*opt.max_dist, fake_traj.view(opt.batch_size, opt.points_num,2)[:,:,1]*opt.max_dist)
    loss_trajectory = loss_trajectory_x + 5*loss_trajectory_y

    loss = loss_trajectory

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    logger.add_scalar('train/loss_trajectory', loss_trajectory.item(), total_steps)

    if total_steps > 0 and total_steps % opt.test_interval == 0:
        eval_error_carla(total_steps)
        eval_error_kitti(total_steps)
        eval_error_robotcar(total_steps)

    if total_steps > 0 and total_steps % opt.checkpoint_interval == 0:
        torch.save(model.state_dict(), 'result/saved_models/%s/'%(opt.dataset_name)+'model_'+str(total_steps)+'.pth')