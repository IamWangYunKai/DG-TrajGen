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
import torch.nn as nn
from torch.autograd import grad
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from learning.boost_model import Discriminator, Cluster
from carla_utils import parse_yaml_file_unsafe
from robo_utils.oxford.oxford_dataset import DIVADataset
from robo_utils.kitti.torch_dataset import BoostDataset as KittiDataset
from learning.dataset import CARLADatasetDomain
from robo_utils.kitti.torch_dataset import BoostDatasetDomain2
from utils import write_params, check_shape, to_device, set_mute

random.seed(datetime.now())
torch.manual_seed(666)
torch.cuda.manual_seed(666)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--test_mode', type=bool, default=False, help='test model switch')
parser.add_argument('--dataset_name', type=str, default="train-DAL-01", help='name of the dataset')
parser.add_argument('--width', type=int, default=400, help='image width')
parser.add_argument('--height', type=int, default=200, help='image height')
parser.add_argument('--scale', type=float, default=25., help='longitudinal length')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--points_num', type=int, default=16, help='points number')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='adam: weight_decay')
parser.add_argument('--lr', type=float, default=3e-4, help='adam: learning rate')
parser.add_argument('--n_cpu', type=int, default=16, help='number of cpu threads to use during batch generation')
parser.add_argument('--checkpoint_interval', type=int, default=200, help='interval between model checkpoints')
parser.add_argument('--test_interval', type=int, default=25, help='interval between model test')
parser.add_argument('--max_dist', type=float, default=25., help='max distance')
parser.add_argument('--max_speed', type=float, default=10., help='max distance')
parser.add_argument('--max_t', type=float, default=3., help='max time')
parser.add_argument('--model_num', type=int, default=1, help='model_num')
opt = parser.parse_args()
if opt.test_mode: opt.batch_size = 1
    
description = 'train DAL'
log_path = 'result/log/'+opt.dataset_name+'/'+str(round(time.time()))+'/'
os.makedirs(log_path, exist_ok=True)
os.makedirs('result/saved_models/%s' % opt.dataset_name, exist_ok=True)
os.makedirs('result/output/%s' % opt.dataset_name, exist_ok=True)

write_params(log_path, parser, description)

logger_cluster = [
    SummaryWriter(log_dir=log_path+'model_'+str(i)) for i in range(1)
]

logger_carla = SummaryWriter(log_dir=log_path+'/carla')
logger_kitti = SummaryWriter(log_dir=log_path+'/kitti')
logger_robotcar = SummaryWriter(log_dir=log_path+'/robotcar')

cluster = Cluster(model_num=1, device=device)
# cluster.load_models('result/saved_models/train-DAL-01/', 1000)

discriminator = Discriminator(input_dim=256*2, output=1).to(device)

param = parse_yaml_file_unsafe('../../params/param_kitti.yaml')
train_loader = DataLoader(BoostDatasetDomain2(param, mode='train', opt=opt), batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)
train_samples = iter(train_loader)

param = parse_yaml_file_unsafe('../../params/param_oxford.yaml')
eval_loader = DataLoader(DIVADataset(param, mode='eval', opt=opt), batch_size=1, shuffle=False, num_workers=1)
eval_samples_robotcar = iter(eval_loader)

param = parse_yaml_file_unsafe('../../params/param_kitti.yaml')
eval_loader2 = DataLoader(KittiDataset(param, 'eval', opt), batch_size=1, shuffle=False, num_workers=1)
eval_samples_kitti = iter(eval_loader2)

eval_loader3 = DataLoader(CARLADatasetDomain(data_index=list(range(1,11)), opt=opt, dataset_path='/media/wang/DATASET/CARLA_HUMAN/town01/'), batch_size=1, shuffle=False, num_workers=1)
eval_samples_carla = iter(eval_loader3)

criterion = nn.MSELoss().to(device)

discriminator_criterion = nn.BCEWithLogitsLoss().to(device)

d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=3e-4, weight_decay=opt.weight_decay)

def show_traj_with_uncertainty(fake_traj, real_traj, step, model_index, logvar=None):
    fake_xy = fake_traj
    x = fake_xy[:,0]*opt.max_dist
    y = fake_xy[:,1]*opt.max_dist

    real_xy = real_traj
    real_x = real_xy[:,0]*opt.max_dist
    real_y = real_xy[:,1]*opt.max_dist

    max_x = 30.
    max_y = 30.

    fig = plt.figure(figsize=(7, 7))
    ax1 = fig.add_subplot(111)

    ax1.plot(x, y, label='trajectory', color = 'r', linewidth=5)

    ax1.plot(real_x, real_y, label='real-trajectory', color = 'b', linewidth=5, linestyle='--')
    ax1.set_xlabel('Forward/(m)')
    ax1.set_ylabel('Sideways/(m)')  
    ax1.set_xlim([0., max_x])
    ax1.set_ylim([-max_y/2, max_y/2])
    plt.legend(loc='lower right')
    
    plt.legend(loc='lower left')
    plt.savefig('result/output/%s/' % opt.dataset_name+str(step)+'_'+str(model_index)+'_curve.png')
    plt.close('all')


def eval_error_kitti(total_step):
    batch = next(eval_samples_kitti)
    to_device(batch, device)
    batch['t'] = batch['t'].view(-1,1)
    batch['v0_array'] = batch['v0_array'].view(-1,1)

    feature = cluster.get_encoder(batch['img'], 0)
    feature_dim = feature.shape[-1]
    feature = feature.unsqueeze(1)
    feature = feature.expand(1, batch['t'].shape[0], feature_dim)
    feature = feature.reshape(1 * batch['t'].shape[0], feature_dim)
    
    output = cluster.get_trajectory(feature, batch['v0_array'], batch['t'], 0)

    output_xy = output[:,:2]

    real_traj = batch['xy'][0].data.cpu().numpy()*opt.max_dist
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

def eval_error_carla(total_step):
    batch = next(eval_samples_carla)
    to_device(batch, device)
    batch['t'] = batch['t'].view(-1,1)
    batch['v0_array'] = batch['v0_array'].view(-1,1)

    feature = cluster.get_encoder(batch['img'], 0)
    feature_dim = feature.shape[-1]
    feature = feature.unsqueeze(1)

    feature = feature.expand(1, batch['t'].shape[0], feature_dim)
    feature = feature.reshape(1 * batch['t'].shape[0], feature_dim)
    
    output = cluster.get_trajectory(feature, batch['v0_array'], batch['t'], 0)

    output_xy = output[:,:2]

    real_traj = batch['xy'][0].data.cpu().numpy()*opt.max_dist
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

def eval_error_robotcar(total_step):
    batch = next(eval_samples_robotcar)
    to_device(batch, device)

    batch['t'] = batch['t'].view(-1,1)
    batch['v0_array'] = batch['v0_array'].view(-1,1)

    feature = cluster.get_encoder(batch['img'], 0)
    feature_dim = feature.shape[-1]
    feature = feature.unsqueeze(1)
    feature = feature.expand(1, batch['t'].shape[0], feature_dim)
    feature = feature.reshape(1 * batch['t'].shape[0], feature_dim)
    
    output = cluster.get_trajectory(feature, batch['v0_array'], batch['t'], 0)

    output_xy = output[:,:2]

    real_traj = batch['xy'][0].data.cpu().numpy()*opt.max_dist
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


for total_steps in range(1000000):
    model_index = 0
    logger = logger_cluster[0]

    batch = next(train_samples)
    check_shape(batch)
    to_device(batch, device)

    real_traj = batch['xy'].view(-1, 10*2)

    mask1 = torch.arange(0,opt.batch_size-1,2)
    mask2 = torch.arange(1,opt.batch_size,2)
    labels = 1- 0.5*torch.sum(torch.abs(batch['domian'][mask1] - batch['domian'][mask2]), dim=1)

    batch['t'] = batch['t'].view(-1,1)
    batch['t'].requires_grad = True
    batch['v0_array'] = batch['v0_array'].view(-1,1)

    check_shape(batch)
    to_device(batch, device)

    feature = cluster.get_encoder(batch['img'], 0)
    feature_dim = feature.shape[-1]
    input_disc = torch.cat([feature[mask1], feature[mask2]], dim=1)
    check_shape(input_disc)
    prediction = discriminator(input_disc)
    check_shape(prediction)

    check_shape(prediction, 'prediction')
    check_shape(labels, 'labels')

    discriminator_loss = discriminator_criterion(prediction.flatten(), labels)
    g_loss = - discriminator_loss

    logger.add_scalar('train/g_loss', g_loss.item(), total_steps)

    feature = feature.unsqueeze(1)
    feature = feature.expand(opt.batch_size, 10, feature_dim)
    feature = feature.reshape(opt.batch_size * 10, feature_dim)
    check_shape(feature, 'feature')

    output_xy = cluster.get_trajectory(feature, batch['v0_array'], batch['t'], 0)
    fake_traj = output_xy.reshape(-1, 10*2)
    check_shape(output_xy, 'output_xy')

    loss_trajectory_x = criterion(real_traj.view(opt.batch_size, 10,2)[:,:,0]*opt.max_dist, fake_traj.view(opt.batch_size, 10,2)[:,:,0]*opt.max_dist)
    loss_trajectory_y = criterion(real_traj.view(opt.batch_size, 10,2)[:,:,1]*opt.max_dist, fake_traj.view(opt.batch_size, 10,2)[:,:,1]*opt.max_dist)
    loss_trajectory = (loss_trajectory_x + 5*loss_trajectory_y)
    # total loss
    loss = loss_trajectory

    check_shape(loss, 'loss')
    cluster.encoder_optimizer[0].zero_grad()
    cluster.trajectory_model_optimizer[0].zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(cluster.parameters(), clip_value=1)
    cluster.encoder_optimizer[0].step()
    cluster.trajectory_model_optimizer[0].step()
    set_mute(True)

    train_step = 1
    if total_steps % train_step == 0:
        feature = cluster.get_encoder(batch['img'], 0)
        input_disc = torch.cat([feature[mask1], feature[mask2]], dim=1)
        prediction = discriminator(input_disc.detach())
        discriminator_loss = discriminator_criterion(prediction.flatten(), labels)
        discriminator.zero_grad()
        discriminator_loss.backward()
        d_optimizer.step()
        logger.add_scalar('train/discriminator_loss', discriminator_loss.item(), total_steps)

    logger.add_scalar('train/loss', loss.item(), total_steps)


    if total_steps > 0 and total_steps % opt.test_interval == 0:
        eval_error_carla(total_steps)
        eval_error_kitti(total_steps)
        eval_error_robotcar(total_steps)

    if total_steps > 0 and total_steps % opt.checkpoint_interval == 0:
        cluster.save_models('result/saved_models/%s/'%(opt.dataset_name), total_steps)
        torch.save(discriminator.state_dict(), 'result/saved_models/%s/'%(opt.dataset_name)+'/%s/discriminator.pth'%total_steps)