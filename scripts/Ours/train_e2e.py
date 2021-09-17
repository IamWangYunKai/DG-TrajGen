#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from os.path import join, dirname
sys.path.insert(0, join(dirname(__file__), '../../'))
import os
import random
import argparse
import numpy as np
from datetime import datetime
import cv2
import seaborn as sns

import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from learning.model import Generator, EncoderWithV
from robo_utils.oxford.oxford_dataset import DIVADataset
from robo_utils.kitti.torch_dataset import OurDataset as KittiDataset
from learning.dataset import CARLADataset
from utils import write_params, to_device
from carla_utils import parse_yaml_file_unsafe
from torchvision.utils import save_image

random.seed(datetime.now())
torch.manual_seed(666)
torch.cuda.manual_seed(666)
torch.set_num_threads(16)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--test_mode', type=bool, default=False, help='test model switch')
parser.add_argument('--dataset_name', type=str, default="train-ours-01", help='name of the dataset')
parser.add_argument('--width', type=int, default=400, help='image width')
parser.add_argument('--height', type=int, default=200, help='image height')
parser.add_argument('--scale', type=float, default=30., help='longitudinal length')
parser.add_argument('--batch_size', type=int, default=28, help='size of the batches')
parser.add_argument('--vector_dim', type=int, default=64, help='vector dim')
parser.add_argument('--points_num', type=int, default=16, help='points number')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='adam: weight_decay')
parser.add_argument('--lr', type=float, default=1e-5, help='adam: learning rate')
parser.add_argument('--n_cpu', type=int, default=14, help='number of cpu threads to use during batch generation')
parser.add_argument('--checkpoint_interval', type=int, default=1000, help='interval between model checkpoints')
parser.add_argument('--test_interval', type=int, default=5, help='interval between model test')
parser.add_argument('--max_dist', type=float, default=25., help='max distance')
parser.add_argument('--max_speed', type=float, default=10., help='max speed')
parser.add_argument('--max_t', type=float, default=3., help='max time')
opt = parser.parse_args()
if opt.test_mode: opt.batch_size = 1
    
description = 'dropout'
log_path = 'result/log/'+opt.dataset_name+'/'
os.makedirs('result/saved_models/%s' % opt.dataset_name, exist_ok=True)
os.makedirs('result/output/%s' % opt.dataset_name, exist_ok=True)

if not opt.test_mode:
    logger = SummaryWriter(log_dir=log_path+'/kitti')
    logger2 = SummaryWriter(log_dir=log_path+'/robotcar')
    logger3 = SummaryWriter(log_dir=log_path+'/carla')
    write_params(log_path, parser, description)


encoder = EncoderWithV(input_dim=6, out_dim=opt.vector_dim).to(device)
generator = Generator(input_dim=1+1+opt.vector_dim, output=2).to(device)
# generator.load_state_dict(torch.load('result/saved_models/train-gan-01/generator_10000.pth'))

trajectory_criterion = torch.nn.MSELoss().to(device)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)


param = parse_yaml_file_unsafe('../../params/param_oxford.yaml')
train_loader = DataLoader(DIVADataset(param, mode='train', opt=opt), batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)
train_samples = iter(train_loader)

param = parse_yaml_file_unsafe('../../params/param_kitti.yaml')
eval_loader = DataLoader(KittiDataset(param, mode='eval', opt=opt), batch_size=1, shuffle=False, num_workers=1)
eval_samples = iter(eval_loader)

param = parse_yaml_file_unsafe('../../params/param_oxford.yaml')
eval_loader2 = DataLoader(DIVADataset(param, mode='eval', opt=opt), batch_size=1, shuffle=False, num_workers=1)
eval_samples2 = iter(eval_loader2)

eval_loader3 = DataLoader(CARLADataset(data_index=list(range(1, 11)), opt=opt, dataset_path='/media/wang/DATASET/CARLA_HUMAN/town01/'), batch_size=1, shuffle=False, num_workers=1)
eval_samples3 = iter(eval_loader3)
    
def show_traj(fake_traj, real_traj, t, step, img=None):
    fake_xy = fake_traj
    x = fake_xy[:,0]
    y = fake_xy[:,1]
    real_xy = real_traj
    real_x = real_xy[:,0]
    real_y = real_xy[:,1]

    max_x = 30.
    max_y = 30.
    
    fig = plt.figure(figsize=(7, 7))
    ax1 = fig.add_subplot(111)

    ax1.plot(x, y, label='trajectory', color = 'r', linewidth=5)
    ax1.plot(real_x, real_y, label='real-trajectory', color = 'b', linewidth=5, linestyle='--')
    ax1.set_xlabel('Forward/(m)')
    ax1.set_ylabel('Sideways/(m)')  
    ax1.set_xlim([0., max_x+5])
    ax1.set_ylim([-max_y, max_y])
    plt.legend(loc='lower right')
    
    t = max_x*t

    plt.legend(loc='lower left')
    #plt.show()
    plt.savefig('result/output/%s/' % opt.dataset_name+str(step)+'_curve.png')
    plt.close('all')

    if img is not None:
        save_image(img.data, 'result/output/%s/' % opt.dataset_name+str(step)+'_img.png', nrow=1, normalize=True)

def eval_error(total_step):
    points_num = 10
    encoder.eval()
    generator.eval()

    batch = next(eval_samples)
    batch['t'] = batch['t'].view(-1,1)
    batch['v0_array'] = batch['v0_array'].view(-1,1)
    batch['xy'] = batch['xy'].view(-1,2)
    to_device(batch, device)

    real_traj = batch['xy'].data.cpu().numpy()*opt.max_dist
    real_x = real_traj[:,0]
    real_y = real_traj[:,1]

    condition = batch['v0_array']

    single_latent = encoder(batch['img'], batch['v_0'])
    single_latent = single_latent.unsqueeze(1)
    latent = single_latent.expand(1, points_num, single_latent.shape[-1])
    latent = latent.reshape(1 * points_num, single_latent.shape[-1])

    output_xy = generator(condition, latent, batch['t'])
    fake_traj = output_xy.data.cpu().numpy()*opt.max_dist

    fake_x = fake_traj[:,0]
    fake_y = fake_traj[:,1]

    ex = np.mean(np.abs(fake_x-real_x))
    ey = np.mean(np.abs(fake_y-real_y))
    fde = np.hypot(fake_x - real_x, fake_y - real_y)[-1]
    ade = np.mean(np.hypot(fake_x - real_x, fake_y - real_y))
        
    logger.add_scalar('eval/ex',  ex.item(),  total_step)
    logger.add_scalar('eval/ey',  ey.item(),  total_step)
    logger.add_scalar('eval/fde', fde.item(), total_step)
    logger.add_scalar('eval/ade', ade.item(), total_step)

    encoder.train()
    generator.train()

def eval_error_robotcar(total_step):
    points_num = 16
    encoder.eval()
    generator.eval()

    batch = next(eval_samples2)
    batch['t'] = batch['t'].view(-1,1)
    batch['v0_array'] = batch['v0_array'].view(-1,1)
    batch['xy'] = batch['xy'].view(-1,2)
    to_device(batch, device)

    real_traj = batch['xy'].data.cpu().numpy()*opt.max_dist
    real_x = real_traj[:,0]
    real_y = real_traj[:,1]

    condition = batch['v0_array']

    single_latent = encoder(batch['img'], batch['v_0'])
    single_latent = single_latent.unsqueeze(1)
    latent = single_latent.expand(1, points_num, single_latent.shape[-1])
    latent = latent.reshape(1 * points_num, single_latent.shape[-1])

    output_xy = generator(condition, latent, batch['t'])
    fake_traj = output_xy.data.cpu().numpy()*opt.max_dist

    fake_x = fake_traj[:,0]
    fake_y = fake_traj[:,1]

    ex = np.mean(np.abs(fake_x-real_x))
    ey = np.mean(np.abs(fake_y-real_y))
    fde = np.hypot(fake_x - real_x, fake_y - real_y)[-1]
    ade = np.mean(np.hypot(fake_x - real_x, fake_y - real_y))
        
    logger2.add_scalar('eval/ex',  ex.item(),  total_step)
    logger2.add_scalar('eval/ey',  ey.item(),  total_step)
    logger2.add_scalar('eval/fde', fde.item(), total_step)
    logger2.add_scalar('eval/ade', ade.item(), total_step)

    encoder.train()
    generator.train()

def eval_error_carla(total_step):
    points_num = 16
    encoder.eval()
    generator.eval()

    batch = next(eval_samples3)
    batch['t'] = batch['t'].view(-1,1)
    batch['v0_array'] = batch['v0_array'].view(-1,1)
    batch['xy'] = batch['xy'].view(-1,2)
    to_device(batch, device)

    real_traj = batch['xy'].data.cpu().numpy()*opt.max_dist
    real_x = real_traj[:,0]
    real_y = real_traj[:,1]

    condition = batch['v0_array']

    single_latent = encoder(batch['img'], batch['v_0'])
    single_latent = single_latent.unsqueeze(1)
    latent = single_latent.expand(1, points_num, single_latent.shape[-1])
    latent = latent.reshape(1 * points_num, single_latent.shape[-1])

    output_xy = generator(condition, latent, batch['t'])
    fake_traj = output_xy.data.cpu().numpy()*opt.max_dist

    fake_x = fake_traj[:,0]
    fake_y = fake_traj[:,1]

    ex = np.mean(np.abs(fake_x-real_x))
    ey = np.mean(np.abs(fake_y-real_y))
    fde = np.hypot(fake_x - real_x, fake_y - real_y)[-1]
    ade = np.mean(np.hypot(fake_x - real_x, fake_y - real_y))
        
    logger3.add_scalar('eval/ex',  ex.item(),  total_step)
    logger3.add_scalar('eval/ey',  ey.item(),  total_step)
    logger3.add_scalar('eval/fde', fde.item(), total_step)
    logger3.add_scalar('eval/ade', ade.item(), total_step)

    encoder.train()
    generator.train()

def grad_penalty(loss, net):
    total_grad = 0
    for param in net.parameters():
        grad = torch.autograd.grad(loss, [param], create_graph=True)[0]
        total_grad += grad.pow(2).mean()
    return total_grad

total_step = 0
print('Start to train ...')
for i, batch in enumerate(train_loader):
    total_step += 1

    batch['img'] = batch['img'].to(device)
    batch['t'] = batch['t'].view(-1,1).to(device)
    batch['v_0'] = batch['v_0'].view(-1,1).to(device)
    batch['v0_array'] = batch['v0_array'].view(-1,1).to(device)
    batch['xy'] = batch['xy'].view(-1,2).to(device)
    batch['t'].requires_grad = True
    
    real_traj = batch['xy'].view(-1, opt.points_num*2)
    condition = batch['v0_array']

    single_latent = encoder(batch['img'], batch['v_0'])
    single_latent = single_latent.unsqueeze(1)
    latent = single_latent.expand(opt.batch_size, opt.points_num, single_latent.shape[-1])
    latent = latent.reshape(opt.batch_size * opt.points_num, single_latent.shape[-1])

    output_xy = generator(condition, latent, batch['t'])

    fake_traj = output_xy.view(-1, opt.points_num*2)

    generator.zero_grad()

    loss_trajectory_x = trajectory_criterion(real_traj.view(opt.batch_size, opt.points_num,2)[:,:,0]*opt.max_dist, fake_traj.view(opt.batch_size, opt.points_num,2)[:,:,0]*opt.max_dist)
    loss_trajectory_y = trajectory_criterion(real_traj.view(opt.batch_size, opt.points_num,2)[:,:,1]*opt.max_dist, fake_traj.view(opt.batch_size, opt.points_num,2)[:,:,1]*opt.max_dist)
    loss_trajectory = loss_trajectory_x + 5*loss_trajectory_y

    train_penalty = grad_penalty(loss_trajectory, generator)
    penalty_weight = min(10, 0.1+total_step/1000)
    train_penalty = penalty_weight*train_penalty
    loss = loss_trajectory + train_penalty
    loss.backward()
    g_optimizer.step()
    logger.add_scalar('train/loss_trajectory', loss_trajectory.item(), total_step)
    logger.add_scalar('train/train_penalty', train_penalty.item(), total_step)

    if total_step % opt.test_interval == 0:
        eval_error(total_step)
        eval_error_robotcar(total_step)
        eval_error_carla(total_step)
    
    if total_step % opt.checkpoint_interval == 0:
        torch.save(generator.state_dict(), 'result/saved_models/%s/generator_%d.pth'%(opt.dataset_name, total_step))
        torch.save(encoder.state_dict(), 'result/saved_models/%s/encoder_%d.pth'%(opt.dataset_name, total_step))
