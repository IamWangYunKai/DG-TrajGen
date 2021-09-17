
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from os.path import join, dirname
sys.path.insert(0, join(dirname(__file__), '../../'))

import os
import time
import random
import argparse
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from diva_model import DIVAResnetBNLinear

from carla_utils import parse_yaml_file_unsafe

from robo_utils.oxford.oxford_dataset import DIVADataset
from robo_utils.kitti.torch_dataset import BoostDataset as KittiDataset
from learning.dataset import CARLADatasetDomain
from utils import write_params, to_device


parser = argparse.ArgumentParser()

parser.add_argument('--test_mode', type=bool, default=False, help='test model switch')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--dataset_name', type=str, default="train-kitti-01", help='name of the dataset')
parser.add_argument('--width', type=int, default=400, help='image width')
parser.add_argument('--height', type=int, default=200, help='image height')
parser.add_argument('--channel', type=int, default=3, help='image channel')
parser.add_argument('--scale', type=float, default=25., help='longitudinal length')
parser.add_argument('--batch_size', type=int, default=32, help='size of the batches')
parser.add_argument('--traj_steps', type=int, default=8, help='traj steps')
parser.add_argument('--points_num', type=int, default=16, help='points number')
parser.add_argument('--domian_num', type=int, default=5, help='domain number')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='adam: weight_decay')
parser.add_argument('--lr', type=float, default=3e-4, help='adam: learning rate')
parser.add_argument('--gamma', type=float, default=0.2, help='xy and vxy loss trade off')
parser.add_argument('--gamma2', type=float, default=0.0, help='xy and axy loss trade off')
parser.add_argument('--img_step', type=int, default=3, help='RNN input image step')
parser.add_argument('--n_cpu', type=int, default=16, help='number of cpu threads to use during batch generation')
parser.add_argument('--checkpoint_interval', type=int, default=1000, help='interval between model checkpoints')
parser.add_argument('--test_interval', type=int, default=5, help='interval between model test')
parser.add_argument('--max_dist', type=float, default=25., help='max distance')
parser.add_argument('--max_speed', type=float, default=10., help='max distance')
parser.add_argument('--max_t', type=float, default=3., help='max time')
parser.add_argument('--model_num', type=int, default=5, help='model_num')

opt = parser.parse_args()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
kwargs = {'num_workers': 16, 'pin_memory': False}

torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)
np.random.seed(opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

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

model = DIVAResnetBNLinear(points_num=opt.points_num, d_dim=opt.domian_num).to(device)

optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

param = parse_yaml_file_unsafe('../../params/param_kitti.yaml')
train_loader = DataLoader(KittiDataset(param, 'train', opt), batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)
train_samples = iter(train_loader)


param = parse_yaml_file_unsafe('../../params/param_oxford.yaml')
eval_loader = DataLoader(DIVADataset(param, mode='eval', opt=opt), batch_size=1, shuffle=False, num_workers=1)
eval_samples_robotcar = iter(eval_loader)

param = parse_yaml_file_unsafe('../../params/param_kitti.yaml')
eval_loader2 = DataLoader(KittiDataset(param, 'eval', opt), batch_size=1, shuffle=False, num_workers=1)
eval_samples_kitti = iter(eval_loader2)

eval_loader3 = DataLoader(CARLADatasetDomain(data_index=list(range(1,11)), opt=opt, dataset_path='/media/wang/DATASET/CARLA_HUMAN/town01/'), batch_size=1, shuffle=False, num_workers=1)
eval_samples_carla = iter(eval_loader3)

trajectory_criterion = torch.nn.MSELoss().to(device)


def eval_error_kitti(total_step):
    model.eval()

    batch = next(eval_samples_kitti)
    to_device(batch, device)

    x = batch['img']
    y = batch['xy'].view(-1,2)
    v0 = batch['v0_array']
    t = batch['t']

    y_hat = model.predict(x, v0, t)
    fake_traj = y_hat.view(-1,2).data.cpu().numpy()*opt.max_dist

    real_traj = batch['xy'][0].data.cpu().numpy()*opt.max_dist

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
    to_device(batch, device)

    x = batch['img']
    y = batch['xy'].view(-1,2)
    v0 = batch['v0_array']
    t = batch['t']

    y_hat = model.predict(x, v0, t)
    fake_traj = y_hat.view(-1,2).data.cpu().numpy()*opt.max_dist

    real_traj = batch['xy'][0].data.cpu().numpy()*opt.max_dist
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
    to_device(batch, device)

    x = batch['img']
    y = batch['xy'].view(-1,2)
    v0 = batch['v0_array']
    t = batch['t']

    y_hat = model.predict(x, v0, t)
    fake_traj = y_hat.view(-1,2).data.cpu().numpy()*opt.max_dist

    real_traj = batch['xy'][0].data.cpu().numpy()*opt.max_dist

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
    to_device(batch, device)
    x = batch['img']
    d = batch['domian']
    y = batch['xy'].view(opt.batch_size, 2*opt.points_num)
    v0 = batch['v0_array']
    t = batch['t']

    optimizer.zero_grad()
    vae_loss, regression_loss = model.loss_function(d, x, y, v0, t)
    loss = regression_loss
    loss.backward()
    optimizer.step()

    logger.add_scalar('train/vae_loss', vae_loss.item(), total_steps)
    logger.add_scalar('train/regression_loss', regression_loss.item(), total_steps)

    if total_steps > 0 and total_steps % opt.test_interval == 0:
        eval_error_carla(total_steps)
        eval_error_kitti(total_steps)
        eval_error_robotcar(total_steps)

    if total_steps % opt.checkpoint_interval == 0:
        torch.save(model.state_dict(), 'result/saved_models/%s/model_%d.pth'%(opt.dataset_name, total_steps))