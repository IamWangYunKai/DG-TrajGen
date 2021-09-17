#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributions as dist

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        try:
            nn.init.constant_(m.bias, 0.01)
        except:
            pass
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.constant_(m.bias, 0.01)


IMG_DIM = 6
class CNN(nn.Module):
    def __init__(self, input_dim=3, out_dim=256):
        super(CNN, self).__init__()
        self.out_dim = out_dim
        self.conv1 = nn.Conv2d(input_dim, 64, 5, stride=3, padding=2)
        self.conv2 = nn.Conv2d(64,  128, 5, stride=4, padding=2)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, self.out_dim, 3, stride=2, padding=1)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.apply(weights_init)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, inplace=True)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, inplace=True)
        x = F.max_pool2d(x, 2, 2)         
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x, inplace=True)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv4(x)
        x = F.leaky_relu(x, inplace=True)
        x = x.view(-1, self.out_dim)
        return x

class IdResidualConvTBlockBNResize(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, output_padding=0, nonlin=nn.LeakyReLU):
        super(IdResidualConvTBlockBNResize, self).__init__()
        self.nonlin = nonlin(inplace=True)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.output_padding = output_padding

        self.conv1 = nn.ConvTranspose2d(self.in_channels, self.out_channels, self.kernel_size, stride=2, padding=self.padding, output_padding=self.output_padding, bias=False)
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.conv2 = nn.ConvTranspose2d(self.out_channels, self.out_channels, self.kernel_size, stride=1, padding=self.padding, output_padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(self.out_channels)

        self.conv_residual = nn.ConvTranspose2d(self.in_channels, self.out_channels, 1, stride=2, padding=0, output_padding=self.output_padding, bias=False)
        self.bn_residual = nn.BatchNorm2d(self.out_channels)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.nonlin(x) # it is better for a vae architecture to have that here, instead of the end of a block

        h = self.conv1(x)
        h = self.bn1(h)
        h = self.nonlin(h)

        h = self.conv2(h)
        h = self.bn2(h)

        residual = self.conv_residual(x)
        residual = self.bn_residual(residual)

        return 0.1 * h + residual

class IdResidualConvTBlockBNIdentity(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, output_padding=0, nonlin=nn.LeakyReLU):
        super(IdResidualConvTBlockBNIdentity, self).__init__()
        self.nonlin = nonlin(inplace=True)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.output_padding = output_padding

        self.conv1 = nn.ConvTranspose2d(self.in_channels, self.out_channels, self.kernel_size, stride=1, padding=self.padding, output_padding=self.output_padding, bias=False)
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.conv2 = nn.ConvTranspose2d(self.out_channels, self.out_channels, self.kernel_size, stride=1, padding=self.padding, output_padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(self.out_channels)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.nonlin(x) # it is better for a vae architecture to have that here, instead of the end of a block

        h = self.conv1(x)
        h = self.bn1(h)
        h = self.nonlin(h)

        h = self.conv2(h)
        h = self.bn2(h)

        return 0.1 * h + x

# Decoders
class px(nn.Module):
    def __init__(self, zd_dim, zx_dim, zy_dim):
        super(px, self).__init__()

        self.fc1 = nn.Sequential(nn.Linear(zd_dim + zx_dim + zy_dim, 64*4*4, bias=False), nn.BatchNorm1d(64*4*4))
        self.rn1 = IdResidualConvTBlockBNIdentity(64, 64, 3, padding=1, output_padding=0, nonlin=nn.LeakyReLU)
        self.rn2 = IdResidualConvTBlockBNResize(64, 64, 3, padding=1, output_padding=1, nonlin=nn.LeakyReLU)
        self.rn3 = IdResidualConvTBlockBNIdentity(64, 64, 3, padding=1, output_padding=0, nonlin=nn.LeakyReLU)
        self.rn4 = IdResidualConvTBlockBNResize(64, 32, 3, padding=1, output_padding=1, nonlin=nn.LeakyReLU)
        self.rn5 = IdResidualConvTBlockBNIdentity(32, 32, 3, padding=1, output_padding=0, nonlin=nn.LeakyReLU)
        self.rn6 = IdResidualConvTBlockBNResize(32, 32, 3, padding=1, output_padding=1, nonlin=nn.LeakyReLU)
        self.rn7 = IdResidualConvTBlockBNIdentity(32, 32, 3, padding=1, output_padding=0, nonlin=nn.LeakyReLU)
        self.rn8 = IdResidualConvTBlockBNResize(32, 32, 3, padding=1, output_padding=1, nonlin=nn.LeakyReLU)
        self.rn9 = IdResidualConvTBlockBNIdentity(32, 32, 3, padding=1, output_padding=0, nonlin=nn.LeakyReLU)
        self.rn10 = IdResidualConvTBlockBNResize(32, 16, 3, padding=1, output_padding=1, nonlin=nn.LeakyReLU)
        self.pool = nn.AdaptiveAvgPool2d((200,400))
        self.conv1 = nn.Conv2d(16, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=IMG_DIM, kernel_size=1, padding=0)

        torch.nn.init.xavier_uniform_(self.fc1[0].weight)
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        self.conv1.bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        self.conv2.bias.data.zero_()

    def forward(self, zd, zx, zy):
        zdzxzy = torch.cat((zd, zx, zy), dim=1)
        h = self.fc1(zdzxzy)
        h = h.view(-1, 64, 4, 4)
        h = self.rn1(h)
        h = self.rn2(h)
        h = self.rn3(h)
        h = self.rn4(h)
        h = self.rn5(h)
        h = self.rn6(h)
        h = self.rn7(h)
        h = self.rn8(h)
        h = self.rn9(h)
        h = self.rn10(h)
        h = F.leaky_relu(h, inplace=True)
        h = self.pool(h)
        h = self.conv1(h)
        loc_img = self.conv2(h)
        return loc_img

class pzd(nn.Module):
    def __init__(self, d_dim, zd_dim):
        super(pzd, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(d_dim, zd_dim, bias=False), nn.BatchNorm1d(zd_dim), nn.LeakyReLU(inplace=True))
        self.fc21 = nn.Sequential(nn.Linear(zd_dim, zd_dim))
        self.fc22 = nn.Sequential(nn.Linear(zd_dim, zd_dim), nn.Softplus())

        torch.nn.init.xavier_uniform_(self.fc1[0].weight)
        self.fc1[1].weight.data.fill_(1)
        self.fc1[1].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc21[0].weight)
        self.fc21[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc22[0].weight)
        self.fc22[0].bias.data.zero_()

    def forward(self, d):
        hidden = self.fc1(d)
        zd_loc = self.fc21(hidden)
        zd_scale = self.fc22(hidden) + 1e-7
        return zd_loc, zd_scale

class pzy(nn.Module):
    def __init__(self, y_dim, zy_dim):
        super(pzy, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(y_dim, zy_dim, bias=False), nn.BatchNorm1d(zy_dim), nn.LeakyReLU(inplace=True))
        self.fc21 = nn.Sequential(nn.Linear(zy_dim, zy_dim))
        self.fc22 = nn.Sequential(nn.Linear(zy_dim, zy_dim), nn.Softplus())

        torch.nn.init.xavier_uniform_(self.fc1[0].weight)
        self.fc1[1].weight.data.fill_(1)
        self.fc1[1].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc21[0].weight)
        self.fc21[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc22[0].weight)
        self.fc22[0].bias.data.zero_()

    def forward(self, y):
        hidden = self.fc1(y)
        zy_loc = self.fc21(hidden)
        zy_scale = self.fc22(hidden) + 1e-7

        return zy_loc, zy_scale

# Encoders
class qzd(nn.Module):
    def __init__(self, zd_dim):
        super(qzd, self).__init__()
        self.cnn = CNN(IMG_DIM)
        self.fc11 = nn.Sequential(nn.Linear(256, zd_dim))
        self.fc12 = nn.Sequential(nn.Linear(256, zd_dim), nn.Softplus())

        torch.nn.init.xavier_uniform_(self.fc11[0].weight)
        self.fc11[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc12[0].weight)
        self.fc12[0].bias.data.zero_()

    def forward(self, x):
        h = self.cnn(x)
        zd_loc = self.fc11(h)
        zd_scale = self.fc12(h) + 1e-7
        return zd_loc, zd_scale


class qzx(nn.Module):
    def __init__(self, zd_dim):
        super(qzx, self).__init__()
        self.cnn = CNN(IMG_DIM)
        self.fc11 = nn.Sequential(nn.Linear(256, zd_dim))
        self.fc12 = nn.Sequential(nn.Linear(256, zd_dim), nn.Softplus())

        torch.nn.init.xavier_uniform_(self.fc11[0].weight)
        self.fc11[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc12[0].weight)
        self.fc12[0].bias.data.zero_()

    def forward(self, x):
        h = self.cnn(x)
        zx_loc = self.fc11(h)
        zx_scale = self.fc12(h) + 1e-7

        return zx_loc, zx_scale


class qzy(nn.Module):
    def __init__(self, zd_dim):
        super(qzy, self).__init__()
        self.cnn = CNN(IMG_DIM)
        self.fc11 = nn.Sequential(nn.Linear(256, zd_dim))
        self.fc12 = nn.Sequential(nn.Linear(256, zd_dim), nn.Softplus())

        torch.nn.init.xavier_uniform_(self.fc11[0].weight)
        self.fc11[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc12[0].weight)
        self.fc12[0].bias.data.zero_()

    def forward(self, x):
        h = self.cnn(x)
        zy_loc = self.fc11(h)
        zy_scale = self.fc12(h) + 1e-4

        return zy_loc, zy_scale


# Auxiliary tasks
class qd(nn.Module):
    def __init__(self, d_dim, zd_dim):
        super(qd, self).__init__()

        self.fc1 = nn.Linear(zd_dim, d_dim)
        self.activation = nn.LeakyReLU(inplace=True)

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.zero_()

    def forward(self, zd):
        h = self.activation(zd)
        loc_d = self.fc1(h)

        return loc_d

class Generator(nn.Module):
    def __init__(self, input_dim=8, output_dim=2):
        super(Generator, self).__init__()
        self.linear1 = nn.Linear(input_dim, 512)
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, 512)
        self.linear4 = nn.Linear(512, 256)
        self.linear5 = nn.Linear(256, output_dim)
        
        self.apply(weights_init)
        
    def forward(self, condition, v0, t):
        x = torch.cat([condition, v0], dim=1)
        x = torch.cat([x, t], dim=1)
        x = self.linear1(x)
        x = F.leaky_relu(x, inplace=True)
        x = self.linear2(x)
        x = F.leaky_relu(x, inplace=True)
        x = self.linear3(x)
        x = F.leaky_relu(x, inplace=True)
        x = self.linear4(x)
        x = torch.cos(x)
        x = self.linear5(x)
        return x
        
class qy(nn.Module):
    def __init__(self, zy_dim):
        super(qy, self).__init__()
        self.trajectory_generation = Generator(input_dim=1+1+zy_dim, output_dim=4)

    def forward(self, zy, v0, t):
        h = F.leaky_relu(zy, inplace=True)
        condition = h.unsqueeze(1)
        condition = condition.expand(h.shape[0], t.shape[-1], h.shape[-1])
        condition = condition.reshape(h.shape[0] * t.shape[-1], h.shape[-1])

        output = self.trajectory_generation(condition, v0.view(-1, 1), t.view(-1, 1))
        output_xy = output[:,:2]
        logvar = output[:,2:]
        return output_xy, logvar

class DIVAResnetBNLinear(nn.Module):
    def __init__(self, points_num=16, d_dim=7):
        super(DIVAResnetBNLinear, self).__init__()
        self.zd_dim = 64
        self.zx_dim = 64
        self.zy_dim = 64
        self.d_dim = d_dim
        self.x_dim = 200 * 400 * IMG_DIM
        self.y_dim = 2*points_num

        self.px = px(self.zd_dim, self.zx_dim, self.zy_dim)
        self.pzd = pzd(self.d_dim, self.zd_dim)
        self.pzy = pzy(self.y_dim, self.zy_dim)

        self.qzd = qzd(self.zd_dim)
        self.qzx = qzx(self.zy_dim)
        self.qzy = qzy(self.zy_dim)

        self.qd = qd(self.d_dim, self.zd_dim)
        self.qy = qy(self.zy_dim)

        self.aux_loss_multiplier_y = 75000
        self.aux_loss_multiplier_d = 100000

        self.beta_d = 0.1
        self.beta_x = 0.1
        self.beta_y = 0.1
        self.reconstruction_loss = torch.nn.MSELoss().to(device)

        self.cuda()

    def forward(self, d, x, y, v0, t):
        # Encode
        zd_q_loc, zd_q_scale = self.qzd(x)
        zx_q_loc, zx_q_scale = self.qzx(x)
        zy_q_loc, zy_q_scale = self.qzy(x)

        # Reparameterization trick
        qzd = dist.Normal(zd_q_loc, zd_q_scale)
        zd_q = qzd.rsample()
        qzx = dist.Normal(zx_q_loc, zx_q_scale)
        zx_q = qzx.rsample()
        qzy = dist.Normal(zy_q_loc, zy_q_scale)
        zy_q = qzy.rsample()

        # Decode
        x_recon = self.px(zd_q, zx_q, zy_q)

        # Prior
        zd_p_loc, zd_p_scale = self.pzd(d)
        zx_p_loc, zx_p_scale = torch.zeros(zd_p_loc.size()[0], self.zx_dim).cuda(),\
                                   torch.ones(zd_p_loc.size()[0], self.zx_dim).cuda()
        zy_p_loc, zy_p_scale = self.pzy(y)

        # Reparameterization trick
        pzd = dist.Normal(zd_p_loc, zd_p_scale)
        pzx = dist.Normal(zx_p_loc, zx_p_scale)
        pzy = dist.Normal(zy_p_loc, zy_p_scale)

        # Auxiliary losses
        d_hat = self.qd(zd_q)
        y_hat, logvar = self.qy(zy_q, v0, t)

        return x_recon, d_hat, y_hat, logvar, qzd, pzd, zd_q, qzx, pzx, zx_q, qzy, pzy, zy_q


    def predict(self, x, v0, t):
        zy_q_loc, zy_q_scale = self.qzy(x)
        y_hat, logvar = self.qy(zy_q_loc, v0, t)
        return y_hat

    def il_loss_function(self, output_xy, logvar, target_xy):
        l2_loss = torch.pow((output_xy - target_xy), 2)
        il_loss = torch.mean((torch.exp(-logvar) * l2_loss + logvar) * 0.5)
        return il_loss

    def loss_function(self, d, x, y, v0, t):
        x_recon, d_hat, y_hat, logvar, qzd, pzd, zd_q, qzx, pzx, zx_q, qzy, pzy, zy_q = self.forward(d, x, y, v0, t)
        CE_x = self.reconstruction_loss(x, x_recon)

        zd_p_minus_zd_q = torch.sum(pzd.log_prob(zd_q) - qzd.log_prob(zd_q))
        KL_zx = torch.sum(pzx.log_prob(zx_q) - qzx.log_prob(zx_q))
        zy_p_minus_zy_q = torch.sum(pzy.log_prob(zy_q) - qzy.log_prob(zy_q))

        _, d_target = d.max(dim=1)
        CE_d = F.cross_entropy(d_hat, d_target, reduction='sum')

        CE_y = self.il_loss_function(y_hat, logvar, y.view(-1, 2))
        return CE_x \
                - self.beta_d * zd_p_minus_zd_q \
                - self.beta_x * KL_zx \
                - self.beta_y * zy_p_minus_zy_q \
                + self.aux_loss_multiplier_d * CE_d \
                + self.aux_loss_multiplier_y * CE_y\
            , CE_y