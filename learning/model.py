#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

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

class CNN(nn.Module):
    def __init__(self, input_dim=1, out_dim=256):
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

class Generator(nn.Module):
    def __init__(self, input_dim=8, output=2):
        super(Generator, self).__init__()
        self.linear1 = nn.Linear(input_dim, 512)
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, 512)
        self.linear4 = nn.Linear(512, 256)
        self.linear5 = nn.Linear(256, output)
        
        self.apply(weights_init)
        
    def forward(self, condition, noise, t):
        x = torch.cat([condition, noise], dim=1)
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

class Discriminator(nn.Module):
    def __init__(self, input_dim=8*6*2, output=1):
        super(Discriminator, self).__init__()
        self.linear1 = nn.Linear(input_dim, 512)
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, 256)
        self.linear4 = nn.Linear(256, output)
        
        self.apply(weights_init)
        
    def forward(self, x):
        x = self.linear1(x)
        x = F.leaky_relu(x, inplace=True)
        x = self.linear2(x)
        x = F.leaky_relu(x, inplace=True)
        x = self.linear3(x)
        x = F.leaky_relu(x, inplace=True)
        x = self.linear4(x)
        return x


class EncoderWithV(nn.Module):
    def __init__(self, input_dim=3, out_dim=256):
        super(EncoderWithV, self).__init__()
        self.out_dim = out_dim
        self.conv1 = nn.Conv2d(input_dim, 64, 5, stride=3, padding=2)
        self.conv2 = nn.Conv2d(64,  128, 5, stride=4, padding=2)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, self.out_dim, 3, stride=2, padding=1)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)

        self.linear1 = nn.Linear(self.out_dim + 1, 256)
        self.linear2 = nn.Linear(256, self.out_dim)
        
        self.apply(weights_init)

    def forward(self, x, v):
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
        x = x.view(-1, self.out_dim)
        x = F.leaky_relu(x, inplace=True)
        x = torch.cat([x, v], 1)
        x = self.linear1(x)
        x = F.leaky_relu(x, inplace=True)
        x = self.linear2(x)
        return x

class MLP(nn.Module):
    def __init__(self, input_dim=257, output_dim=2):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_dim, 512)
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, 256)
        self.linear4 = nn.Linear(256, output_dim)
        
        self.apply(weights_init)
        
    def forward(self, x, v0):
        x = torch.cat([x, v0], dim=1)
        x = self.linear1(x)
        x = F.leaky_relu(x, inplace=True)
        # x = torch.tanh(x)
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear2(x)
        x = F.leaky_relu(x, inplace=True)
        # x = torch.tanh(x)
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear3(x)
        x = F.leaky_relu(x, inplace=True)
        # x = torch.tanh(x)
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear4(x)
        return x

class MixStyle(nn.Module):
    def __init__(self, p=0.5, alpha=0.3, eps=1e-6):
        """
        Args:
          p (float): probability of using MixStyle.
          alpha (float): parameter of the Beta distribution.
          eps (float): scaling parameter to avoid numerical issues.
        """
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps

    def forward(self, x):
        if not self.training:
            return x

        if random.random() > self.p:
            return x

        B = x.size(0)

        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x - mu) / sig

        lmda = self.beta.sample((B, 1, 1, 1))
        lmda = lmda.to(x.device)

        perm = torch.randperm(B)

        mu2, sig2 = mu[perm], sig[perm]
        mu_mix = mu * lmda + mu2 * (1 - lmda)
        sig_mix = sig * lmda + sig2 * (1 - lmda)

        return x_normed * sig_mix + mu_mix

    def __init__(self, p=0.5, alpha=0.3, eps=1e-6):
        """
        Args:
          p (float): probability of using MixStyle.
          alpha (float): parameter of the Beta distribution.
          eps (float): scaling parameter to avoid numerical issues.
        """
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps

    def forward(self, x):
        if not self.training:
            return x

        if random.random() > self.p:
            return x

        B = x.size(0)

        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x - mu) / sig

        lmda = self.beta.sample((B, 1, 1, 1))
        lmda = lmda.to(x.device)

        perm = torch.randperm(B)

        mu2, sig2 = mu[perm], sig[perm]
        mu_mix = mu * lmda + mu2 * (1 - lmda)
        sig_mix = sig * lmda + sig2 * (1 - lmda)

        return x_normed * sig_mix + mu_mix

class MixStyleEncoder(nn.Module):
    def __init__(self, input_dim=3, out_dim=256):
        super(MixStyleEncoder, self).__init__()
        self.out_dim = out_dim
        self.conv1 = nn.Conv2d(input_dim, 64, 5, stride=3, padding=2)
        self.conv2 = nn.Conv2d(64,  128, 5, stride=4, padding=2)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, self.out_dim, 3, stride=2, padding=1)
        self.mixstyle = MixStyle(p=0.5, alpha=0.1)

        self.apply(weights_init)

    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x)
        x = self.mixstyle(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = F.leaky_relu(x)
        x = self.mixstyle(x)
        x = F.max_pool2d(x, 2, 2)         
        x = self.conv3(x)
        x = F.leaky_relu(x)
        x = self.mixstyle(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv4(x)
        x = F.leaky_relu(x)
        x = x.view(-1, self.out_dim)
        return x

class ModelMixStyle(nn.Module):
    def __init__(self):
        super(ModelMixStyle, self).__init__()
        self.encoder = MixStyleEncoder(input_dim=6, out_dim=256)
        self.decoder = Generator(input_dim=256+2, output_dim=2)

    def forward(self):
        pass

    def get_encoder(self, x):
        return self.encoder(x)

    def get_trajectory(self, x, v0, t):
        return self.decoder(x, v0, t)

class MixstyleWithV(nn.Module):
    def __init__(self, input_dim=3, out_dim=256):
        super(MixstyleWithV, self).__init__()
        self.out_dim = out_dim
        self.conv1 = nn.Conv2d(input_dim, 64, 5, stride=3, padding=2)
        self.conv2 = nn.Conv2d(64,  128, 5, stride=4, padding=2)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, self.out_dim, 3, stride=2, padding=1)
        
        self.mixstyle = MixStyle(p=0.5, alpha=0.1)

        self.linear1 = nn.Linear(self.out_dim + 1, 256)
        self.linear2 = nn.Linear(256, self.out_dim)
        
        self.apply(weights_init)

    def forward(self, x, v):
        x = self.conv1(x)
        x = F.leaky_relu(x, inplace=True)
        x = self.mixstyle(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = F.leaky_relu(x, inplace=True)
        x = self.mixstyle(x)
        x = F.max_pool2d(x, 2, 2)         
        x = self.conv3(x)
        x = F.leaky_relu(x, inplace=True)
        x = self.mixstyle(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv4(x)
        x = x.view(-1, self.out_dim)
        x = F.leaky_relu(x, inplace=True)
        x = torch.cat([x, v], 1)
        x = self.linear1(x)
        x = F.leaky_relu(x, inplace=True)
        x = self.linear2(x)
        return x


class ModelMixStyleWithV(nn.Module):
    def __init__(self, latent_dim=256):
        super(ModelMixStyleWithV, self).__init__()
        self.encoder = MixstyleWithV(input_dim=6, out_dim=latent_dim)
        self.decoder = Generator(input_dim=latent_dim+2, output=2)

    def forward(self):
        pass

    def get_encoder(self, x, v):
        return self.encoder(x, v)

    def get_trajectory(self, x, v0, t):
        return self.decoder(x, v0, t)