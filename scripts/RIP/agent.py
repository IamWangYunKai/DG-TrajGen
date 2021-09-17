#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.optim as optim
from dim_model import ImitativeModel

import numpy as np
import scipy.interpolate

class RIPAgent():
    def __init__(self, output_shape=(16, 2), model_num=3, opt_steps=10, lr=1e-1):
        self.output_shape = output_shape
        self.model_num = model_num
        self.opt_steps = opt_steps
        self.lr = lr
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        models = [ImitativeModel(output_shape=self.output_shape) for _ in range(self.model_num)]
        self.models = [model.to(self.device) for model in models]

    def step(self, observation):
        #epsilon = 1.0
        batch_size = observation['visual_features'].shape[0]

        # Sets initial sample to base distribution's mean.
        x = self.models[0]._decoder._base_dist.mean.clone().detach().repeat(
            batch_size, 1).view(
                batch_size,
                *self.models[0]._output_shape,
            )
        x.requires_grad = True

        # The contextual parameters, caches for efficiency.
        zs = [model._params(**observation) for model in self.models]

        # Initialises a gradient-based optimiser.
        optimizer = optim.Adam(params=[x], lr=self.lr)

        # Stores the best values.
        x_best = x.clone()
        loss_best = torch.ones(()).to(x.device) * 1000.0

        for _ in range(self.opt_steps):
            # Resets optimizer's gradients.
            optimizer.zero_grad()
            # Operate on `y`-space.
            y, _ = self.models[0]._decoder._forward(x=x, z=zs[0])
            # Iterates over the `K` models and calculates the imitation posterior.
            imitation_posteriors = list()
            for model, z in zip(self.models, zs):
                # Calculates imitation prior.
                _, log_prob, logabsdet = model._decoder._inverse(y=y, z=z)
                imitation_prior = torch.mean(log_prob - logabsdet)
                # Calculates goal likelihodd.
                goal_likelihood = 0.
                # goal_likelihood = model._goal_likelihood(
                #     y=y,
                #     goal=observation["goal"],
                #     epsilon=epsilon,
                # )
                imitation_posteriors.append(imitation_prior + goal_likelihood)
            # Aggregate scores from the `K` models.
            imitation_posteriors = torch.stack(imitation_posteriors, dim=0)

            # WCM
            loss, _ = torch.min(-imitation_posteriors, dim=0)

            # Backward pass.
            loss.backward(retain_graph=True)
            # Performs a gradient descent step.
            optimizer.step()
            # Book-keeping
            if loss < loss_best:
                x_best = x.clone()
                loss_best = loss.clone()

        plan, _ = self.models[0]._decoder._forward(x=x_best, z=zs[0])
        ######
        plan = plan.detach().cpu().numpy()[0]  # [T, 2]

        xy = plan
        # player_future_length = 40
        # increments = player_future_length // plan.shape[0]
        # time_index = list(range(0, player_future_length, increments))  # [T]
        # plan_interp = scipy.interpolate.interp1d(x=time_index, y=plan, axis=0)
        # xy = plan_interp(np.arange(0, time_index[-1]))
        return xy