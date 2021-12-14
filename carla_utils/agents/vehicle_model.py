
import numpy as np

import torch
import torch.nn as nn

from ..basic import pi2pi_numpy, pi2pi_tensor
from ..augment import State


class RealModel(object):
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    def forward(self, vehicle, control):
        vehicle.apply_control(control)


class BicycleModel2D(RealModel):
    def __init__(self, dt, wheelbase):
        self.dt, self.wheelbase = dt, wheelbase
        
    def forward(self, state: State, action):
        a, steer = action[0], action[1]
        x, y, theta, v = state.x, state.y, state.theta, state.v
        next_state = State(
            x=x + self.dt *v * np.cos(theta),
            y=y + self.dt *v * np.sin(theta),
            theta=pi2pi_numpy(theta + self.dt * v * np.tan(steer) / self.wheelbase),
            v=v + self.dt *a,
        )
        return next_state


class BicycleModel2DParallel(nn.Module):
    def __init__(self, dt, wheelbase):
        super(BicycleModel2DParallel, self).__init__()
        
        self.dt, self.wheelbase = dt, wheelbase
        

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        """

        Args:
            state: (x, y, theta, v), torch.Size([batch_size, dim_state]
            action: (a, steer), torch.Size([batch_size, dim_action])
        """

        a, steer = action[:,0], action[:,1]
        x, y, theta, v = state[:,0], state[:,1], state[:,2], state[:,3]
        next_state = torch.stack([
                x + self.dt *v * torch.cos(theta),
                y + self.dt *v * torch.sin(theta),
                pi2pi_tensor(theta + self.dt * v * torch.tan(steer) / self.wheelbase),
                v + self.dt *a,
            ], dim=1)
        return next_state



class SteerModel(RealModel):
    def __init__(self, dt, alpha=0.0):
        self.dt = dt
        self.xk, self.y = 0.0, 0.0
        self.alpha = alpha
    
    def forward(self, u):
        """
            u: normalized control
        """
        self.y = self.xk
        # alpha = np.clip(self.alpha + np.clip(np.random.normal(scale=0.2), -0.2, 0.2), 0, 1)
        alpha = self.alpha
        self.xk = alpha * self.xk + (1-alpha) * u
        return self.y
        return self.xk

