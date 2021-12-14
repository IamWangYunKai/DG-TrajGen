
from typing import List

from ..augment import error_state, State


class BaseCurveOld(object):
    '''
        In vehicle coordinate.
    '''

    def __init__(self, states, sampling_resolution):
        self.states: List[State] = states
        self.sampling_resolution = sampling_resolution

        self.x, self.y, self.theta, self.k = [], [], [], []
        for s in self.states:
            self.x.append(s.x)
            self.y.append(s.y)
            self.theta.append(s.theta)
            self.k.append(s.k)

        self._max_coverage = 0


    def __len__(self):
        return len(self.states)


    def target_state(self, current_state: State):
        self._step_coverage(current_state)
        index = min(len(self)-1, self._max_coverage+1)
        return self.states[index]
    

    def _step_coverage(self, current_state: State):
        index = self._max_coverage
        for index in range(self._max_coverage, len(self)):
            longitudinal_e, _, _ = error_state(current_state, self.states[min(len(self)-1, index+1)])
            if longitudinal_e < 0:
                break
        self._max_coverage = index
        return


    def draw_plt(self):
        import matplotlib.pyplot as plt
        plt.plot(self.x, self.y, '-r')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()



# =============================================================================
# -- parallel  ----------------------------------------------------------------
# =============================================================================


import torch
from .functions.spiral import Spiral

class BaseCurve(object):
    '''
        In vehicle coordinate.
    '''

    def __init__(self, curve, id=0):
        self.x, self.y, self.theta, self.k = list([*curve])
        self.id = id
        self._max_coverage = 0


    def __len__(self):
        return len(self.x)

    def states(self, index, state0=None):
        x, y, theta, k = self.x[index], self.y[index], self.theta[index], self.k[index]
        if state0 == None:
            state = State(x=x, y=y, theta=theta, k=k)
        else:
            state = State(x=x, y=y, theta=theta, k=k).local2world(state0)
        return state

    def target_state(self, current_state: State):
        self._step_coverage(current_state)
        index = min(len(self)-1, self._max_coverage+1)
        return self.states(index)
    

    def _step_coverage(self, current_state: State):
        index = self._max_coverage
        for index in range(self._max_coverage, len(self)):
            longitudinal_e, _, _ = error_state(current_state, self.states(min(len(self)-1, index+1)))
            if longitudinal_e < 0:
                break
        self._max_coverage = index
        return


    @staticmethod
    def from_param(param, l=20, n=100):
        """
            param: torch.Size([order+1])
        """
        
        param = param.view(1,-1)  ### torch.Size([1,order+1])
        ls = torch.linspace(0,l, n).view(1,-1)  ### torch.Size([1,n])
        x = Spiral.x(ls, param)
        y = Spiral.y(ls, param)
        theta = Spiral.theta(ls, param)
        k = Spiral.curvature(ls, param)
        curve = BaseCurve(torch.cat([x, y, theta, k], dim=0))
        return curve


    def draw_plt(self):
        import matplotlib.pyplot as plt
        plt.plot(self.x, self.y, '-r')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()



class BaseCurves(object):
    def __init__(self, curves: List[torch.Tensor]):
        """
            states: List of torch.Size([4,num_points])
                4 means (x, y, theta, k)
        """

        self.curves = curves
        self.num_curves = len(curves)

    def __len__(self):
        return self.num_curves
    
    def get(self, index):
        curve = self.curves[index]
        return BaseCurve(curve, id=index)


    @staticmethod
    def from_params(params, lengths, n=100):
        """
            params: torch.Size([batch_size,order+1])
            lengths: torch.Size([bathc_size, num_points])
        """

        x = Spiral.x(lengths, params)
        y = Spiral.y(lengths, params)
        theta = Spiral.theta(lengths, params)
        k = Spiral.curvature(lengths, params)
        states = torch.stack([x, y, theta, k], dim=1)
        curves = BaseCurves( list([*states]) )
        return curves


    def draw_plt(self):
        import matplotlib.pyplot as plt
        for i in range(self.num_curves):
            c = self.get(i)
            plt.plot(c.x, c.y, '-')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()
