
import numpy as np
from scipy import integrate

class QuadraticSpiral(object):
    '''
        curve length start from 0
        p: parameter, np.array, column vector
        curvature_vector: np.array, row vector
    '''
    @staticmethod
    def x(s, p):
        return integrate.quad(QuadraticSpiral.cos_theta, 0,s, args=p)[0]
    @staticmethod
    def y(s, p):
        return integrate.quad(QuadraticSpiral.sin_theta, 0,s, args=p)[0]
    @staticmethod
    def cos_theta(s, p):
        return np.cos(QuadraticSpiral.theta(s, p))
    @staticmethod
    def sin_theta(s, p):
        return np.sin(QuadraticSpiral.theta(s, p))
    @staticmethod
    def theta(s, p):
        theta_vector = np.array([[s, s**2/2, s**3/3]])
        # print(theta_vector.shape, p.shape)
        return np.dot(theta_vector, p)[0][0]
    @staticmethod
    def curvature(s, p):
        curvature_vector = np.array([[1, s, s**2]])
        return np.dot(curvature_vector, p)[0][0]


class ConstantSpiral(object):
    @staticmethod
    def x(s, p):
        return integrate.quad(ConstantSpiral.cos_theta, 0,s, args=p)[0]
    @staticmethod
    def y(s, p):
        return integrate.quad(ConstantSpiral.sin_theta, 0,s, args=p)[0]
    @staticmethod
    def cos_theta(s, p):
        return np.cos(ConstantSpiral.theta(s, p))
    @staticmethod
    def sin_theta(s, p):
        return np.sin(ConstantSpiral.theta(s, p))
    @staticmethod
    def theta(s, p):
        return p[0][0] * s
    @staticmethod
    def curvature(s, p):
        return p[0][0]



# =============================================================================
# -- parallel  ----------------------------------------------------------------
# =============================================================================


import torch

class Spiral(object):
    order = 2

    '''
        s (弧长):      torch.Size([batch_size, horizon])
        p (多项式系数): torch.Size([batch_size, order+1])
    '''

    @staticmethod
    def x(s, p):
        lower_limit = torch.zeros(*s.shape, dtype=torch.float32)
        return integral_simpson(Spiral.cos_theta, lower_limit,s, args=(p,))
    @staticmethod
    def y(s, p):
        lower_limit = torch.zeros(*s.shape, dtype=torch.float32)
        return integral_simpson(Spiral.sin_theta, lower_limit,s, args=(p,))
    @staticmethod
    def cos_theta(s, p):
        return torch.cos(Spiral.theta(s, p))
    @staticmethod
    def sin_theta(s, p):
        return torch.sin(Spiral.theta(s, p))
    @staticmethod
    def theta(s, p):
        order = p.shape[-1] - 1
        theta_vector = []
        for i in range(1, order+2):
            theta_vector.append( s**i / i )
        theta_vector = torch.stack(theta_vector, dim=-1)
        return torch.matmul(theta_vector, p.unsqueeze(-1)).squeeze(-1)
    @staticmethod
    def curvature(s, p):
        order = p.shape[-1] - 1
        curvature_vector = []
        for i in range(1, order+2):
            curvature_vector.append( s**(i-1) )
        curvature_vector = torch.stack(curvature_vector, dim=-1)
        return torch.matmul(curvature_vector, p.unsqueeze(-1)).squeeze(-1)




def integral_simpson(func, lower_limit, upper_limit, args, n=1000):
    if n % 2 != 0:
        n += 1

    coefficient = torch.tensor([4,2]).repeat(int(n/2))[:-1]
    delta_x = torch.from_numpy(np.linspace(lower_limit, upper_limit, n+1)).to(torch.float32)[1:-1]

    coefficient = coefficient.view(-1,1,1).repeat(1,1,delta_x.shape[-1])
    res_intermediate = func(lower_limit + delta_x, *args)

    res = func(lower_limit, *args) + func(upper_limit, *args)
    res += (coefficient * res_intermediate).sum(dim=0)
    res *= (upper_limit - lower_limit) / n / 3.0


    ### -------

    # res = func(lower_limit, *args) + func(upper_limit, *args)
    # delta_x = (upper_limit - lower_limit) / n
    # debugs = []
    # for i in range(1, n):
    #     if i % 2 != 0:
    #         coefficient = 4
    #     else:
    #         coefficient = 2
    #     res += coefficient * func(lower_limit + i*delta_x, *args)
    #     debugs.append(i*delta_x)
    #     # print(i*delta_x, i, coefficient, func(lower_limit + i*delta_x, *args))
    # res *= delta_x / 3.0
    return res

