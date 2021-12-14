
import numpy as np
import copy

from ..basic import Data
from ..augment import State
from .base import BaseCurve, BaseCurves, BaseCurveOld
from .functions import spiral


class QuadraticSpiral_v0(object):
    Spiral = spiral.QuadraticSpiral

    max_phi = np.deg2rad(50)
    max_radius = 15.0
    min_radius = 13.0

    delta_s = 0.1  ### meter

    '''
        Do not support backup.
    '''
    def __init__(self):
        self.h_vector = np.array([0.05, 0.05, 0.1]).reshape(3, 1)
        self.threshold = 0.05

        self.step_size = 0.7
        self.use_fixed_step_size = True
        self.max_iter = 10
    

    def run_step(self, radius, phi, k0):
        '''
            radius: [-1, 1]
            phi: [-1, 1]
        '''

        radius = 0.5 * ((self.max_radius-self.min_radius) * radius + self.max_radius + self.min_radius)
        phi = phi * self.max_phi

        x, y = radius * np.cos(phi), radius * np.sin(phi)
        theta = phi
    
        state = np.array([x, y, theta]).reshape(3,-1)
        param_initial = np.array([0.0, 0.0, radius]).reshape(3, 1)
        param, min_cost_state, cost = self.solve_curve(state, param_initial, k0)

        base_curve = self.get_curve(k0, param)
        info = Data(param=param, min_cost_state=min_cost_state, cost=cost)
        return base_curve, info


    def solve_curve(self, state_constraint_vector, param_initial, k0):
        max_iter = self.max_iter

        param = copy.copy(param_initial)
        state_vector = self.get_state_vector(k0, param)

        min_cost_param = param
        min_cost_state_vector = state_vector
        min_cost = np.linalg.norm(state_constraint_vector - min_cost_state_vector)

        for i in range(max_iter):
            delta_state = state_constraint_vector - state_vector

            cost = np.linalg.norm(delta_state)
            if cost < min_cost:
                min_cost = cost
                min_cost_param = param
                min_cost_state_vector = state_vector

            if cost <= self.threshold:
                break

            J = self.calculate_Jacobian(k0, param)
            delta_param = - np.dot( np.linalg.inv(J), delta_state )
            if self.use_fixed_step_size == True:
                param -= self.step_size * delta_param
            else:
                step_size = self.get_optimal_step_size(k0, state_constraint_vector, param, delta_param)
                param -= step_size * delta_param

            state_vector = self.get_state_vector(k0, param)

        return min_cost_param, min_cost_state_vector, min_cost


    def get_state_vector(self, k0, param, ratio=1.0):
        l = param[2][0]
        standard_param = self.standard_param_from_param(k0, param)
        x = self.Spiral.x(ratio*l, standard_param)
        y = self.Spiral.y(ratio*l, standard_param)
        theta = self.Spiral.theta(ratio*l, standard_param)
        return np.array([x, y, theta]).reshape(3, 1)
    
    def get_state_vector_with_k(self, k0, param, ratio=1.0):
        l = param[2][0]
        standard_param = self.standard_param_from_param(k0, param)
        x = self.Spiral.x(ratio*l, standard_param)
        y = self.Spiral.y(ratio*l, standard_param)
        theta = self.Spiral.theta(ratio*l, standard_param)
        k = self.Spiral.curvature(ratio*l, standard_param)
        return np.array([x, y, theta, k]).reshape(4, 1)


    def calculate_Jacobian(self, k0, param):
        km, kf, l = param[0], param[1], param[2]

        J_column0 = ( self.get_state_vector(k0, [km+self.h_vector[0][0], kf, l]) \
            - self.get_state_vector(k0, [km-self.h_vector[0][0], kf, l]) ) / (2*self.h_vector)
        J_column1 = ( self.get_state_vector(k0, [km, kf+self.h_vector[1][0], l]) \
            - self.get_state_vector(k0, [km, kf-self.h_vector[1][0], l]) ) / (2*self.h_vector)
        J_column2 = ( self.get_state_vector(k0, [km, kf, l+self.h_vector[2][0]]) \
            - self.get_state_vector(k0, [km, kf, l-self.h_vector[2][0]]) ) / (2*self.h_vector)

        J = np.hstack((J_column0, J_column1, J_column2))
        return J


    def standard_param_from_param(self, k0, param):
        '''
            param: shape is (3,1)
        '''
        km, kf, l = param[0][0], param[1][0], param[2][0]
        return np.array([k0, (4*km-kf-3*k0)/l, 2*(k0+kf-2*km)/l**2]).reshape(3, 1)
    

    def get_curve(self, k0, param):
        l = param[2][0]
        states = []
        for i in np.linspace(0, l, int(l / self.delta_s) ):
            p = self.get_state_vector_with_k(k0, param, i /l)
            x, y, theta, k = p[0][0], p[1][0], p[2][0], p[3][0]
            states.append( State(x=x, y=y, theta=theta, k=k) )
        return BaseCurveOld(states, sampling_resolution=self.delta_s)





# =============================================================================
# -- parallel  ----------------------------------------------------------------
# =============================================================================




import torch


class QuadraticSpiral_v1(object):
    Spiral = spiral.Spiral

    max_phi = np.deg2rad(50)
    max_radius = 15.0
    min_radius = 13.0

    delta_s = 0.1  ### meter

    '''
        Do not support backup.
    '''
    def __init__(self):
        self.solver = QuadraticSpiralSolver(self.Spiral)
    

    def run_step(self, radius, phi, k0):
        '''
            radius, phi, k0: torch.Size([batch_size])
            radius: [-1, 1]
            phi: [-1, 1]
        '''

        radius = 0.5 * ((self.max_radius-self.min_radius) * radius + self.max_radius + self.min_radius)
        phi = phi * self.max_phi

        x, y = radius * np.cos(phi), radius * np.sin(phi)
        theta = phi
        
        param, info = self.solver.solve(k0, x,y,theta)
        base_curves = self.get_curves(k0, info.solution)
        return base_curves, info


    def get_curves(self, k0, solution):
        """
            k0: torch.Size([batch_size])
            solution: torch.Size([batch_size, order+1])
        """

        ratios = torch.linspace(0, 1, 200).unsqueeze(0)
        end_states = self.solver.get_end_state_with_k(k0, solution, ratios)
        end_states = list([*end_states])
        return BaseCurves(end_states)


    def get_curves_old(self, k0, solution):  ## deprecated
        param = self.solver.solution2param(k0, solution)
        base_curves = []
        for k00, s, p in zip(k0, solution, param):
            l = s[2]
            ratios = torch.linspace(0, l, int(l / self.delta_s) ) / l
            ratios = ratios.unsqueeze(0)
            k00 = k00.unsqueeze(0)
            s = s.unsqueeze(0)
            p = p.unsqueeze(0)
            end_states = self.solver.get_end_state_with_k(k00, s, ratios).squeeze(0)
            states = []
            for (x,y,theta,k) in end_states.T:
                states.append( State(x=x, y=y, theta=theta, k=k) )
            base_curves.append( BaseCurveOld(states, sampling_resolution=self.delta_s) )
        return base_curves



class QuadraticSpiralSolver(object):
    """
        Running in local coordinate.
    """

    def __init__(self, Spiral):
        self.Spiral: spiral.Spiral = Spiral
        self.h_vector = torch.tensor([0.05, 0.05, 0.1])
        self.threshold = 0.05

        self.step_size = 0.7
        self.max_iter = 10


    @staticmethod
    def solution2param(k0, solution):
        km, kf, l = solution[:,0], solution[:,1], solution[:,2]
        return torch.stack([k0, (4*km-kf-3*k0)/l, 2*(k0+kf-2*km)/l**2], dim=1)


    def solve(self, k0, x, y, theta):
        """
        Args:
            k0, x, y, theta: torch.Size([batch_size])
        """

        end_state_constraint = torch.stack([x, y, theta], dim=1)

        radius = torch.hypot(x, y)
        solution = torch.stack([
            torch.zeros_like(radius),
            torch.zeros_like(radius),
            radius,
        ], dim=1)
        end_state = self.get_end_state(k0, solution)
        cost = None
        for i in range(self.max_iter):
            delta_state = end_state_constraint - end_state

            cost = torch.linalg.norm(delta_state, dim=1)
            if cost.max().item() <= self.threshold:
                break
                
            J = self.calculate_Jacobian(k0, solution)
            delta_solution = -torch.bmm(torch.inverse(J), delta_state.unsqueeze(2))
            solution -= self.step_size * delta_solution.squeeze(2)

            end_state = self.get_end_state(k0, solution)

        param = self.solution2param(k0, solution)
        info = Data(solution=solution, cost=cost)
        return param, info


    def get_end_state(self, k0, solution, ratio=1.0):
        l = solution[:,2].unsqueeze(1)
        param = self.solution2param(k0, solution)
        x = self.Spiral.x(ratio*l, param)
        y = self.Spiral.y(ratio*l, param)
        theta = self.Spiral.theta(ratio*l, param)
        return torch.stack([x, y, theta], dim=1).squeeze(-1)

    def get_end_state_with_k(self, k0, solution, ratio=1.0):
        """
            k0: torch.Size([batch_size])
            solution: torch.Size([batch_size, order+1])
            ratio: torch.Size([batch_size, num_points])
        """

        l = solution[:,2].unsqueeze(1)  ### torch.Size([batch_size, 1])
        param = self.solution2param(k0, solution)  ### torch.Size([batch_size, order+1])
        x = self.Spiral.x(ratio*l, param)
        y = self.Spiral.y(ratio*l, param)
        theta = self.Spiral.theta(ratio*l, param)
        k = self.Spiral.curvature(ratio*l, param)
        return torch.stack([x, y, theta, k], dim=1).squeeze(-1)  ## ! warning: why squeeze here?


    def calculate_Jacobian(self, k0, solution):
        km, kf, l = solution[:,0], solution[:,1], solution[:,2]

        J_column0 = ( self.get_end_state(k0, torch.stack([km+self.h_vector[0], kf, l], dim=1) )    \
                    - self.get_end_state(k0, torch.stack([km-self.h_vector[0], kf, l], dim=1) ) )  \
                    / (2*self.h_vector)

        J_column1 = ( self.get_end_state(k0, torch.stack([km, kf+self.h_vector[1], l], dim=1) )    \
                    - self.get_end_state(k0, torch.stack([km, kf-self.h_vector[1], l], dim=1) ) )  \
                    / (2*self.h_vector)

        J_column2 = ( self.get_end_state(k0, torch.stack([km, kf, l+self.h_vector[2]], dim=1) )    \
                    - self.get_end_state(k0, torch.stack([km, kf, l-self.h_vector[2]], dim=1) ) )  \
                    / (2*self.h_vector)

        J = torch.stack([J_column0, J_column1, J_column2], dim=2)
        return J
