import torch
import torch.nn as nn
import math


class forward_diffusion():

    def __init__(self,Tmax,beta_min,beta_max):
        # Scheduling happens here
        beta = torch.linspace(start = beta_min, end = beta_max, steps = Tmax).view(1,-1)
        alpha = 1-beta

        alpha_bar = torch.zeros_like(alpha)
        alpha_bar[:,0] = alpha[:,0]
        for i in range(1,alpha_bar.size(-1)):
            alpha_bar[:,i] = alpha_bar[:,i-1] * alpha[:,i]

        self.params = {
            "alpha" : alpha,
            "beta" : beta,
            "alpha_bar" : alpha_bar
        }
    
    def get_param(self,_t):
        """
        _t (Batch,1)
        """
        params = {
            k : torch.index_select(v,-1,_t.squeeze()) for k, v in zip(self.params.keys(),self.params.values())
        }
        return params
    
    def forward(self,x_0,_t):

        # White noise (same as x : N,Nc,W,L)
        noise = torch.randn_like(x_0)

        # Retrive the params as needed (each param is of batch x 1)
        params = self.get_param(_t)
        alpha_bar = params['alpha_bar'].view(-1,1,1,1)

        # Apply transformation as needed
        x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1-alpha_bar) * noise

        return x_t

        

def alpha_beta_scheduler_torch(Tmax = 1000,beta_min = 1e-4,beta_max = 3e-3):
    """
    Evaluates all betas/alphas possible and returns as np.array (Torch V)

    ---Input---
    Tmax : (int) Maximum iteration
    beta_min/beta_max : (int) Max and min of beta values

    ---Output---
    betas : (np.array Tmax,) - Scheduled beta
    """

    betas = torch.linspace(start = beta_min, end = beta_max, steps = Tmax)
    alphas = 1.0 - betas
    return alphas,betas

def advance_diffusion(data, t = 0,beta_min = 1e-4, beta_max = 0.02, Tmax = 1000):
    """
    This function directly transforms the original data x_0 to t-th stage forward transform (close-form)
    linear variance schedule is used (Torch)

    -Input-
    data (n_dims,) : Single Input data 
    beta_min,beta_max (double) : Bound for beta for variance scheduling
    t (int) : Target iteration
    Tmax (int) : Max iteration

    -Output-
    target_data (n_dims,) : Single output data at epoch t 
    eps (n_dims,) : Noise used to distort (shall be predicted from NN)
    """
    # Initialize beta and alpha
    alphas,betas = alpha_beta_scheduler_torch(Tmax = Tmax, beta_min = beta_min, beta_max = beta_max)

    # First extract single beta and get alpha
    beta_t = betas[t]
    alpha_t = alphas[t]

    # Products to get alpha_geom_t
    alpha_geom_t = torch.prod(alphas[:t], dim=0)

    # Epsilon
    eps = torch.randn_like(data)

    # Evaluate x_t
    x_t = math.sqrt(alpha_geom_t) * data + math.sqrt(1-alpha_geom_t) * eps
    return x_t, eps