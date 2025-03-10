import torch
import math

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