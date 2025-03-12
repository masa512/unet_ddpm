import torch
import torch.nn as nn

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
        self.Tmax = Tmax
        self.beta_min = beta_min
        self.beta_max = beta_max
    
    def get_params(self,_t):
        """
        _t (Batch,1)
        """
        params = {
            k : torch.index_select(v,-1,_t.squeeze()) for k, v in zip(self.params.keys(),self.params.values())
        }
        return params

    def get_max_time(self):

        return self.Tmax
    
    def get_beta_range(self):

        return (self.beta_min,self.beta_max)

    
    def diffuse(self,x_0,_t):

        # White noise (same as x : N,Nc,W,L)
        noise = torch.randn_like(x_0)

        # Retrive the params as needed (each param is of batch x 1)
        params = self.get_params(_t)
        alpha_bar = params['alpha_bar'].view(-1,1,1,1)

        # Apply transformation as needed
        x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1-alpha_bar) * noise

        return x_t,noise