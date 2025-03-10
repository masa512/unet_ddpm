"""
Resnet - variation of Unet
Current version without context embedding
"""
import torch
from torch.autograd import forward_ad
import torch.nn as nn
import torch.nn.functional as F
from .embedding import pos_encoder


class norm_act_conv(nn.Module):

    def __init__(self,in_channels,out_channels,kernel_size,activation=nn.SiLU, normalization=nn.GroupNorm, norm_kwargs = {}):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride = 1,
            padding = (kernel_size-1)//2
        )
        self.normalize = nn.Identity()
        self.activation = nn.Identity()
        if normalization:
            if normalization == nn.GroupNorm:
                norm_kwargs['num_channels'] = in_channels

                if 'num_groups' not in norm_kwargs:
                    norm_kwargs['num_groups'] = in_channels
                
            elif normalization == nn.BatchNorm2d:
                _ = norm_kwargs.pop('num_channels')
                _ = norm_kwargs.pop('num_groups')
                norm_kwargs['num_features'] = in_channels 
               
            self.normalize = normalization(**norm_kwargs)
          
        if activation:
            self.activation = activation()
    
    def forward(self,x):

        return self.conv(self.activation(self.normalize(x)))

class time_embedding(nn.Module):
    """
    Includes 1)Sinusoidal Encoding & 2)NN processing
    """
    def __init__(self,sinu_emb_dim ,out_channels,Tmax, activation = nn.SiLU):

        super(time_embedding,self).__init__()
        self.Tmax = Tmax
        self.sinu_emb_dim = sinu_emb_dim
        self.sinu_emb = pos_encoder(Tmax,sinu_emb_dim)
        self.t_emb_layer = nn.Sequential(
            activation(),
            nn.Linear(
                in_features=sinu_emb_dim,
                out_channels =  out_channels)
        )
    
    def forward(self,_t):

        ### We'll expect a time tensor [0,Tmax) of size (batch,1)

        # Step 1 : Extract the embeddings for each batch
        sinu_embedding = torch.index_select(self.sinu_emb,dim=0,index=_t)

        # Step 2 : Pass the output to NN
        embedding = self.t_emb_layer(sinu_embedding)

        return embeddingß

