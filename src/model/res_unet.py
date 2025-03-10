"""
Resnet - variation of Unet
Current version without context embedding
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


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
          
        



class res_block(nn.Module):

    def __init__(self,in_channel,out_channel,activation,normalization):

        return None
