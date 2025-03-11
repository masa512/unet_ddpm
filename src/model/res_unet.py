"""
Resnet - variation of Unet
Current version without context embedding
Inspiration from https://matthewmacfarquhar.medium.com/de-noising-diffusion-probabalistic-models-21f9adf586b0

"""
import torch
from torch.autograd import forward_ad
import torch.nn as nn
import torch.nn.functional as F
from .embedding import pos_encoder


class norm_act_conv(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        activation=nn.SiLU, 
        normalization=nn.GroupNorm, 
        norm_kwargs = {}
        ):

        super(norm_act_conv,self).__init__()
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

        super().__init__()
        self.Tmax = Tmax
        self.sinu_emb_dim = sinu_emb_dim
        self.sinu_emb = pos_encoder(Tmax,sinu_emb_dim)
        self.t_emb_layer = nn.Sequential(
            activation(),
            nn.Linear(
                in_features=sinu_emb_dim,
                out_features =  out_channels)
        )
    
    def forward(self,_t):

        ### We'll expect a time tensor [0,Tmax) of size (batch,1)

        # Step 1 : Extract the embeddings for each batch
        sinu_embedding = torch.index_select(self.sinu_emb,dim=0,index=_t.squeeze())

        # Step 2 : Pass the output to NN
        embedding = self.t_emb_layer(sinu_embedding)

        return embedding

class resnet_block(nn.Module):

    """
    Equivlent to double_conv in Unet
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        sinu_emb_dim,
        Tmax, 
        residual_layer = True, 
        activation=nn.SiLU, 
        normalization=nn.GroupNorm,
        norm_kwargs = {}):

        super().__init__()
        
        # Define residual layer
        self.residual_layer  = nn.Identity()
        if residual_layer:
            self.residual_layer = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1)

        # Define the two norm_act_conv layers 
        self.nac1 = norm_act_conv(in_channels,out_channels,kernel_size,activation, normalization, norm_kwargs = {})
        self.nac2 = norm_act_conv(out_channels,out_channels,kernel_size,activation, normalization, norm_kwargs = {})
        
        # Define the time embedding layer
        self.t_emb = time_embedding(sinu_emb_dim,out_channels,Tmax,activation)

    
    def forward(self,x,_t):

        # Step 1 : Pass input through first nac and residual
        res,x = self.residual_layer(x),self.nac1(x)

        # Step 2 : Add the first time embedding to x

        t_vector = self.t_emb(_t)
        # Make addable to the image features (W,L)
        t_resized = t_vector.unsqueeze(-1).unsqueeze(-1)

        x =  t_resized + x

        # Step 3 : Apply the last nac
        x = self.nac2(x)

        # Step 4 : Add residual connection to x and return
        x = x + res
        return x

class res_encoder(nn.Module):
    """
    Full U-Net Encoder Layer
    Depth excludes the input layer (same goes for decoder regarding the output layer)
    """
    def __init__(
        self,
        base_channels,
        kernel_size,
        depth,
        sinu_emb_dim,
        Tmax, 
        residual_layer = True, 
        activation=nn.SiLU, 
        normalization=nn.GroupNorm,
        pool = nn.AvgPool2d,
        norm_kwargs = {}
        ):

        self.enc_seq = nn.ModuleList()

        for i in range(1,depth+1):

            # Append Sequence of 1) Res Block 2) Pool (either avg or max)
            self.enc_seq.append(nn.Sequential(
                resnet_block(
                    in_channels = base_channels * 2**(i-1),
                    out_channels = base_channels * 2**(i),
                    kernel_size = kernel_size,
                    sinu_emb_dim = sinu_emb_dim,
                    Tmax = Tmax, 
                    residual_layer = residual_layer, 
                    activation=activation, 
                    normalization= normalization,
                    norm_kwargs = norm_kwargs
                )
            ))
        
        self.pool = pool(2,2)

    def forward(self,x,_t):

        # Initialize skip connection
        res = {}

        # each encoder block
        for i, e in enumerate(self.enc_seq):
            # First apply the conv layer
            r = e(x,_t)
            res[i+1] = r
            # Apply pool
            x = self.pool(r)
        
        return x,res


