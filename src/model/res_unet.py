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
from .fwd_diffusion import forward_diffusion

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
    def __init__(self,sinu_emb_dim ,out_channels, activation = nn.SiLU):

        super().__init__()
        self.sinu_emb_dim = sinu_emb_dim
        self.t_emb_layer = nn.Sequential(
            activation(),
            nn.Linear(
                in_features=sinu_emb_dim,
                out_features =  out_channels)
        )
    
    def forward(self,t_emb):

        ### We'll expect a time tensor [0,Tmax) of size (batch,1)

        # Step 2 : Pass the output to NN
        embedding = self.t_emb_layer(t_emb)

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
        self.t_embedder = time_embedding(sinu_emb_dim,out_channels,activation)

    
    def forward(self,x, t_emb):

        # Step 1 : Pass input through first nac and residual
        res,x = self.residual_layer(x),self.nac1(x)

        # Step 2 : Apply nonlinear embedder to the sinusoidal embedding vector t_emb
        t_vector = self.t_embedder(t_emb)[:,:,None,None]

        # Make addable to the image features (W,L)
        x =  t_vector + x

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
        residual_layer = True, 
        activation=nn.SiLU, 
        normalization=nn.GroupNorm,
        pool = nn.AvgPool2d,
        norm_kwargs = {}
        ):
        super().__init__()

        self.enc_seq = nn.ModuleList()

        for i in range(depth):

            # Append Sequence of 1) Res Block 2) Pool (either avg or max)
            self.enc_seq.append(
                resnet_block(
                    in_channels = base_channels * 2**(i),
                    out_channels = base_channels * 2**(i+1),
                    kernel_size = kernel_size,
                    sinu_emb_dim = sinu_emb_dim,
                    residual_layer = residual_layer, 
                    activation=activation, 
                    normalization= normalization,
                    norm_kwargs = norm_kwargs
                ))
        
        self.pool = pool(2,2)

    def forward(self,x,t_emb):

        # Initialize skip connection
        res = {}

        # each encoder block
        for i, e in enumerate(self.enc_seq):
            # First apply the conv layer
            r = e(x,t_emb)
            res[i+1] = r
            # Apply pool
            x = self.pool(r)
        
        return x,res

class res_decoder(nn.Module):

    def __init__(
        self,
        base_channels,
        kernel_size,
        depth,
        sinu_emb_dim,
        residual_layer = True, 
        activation=nn.SiLU, 
        normalization=nn.GroupNorm,
        upsample_conv = nn.ConvTranspose2d,
        norm_kwargs = {}
        ):

        super().__init__()

        self.dec_seq = nn.ModuleList()
        self.upsample = nn.ModuleList()

        for i in reversed(range(1,depth+1)):

            self.dec_seq.append(
                resnet_block(
                    in_channels = base_channels * 2**(i+1),
                    out_channels = base_channels * 2**(i),
                    kernel_size = kernel_size,
                    sinu_emb_dim = sinu_emb_dim,
                    residual_layer = residual_layer, 
                    activation=activation, 
                    normalization= normalization,
                    norm_kwargs = norm_kwargs
                )
            )

            self.upsample.append(
                upsample_conv(
                    in_channels=base_channels * (2**(i+1)),
                    out_channels=base_channels * (2**(i)),
                    kernel_size=2,
                    stride = 2
                )
            )
        

    def forward(self,x,t_emb,res):

        for d,u,r in zip(self.dec_seq,self.upsample,reversed(res.values())):

            # Upsample first
            x_u = u(x)
            # Cat the residual
            x_c = torch.cat((x_u,r), dim=1)
            # Conv layer
            x = d(x_c,t_emb)

        return x


class res_bottle(nn.Module):

    def __init__(
        self,
        base_channels,
        kernel_size,
        depth,
        sinu_emb_dim,
        residual_layer = True, 
        activation=nn.SiLU, 
        normalization=nn.GroupNorm,
        norm_kwargs = {}
        ):

        super().__init__()

        # Two unique time embedding layers already absorbed in the res layer defined
        self.res_block1 = resnet_block(
                    in_channels = base_channels * (2**depth),
                    out_channels = base_channels * (2**(depth+1)),
                    kernel_size = kernel_size,
                    sinu_emb_dim = sinu_emb_dim,
                    residual_layer = residual_layer, 
                    activation=activation, 
                    normalization= normalization,
                    norm_kwargs = norm_kwargs
                    )
        self.res_block2 = resnet_block(
                    in_channels = base_channels * (2**(depth+1)),
                    out_channels = base_channels * (2**(depth+1)),
                    kernel_size = kernel_size,
                    sinu_emb_dim = sinu_emb_dim,
                    residual_layer = residual_layer, 
                    activation=activation, 
                    normalization= normalization,
                    norm_kwargs = norm_kwargs
                    )

    def forward(self,x,t_emb):

        x = self.res_block1(x,t_emb)
        x = self.res_block2(x,t_emb)

        return x

class input_layer(nn.Module):

    def __init__(
        self,
        input_channels,
        base_channels,
        kernel_size
        ):

        super().__init__()

        self.layer = nn.Conv2d(
            in_channels=input_channels,
            out_channels=base_channels,
            kernel_size=kernel_size,
            stride = 1,
            padding = (kernel_size-1)//2
        )
    
    def forward(self,x):

        return self.layer(x)


class output_layer(nn.Module):

    def __init__(
        self,
        base_channels,
        output_channels,
        kernel_size,
        activation=nn.SiLU, 
        normalization=nn.GroupNorm,
        norm_kwargs = {}
        ):

        super().__init__()

        self.nac = norm_act_conv(
          in_channels = base_channels * 2,
          out_channels = base_channels,
          kernel_size = kernel_size,
          activation=activation, 
          normalization=normalization, 
          norm_kwargs = norm_kwargs
        )

        self.final_layer = nn.Conv2d(
            in_channels=base_channels,
            out_channels=output_channels,
            kernel_size=kernel_size,
            stride = 1,
            padding = (kernel_size-1)//2
        )
    
    def forward(self,x):
        return self.final_layer(self.nac(x))

class res_unet(nn.Module):

    def __init__(
        self,
        input_channels,
        base_channels,
        output_channels,
        kernel_size,
        depth,
        sinu_emb_dim,
        residual_layer = True, 
        activation=nn.SiLU, 
        normalization=nn.GroupNorm,
        pool = nn.AvgPool2d,
        upsample_conv = nn.ConvTranspose2d,
        norm_kwargs = {}
        ):
        super().__init__()

        self.input_layer = input_layer(
          input_channels = input_channels,
          base_channels = base_channels,
          kernel_size = kernel_size
        )
        
        self.encoder = res_encoder(
            base_channels = base_channels,
            kernel_size = kernel_size,
            depth = depth,
            sinu_emb_dim = sinu_emb_dim,
            residual_layer = residual_layer, 
            activation= activation, 
            normalization=normalization,
            pool = pool,
            norm_kwargs = norm_kwargs
        )

        self.bottle_neck = res_bottle(
            base_channels = base_channels,
            kernel_size = kernel_size,
            depth = depth,
            sinu_emb_dim = sinu_emb_dim,
            residual_layer = residual_layer, 
            activation=activation, 
            normalization=normalization,
            norm_kwargs = norm_kwargs
        )

        self.decoder = res_decoder(
            base_channels = base_channels,
            kernel_size = kernel_size,
            depth = depth,
            sinu_emb_dim = sinu_emb_dim,
            residual_layer = residual_layer, 
            activation=activation, 
            normalization=normalization,
            upsample_conv = upsample_conv,
            norm_kwargs = norm_kwargs
        )

        self.output_layer = output_layer(
            base_channels = base_channels,
            output_channels = output_channels,
            kernel_size = kernel_size,
            activation=activation, 
            normalization=normalization,
            norm_kwargs = norm_kwargs
        )

    def forward(self,x,t_emb):

        x = self.input_layer(x)
        x,res = self.encoder(x,t_emb)
        x = self.bottle_neck(x,t_emb)
        x = self.decoder(x,t_emb,res)
        x = self.output_layer(x)

        return x

    

