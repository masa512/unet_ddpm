import torch
import torch.nn.functional as F 
import torch.nn as nn
######## Pure Unet Implementation ###############

class cbr_block(nn.Module):

    def __init__(self,in_channels,out_channels,kernel_size,include_relu = True ,include_bn = True):

        super().__init__()

        # Initialize CBR Block with just conv
        self.cbr = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride = 1,
                padding = (kernel_size-1)//2
            )
        )

        # Add relu if needed
        if include_relu:
            self.cbr.append(
                nn.ReLU()
            )

        # Add bn if needed
        if include_bn:
            self.cbr.append(
                nn.BatchNorm2d(num_features=out_channels)
            )

    def forward(self,x):

        return self.cbr(x)

class double_cbr_block(nn.Module):

    def __init__(self,in_channels,out_channels,kernel_size,include_relu = True ,include_bn = True):

        super().__init__()

        # initialize the two cbr blocks
        self.cbr1 = cbr_block(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size = kernel_size,
            include_relu= include_relu,
            include_bn = include_bn
        )

        self.cbr2 = cbr_block(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size = kernel_size,
            include_relu= include_relu,
            include_bn = include_bn
        )
    
    def forward(self,x):

        x = self.cbr1(x)
        x = self.cbr2(x)
        return x

class bottle_neck(nn.Module):

    def __init__(self,in_channels,kernel_size,include_relu = True ,include_bn = True):

        super().__init__()

        self.bottle = double_cbr_block(
                in_channels= in_channels,
                out_channels= 2*in_channels,
                kernel_size = kernel_size,
                include_relu= include_relu,
                include_bn = include_bn
        )
    
    def forward(self,x):

        return self.bottle(x)



class encoder(nn.Module):
    
    def __init__(self,input_channels,in_channels, kernel_size, depth = 1 ,include_relu = True ,include_bn = True):

        super().__init__()

        # Depth excludes the input block

        # Define input block
        self.input_block = double_cbr_block(
                in_channels= input_channels,
                out_channels= in_channels,
                kernel_size = kernel_size,
                include_relu= include_relu,
                include_bn = include_bn
        )
        
        self.enc_seq = nn.ModuleList()

        for i in range(1,depth+1):

            self.enc_seq.add_module(f'enc{i}',
                double_cbr_block(
                in_channels= in_channels * (2**(i-1)),
                out_channels= in_channels * (2**(i)),
                kernel_size = kernel_size,
                include_relu= include_relu,
                include_bn = include_bn
            )
            )
        
        self.pool = nn.MaxPool2d(2,2)
    
    def forward(self,x):

        # Initialize
        res = {}

        # Input Layer
        r = self.input_block(x)
        res[0] = r
        x = self.pool(r)

        # each encoder block
        for i, e in enumerate(self.enc_seq):
            # First apply the conv layer
            r = e(x)
            res[i+1] = r
            # Apply pool
            x = self.pool(r)
        
        return x,res

class Interpolate(nn.Module):
    def __init__(self, scale, mode):
        super().__init__()
        self.interp = nn.functional.interpolate
        self.scale = scale # Int
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, scale=self.scale, mode=self.mode, align_corners=False)
        return x

class decoder(nn.Module):

    def __init__(self,in_channels, output_channels, kernel_size, depth = 1, include_relu = True ,include_bn = True):
        super().__init__()
        self.depth = depth

        # Add the module list for non-final layers
        self.upsample = nn.ModuleList(
            [nn.ConvTranspose2d(in_channels=in_channels//(2**(i)),
                              out_channels=in_channels//(2**(i+1)),
                              kernel_size=2,
                              stride = 2
                            ) for i in range(depth)]
        )
        self.dec_conv = nn.ModuleList(
            [double_cbr_block(in_channels=in_channels//(2**(i)),
                              out_channels=in_channels//(2**(i+1)),
                              kernel_size=kernel_size,
                              include_relu= include_relu,
                              include_bn = include_bn

                            ) for i in range(depth)]
        )

        # Append the ones for the final layer

        self.upsample.append(
            nn.ConvTranspose2d(in_channels=in_channels//(2**(depth)),
                              out_channels=in_channels//(2**(depth+1)),
                              kernel_size=2,
                              stride = 2)
        )
        self.dec_conv.append(
            double_cbr_block(in_channels=in_channels//(2**(depth)),
                              out_channels=in_channels//(2**(depth+1)),
                              kernel_size=kernel_size,
                              include_relu= include_relu,
                              include_bn = include_bn
                            )
        )


        self.output_conv = nn.Conv2d(
            in_channels=in_channels//(2**(depth+1)),
            out_channels=output_channels,
            kernel_size= 1,
            stride = 1,
            padding = 0
        )

    def forward(self,x,res):
        # APPLY THE Decoding layers
        for r,u,d in zip(reversed(res.values()),self.upsample,self.dec_conv):

            # First the upsampe before combining
            x_u = u(x)
            # Concatenate
            x_c = torch.cat((x_u,r), dim=1) # Channel dim cat
            # Apply channel convolution
            x = d(x_c)
        
        # APPLY Single Conv to match channel
        return self.output_conv(x)


class unet(nn.Module):

    def __init__(self, input_channels, base_channels, output_channels, kernel_size, depth, include_relu = True ,include_bn = True):

        super().__init__()

        # Define the encoder,decoder, bottle neck

        enc_out_channels = base_channels * (2**(depth))
        dec_in_channels = base_channels * (2**(depth+1))

        self.encoder = encoder(input_channels,base_channels, kernel_size, depth ,include_relu ,include_bn)
        self.bottle_neck = bottle_neck(enc_out_channels,kernel_size,include_relu,include_bn)
        self.decoder = decoder(dec_in_channels, output_channels, kernel_size, depth, include_relu ,include_bn)

    
    def forward(self,x):

        # Encoder pass
        y, res = self.encoder(x)
        # Bottleneck pass
        y = self.bottle_neck(y)
        # Decoder pass
        out = self.decoder(y,res)

        return out