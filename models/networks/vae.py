import numpy as np 
import torch
import torch.nn as nn

# definition of elements for VAE model
class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Conv, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv(x)
    
class ConvTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvTranspose, self).__init__()


        self.conv = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv(x)

class VAE(nn.Module):
    
    #  similar to this model: https://ieeexplore.ieee.org/abstract/document/9119450
    def __init__(self, opt):
        super(VAE, self).__init__()
        # torch.manual_seed(opt.seed)
        self.img_size = opt.crop_size
        self.nef = 64   # number of encoder filters
        self.ndf = 64   # number of decoder filters
        self.input_nc = opt.input_nc
        self.output_nc = opt.label_nc
        self.z_dim = opt.zdim   # the sized of the latent code
        self.bneck = 256
        self.conv_block_1 = nn.Sequential(Conv(self.input_nc, self.nef, 3, stride=2, padding=1), Conv( self.nef, self.nef, 3, stride=1, padding=1),
            Conv( self.nef, self.nef, 3, stride=1, padding=1))

        self.conv_block_2 = nn.Sequential(Conv(self.nef, 2* self.nef, 3, stride=2, padding=1), Conv( 2*self.nef, 2*self.nef, 3, stride=1, padding=1),
            Conv( 2*  self.nef, 2* self.nef, 3, stride=1, padding=1))

        self.conv_block_3 = nn.Sequential(Conv(2*self.nef, 4* self.nef, 3, stride=2, padding=1), Conv( 4* self.nef, 4* self.nef, 3, stride=1, padding=1),
            Conv( 4*  self.nef, 4* self.nef, 3, stride=1, padding=1))

        self.conv_block_4 = nn.Sequential(Conv(4*self.nef, 8* self.nef, 3, stride=2, padding=1), Conv( 8*self.nef, 8* self.nef, 3, stride=1, padding=1),
            Conv( 8*  self.nef, 8* self.nef, 3, stride=1, padding=1))

        # self.conv_layer_6 = Conv(16*self.nef, 32*self.nef, 3, stride=2, padding=1)
        if self.img_size == 128:
            s = 8
        elif self.img_size == 64:
            s = 4
        else:
            raise RuntimeError(' the image size is not supported') 
        self.fc_bneck = nn.Linear(8 * self.nef * s * s, self.bneck)
        self.bneck_bneck = nn.Linear(self.bneck, self.bneck)

        self.encoder_mu = nn.Linear(self.bneck, self.z_dim)
        self.encoder_logvar = nn.Linear(self.bneck, self.z_dim)

        self.fc_z_bneck = nn.Linear(self.z_dim, self.bneck)
        self.fc_bneck_bneck = nn.Linear(self.bneck, self.bneck)


        self.fc_z1 = nn.Linear(self.bneck, 8 * self.ndf * s * s)

        # self.conv_t_block_1 = nn.Sequential( ConvTranspose(8 * self.ndf , 8 * self.ndf , 2, stride=2, padding=1),
        # ConvTranspose(8 * self.ndf , 4 * self.ndf , 3, stride=3, padding=1))

        self.conv_t_block_1 = nn.Sequential( nn.UpsamplingBilinear2d(scale_factor=2), Conv(8 * self.ndf , 8 * self.ndf , 3, stride=1, padding=1),
        Conv(8 * self.ndf , 4 * self.ndf , 3, stride=1, padding=1))
        self.conv_t_block_2 = nn.Sequential( nn.UpsamplingBilinear2d(scale_factor=2), Conv(4 * self.ndf , 4 * self.ndf , 3, stride=1, padding=1),
        Conv(4 * self.ndf , 2 * self.ndf , 3, stride=1, padding=1))
        self.conv_t_block_3 = nn.Sequential( nn.UpsamplingBilinear2d(scale_factor=2), Conv(2 * self.ndf , 2 * self.ndf , 3, stride=1, padding=1), 
        Conv(2 * self.ndf , 1 * self.ndf , 3, stride=1, padding=1 ))
        self.conv_t_block_4 = nn.Sequential( nn.UpsamplingBilinear2d(scale_factor=2), Conv(1 * self.ndf , 1 * self.ndf , 3, stride=1, padding=1), 
        Conv(1 * self.ndf , 1 * self.ndf , 3, stride=1, padding=1))


        self.last_layer = nn.Sequential(
            nn.Conv2d( self.ndf, int(self.ndf/2), 3, padding=1),
            nn.BatchNorm2d(int(self.ndf/2)),
            nn.Conv2d(int(self.ndf/2), self.output_nc, 3, padding=1),
            # nn.Tanh()
            # nn.Sigmoid()
            # nn.Softmax2d()
        )
        
    def encode(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.conv_block_4(x)

        x = x.view(x.shape[0], -1)
        x = self.fc_bneck(x)
        x = self.bneck_bneck(x)
        mu = self.encoder_mu(x)
        logvar = self.encoder_logvar(x)
        return mu , logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def decode(self, z):
        z = self.fc_z_bneck(z)
        z = self.fc_bneck_bneck(z)
        z = self.fc_z1(z)
        if self.img_size == 128:
            s = 8
        elif self.img_size == 64:
            s = 4
        else:
            raise RuntimeError(' the image size is not supported') 
        z = z.view(-1, 8 * self.nef , s, s )
        z = self.conv_t_block_1(z)
        z = self.conv_t_block_2(z)
        z = self.conv_t_block_3(z)
        z = self.conv_t_block_4(z)
        z = self.last_layer(z)

        return z 

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        out = self.decode(z)

        return out, mu, logvar