"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer
from models.networks.architecture import ResnetBlock as ResnetBlock
from models.networks.architecture import SPADEResnetBlock as SPADEResnetBlock
### sina
# ref:"https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html"
# Set random seed for reproducibility
import random
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
### sina

class SPADEGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
        parser.add_argument('--num_upsampling_layers',
                            choices=('few','normal', 'more', 'most'), default='normal',
                            help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")

        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ngf

        self.sw, self.sh = self.compute_latent_vector_size(opt)

        if self.opt.use_vae:
            # In case of VAE, we will sample from random z vector
            
            self.fc = nn.Linear(opt.z_dim, 16 * nf * self.sw * self.sh)
            ### sina
        elif self.opt.use_noise:
            self.fc = nn.Linear(opt.z_dim, 16 * nf * self.sw * self.sh)
            ### sina
        else:
            # Otherwise, we make the network deterministic by starting with
            # downsampled segmentation map instead of random z
            self.fc = nn.Conv2d(self.opt.semantic_nc, 16 * nf, 3, padding=1)

        self.head_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.G_middle_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)
        self.G_middle_1 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.up_0 = SPADEResnetBlock(16 * nf, 8 * nf, opt)
        self.up_1 = SPADEResnetBlock(8 * nf, 4 * nf, opt)
        self.up_2 = SPADEResnetBlock(4 * nf, 2 * nf, opt)
        self.up_3 = SPADEResnetBlock(2 * nf, 1 * nf, opt)

        final_nc = nf

        if opt.num_upsampling_layers == 'most':
            self.up_4 = SPADEResnetBlock(1 * nf, nf // 2, opt)
            final_nc = nf // 2

        self.conv_img = nn.Sequential(
            nn.Conv2d(final_nc, opt.output_nc, 3, padding=1),  # sina changing the number of the channels for the output conv layer from 3 to opt.output_nc
            nn.Tanh()
        )

        self.up = nn.Upsample(scale_factor=2)

    def compute_latent_vector_size(self, opt):
        if opt.num_upsampling_layers == 'few':
            num_up_layers = 4
        elif opt.num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif opt.num_upsampling_layers == 'more':
            num_up_layers = 6
        elif opt.num_upsampling_layers == 'most':
            num_up_layers = 7
        else:
            raise ValueError('opt.num_upsampling_layers [%s] not recognized' %
                             opt.num_upsampling_layers)

        sw = opt.crop_size // (2**num_up_layers)
        sh = round(sw / opt.aspect_ratio)

        return sw, sh

    def forward(self, input, z=None, input_dist=None):
        seg = input

        if self.opt.use_vae:
            # we sample z from unit normal and reshape the tensor
            if z is None:
                z = torch.randn(input.size(0), self.opt.z_dim,
                                dtype=torch.float32, device=input.get_device())
            x = self.fc(z)
            x = x.view(-1, 16 * self.opt.ngf, self.sh, self.sw)
            ### sina
        elif self.opt.use_noise:
            # print('yes got the noise')
            z = torch.randn(input.size(0), self.opt.z_dim,
                                dtype=torch.float32, device=input.get_device())
            x = self.fc(z)
            x = x.view(-1, 16 * self.opt.ngf, self.sh, self.sw)
            ### sina
        else:
            # we downsample segmap and run convolution
            x = F.interpolate(seg, size=(self.sh, self.sw))
            x = self.fc(x)

        x = self.head_0(x, seg, input_dist)

        if self.opt.num_upsampling_layers != 'few':
            
            x = self.up(x)
            x = self.G_middle_0(x, seg, input_dist)

        if self.opt.num_upsampling_layers == 'more' or \
           self.opt.num_upsampling_layers == 'most':
            x = self.up(x)

        x = self.G_middle_1(x, seg, input_dist)

        x = self.up(x)
        x = self.up_0(x, seg, input_dist)
        x = self.up(x)
        x = self.up_1(x, seg, input_dist)
        x = self.up(x)
        x = self.up_2(x, seg, input_dist)
        
        
        x = self.up(x)
        x = self.up_3(x, seg, input_dist)

        if self.opt.num_upsampling_layers == 'most':
            x = self.up(x)
            x = self.up_4(x, seg, input_dist)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        # x = nn.Tanh(x)

        return x


class SPADEEncGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
        parser.add_argument('--num_upsampling_layers',
                            choices=('few','normal', 'more', 'most'), default='normal',
                            help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")

        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ngf

        self.sw, self.sh = self.compute_latent_vector_size(opt)

        if self.opt.use_vae:
            # In case of VAE, we will sample from random z vector
            
            # self.fc = nn.Linear(opt.z_dim, 16 * nf * self.sw * self.sh)
        #FIXME: 
    
            self.fc = nn.Conv2d(self.opt.encoder_nc, 16 * nf, 3, padding=1) # encoder_nc should be the numbe of the channels for the encoder output 512?
            ### sina
        elif self.opt.use_noise:
            self.fc = nn.Linear(opt.z_dim, 16 * nf * self.sw * self.sh)
            ### sina
        else:
            # Otherwise, we make the network deterministic by starting with
            # downsampled segmentation map instead of random z
            self.fc = nn.Conv2d(self.opt.semantic_nc, 16 * nf, 3, padding=1)

        self.head_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.G_middle_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)
        self.G_middle_1 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.up_0 = SPADEResnetBlock(16 * nf, 8 * nf, opt)
        self.up_1 = SPADEResnetBlock(8 * nf, 4 * nf, opt)
        self.up_2 = SPADEResnetBlock(4 * nf, 2 * nf, opt)
        self.up_3 = SPADEResnetBlock(2 * nf, 1 * nf, opt)

        final_nc = nf

        if opt.num_upsampling_layers == 'most':
            self.up_4 = SPADEResnetBlock(1 * nf, nf // 2, opt)
            final_nc = nf // 2

        self.conv_img = nn.Sequential(
            nn.Conv2d(final_nc, opt.output_nc, 3, padding=1),  # sina changing the number of the channels for the output conv layer from 3 to opt.output_nc
            nn.Tanh()
        )

        self.up = nn.Upsample(scale_factor=2)

    def compute_latent_vector_size(self, opt):
        if opt.num_upsampling_layers == 'few':
            num_up_layers = 4
        elif opt.num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif opt.num_upsampling_layers == 'more':
            num_up_layers = 6
        elif opt.num_upsampling_layers == 'most':
            num_up_layers = 7
        else:
            raise ValueError('opt.num_upsampling_layers [%s] not recognized' %
                             opt.num_upsampling_layers)

        sw = opt.crop_size // (2**num_up_layers)
        sh = round(sw / opt.aspect_ratio)

        return sw, sh

    def forward(self, input, z=None, input_dist=None):
        seg = input

        if self.opt.use_vae:
            # we sample z from unit normal and reshape the tensor
            if z is None:
                z = torch.randn(input.size(0), self.opt.z_dim,
                                dtype=torch.float32, device=input.get_device())
            x = self.fc(z)
            x = x.view(-1, 16 * self.opt.ngf, self.sh, self.sw)
            ### sina
        elif self.opt.use_noise:
            # print('yes got the noise')
            z = torch.randn(input.size(0), self.opt.z_dim,
                                dtype=torch.float32, device=input.get_device())
            x = self.fc(z)
            x = x.view(-1, 16 * self.opt.ngf, self.sh, self.sw)
            ### sina
        else:
            # we downsample segmap and run convolution
            x = F.interpolate(seg, size=(self.sh, self.sw))
            x = self.fc(x)

        x = self.head_0(x, seg, input_dist)

        if self.opt.num_upsampling_layers != 'few':
            
            x = self.up(x)
            x = self.G_middle_0(x, seg, input_dist)

        if self.opt.num_upsampling_layers == 'more' or \
           self.opt.num_upsampling_layers == 'most':
            x = self.up(x)

        x = self.G_middle_1(x, seg, input_dist)

        x = self.up(x)
        x = self.up_0(x, seg, input_dist)
        x = self.up(x)
        x = self.up_1(x, seg, input_dist)
        x = self.up(x)
        x = self.up_2(x, seg, input_dist)
        
        
        x = self.up(x)
        x = self.up_3(x, seg, input_dist)

        if self.opt.num_upsampling_layers == 'most':
            x = self.up(x)
            x = self.up_4(x, seg, input_dist)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        # x = nn.Tanh(x)

        return x


class Pix2PixHDGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--resnet_n_downsample', type=int, default=4, help='number of downsampling layers in netG')
        parser.add_argument('--resnet_n_blocks', type=int, default=9, help='number of residual blocks in the global generator network')
        parser.add_argument('--resnet_kernel_size', type=int, default=3,
                            help='kernel size of the resnet block')
        parser.add_argument('--resnet_initial_kernel_size', type=int, default=7,
                            help='kernel size of the first convolution')
        parser.set_defaults(norm_G='instance')
        return parser

    def __init__(self, opt):
        super().__init__()
        input_nc = opt.label_nc + (1 if opt.contain_dontcare_label else 0) + (0 if opt.no_instance else 1)

        norm_layer = get_nonspade_norm_layer(opt, opt.norm_G)
        activation = nn.ReLU(False)

        model = []

        # initial conv
        model += [nn.ReflectionPad2d(opt.resnet_initial_kernel_size // 2),
                  norm_layer(nn.Conv2d(input_nc, opt.ngf,
                                       kernel_size=opt.resnet_initial_kernel_size,
                                       padding=0)),
                  activation]

        # downsample
        mult = 1
        for i in range(opt.resnet_n_downsample):
            model += [norm_layer(nn.Conv2d(opt.ngf * mult, opt.ngf * mult * 2,
                                           kernel_size=3, stride=2, padding=1)),
                      activation]
            mult *= 2

        # resnet blocks
        for i in range(opt.resnet_n_blocks):
            model += [ResnetBlock(opt.ngf * mult,
                                  norm_layer=norm_layer,
                                  activation=activation,
                                  kernel_size=opt.resnet_kernel_size)]

        # upsample
        for i in range(opt.resnet_n_downsample):
            nc_in = int(opt.ngf * mult)
            nc_out = int((opt.ngf * mult) / 2)
            model += [norm_layer(nn.ConvTranspose2d(nc_in, nc_out,
                                                    kernel_size=3, stride=2,
                                                    padding=1, output_padding=1)),
                      activation]
            mult = mult // 2

        # final output conv
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(nc_out, opt.output_nc, kernel_size=7, padding=0),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input, z=None):
        return self.model(input)







class StyleSPADEGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        
        parser.add_argument('--num_upsampling_layers',
                            choices=('few','normal', 'more', 'most'), default='few',
                            help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")
        
        # parser.add_argument('--resnet_n_downsample', type=int, default=5, help='number of downsampling layers in netG')
        # parser.add_argument('--resnet_n_blocks', type=int, default=2, help='number of residual blocks in the global generator network')
        # parser.add_argument('--resnet_kernel_size', type=int, default=3,
                            # help='kernel size of the resnet block')
        # parser.add_argument('--resnet_initial_kernel_size', type=int, default=7,
        #                     help='kernel size of the first convolution')
        parser.set_defaults(resnet_n_downsample=5)
        parser.set_defaults(resnet_n_blocks=2)
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
        # parser.set_defaults(norm_G='spectralspadeinstance3x3') dont use
        # parser.set_defaults(norm_G='instance')
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ngf
        norm_layer_style = get_nonspade_norm_layer(opt, 'spectralsync_batch')
        # norm_layer_style = get_nonspade_norm_layer(opt, 'spectralinstance') dont use
        # norm_layer_style = get_nonspade_norm_layer(opt, opt.norm_E)
        
        activation = nn.ReLU(False)
        model = []

        ##   style encoder 

         # initial conv
        model += [nn.ReflectionPad2d(opt.resnet_initial_kernel_size // 2),
                  norm_layer_style(nn.Conv2d(self.opt.output_nc, opt.ngf,
                                       kernel_size=opt.resnet_initial_kernel_size,
                                       padding=0)),
                  activation]

        # downsample
        mult = 1
        for i in range(opt.resnet_n_downsample):
            model += [norm_layer_style(nn.Conv2d(opt.ngf * mult, opt.ngf * mult * 2,
                                           kernel_size=3, stride=2, padding=1)),
                      activation]
            mult *= 2


        # resnet blocks
        for i in range(opt.resnet_n_blocks):
            model += [ResnetBlock(opt.ngf * mult,
                                  norm_layer=norm_layer_style,
                                  activation=activation,
                                  kernel_size=opt.resnet_kernel_size)]

        self.model = nn.Sequential(*model)

        # self.sw, self.sh = self.compute_latent_vector_size(opt)

        # if self.opt.use_vae:
        #     # In case of VAE, we will sample from random z vector
            
        #     # self.fc = nn.Linear(opt.z_dim, 16 * nf * self.sw * self.sh)
        # #FIXME: 
    
        #     self.fc = nn.Conv2d(self.opt.encoder_nc, 16 * nf, 3, padding=1) # encoder_nc should be the numbe of the channels for the encoder output 512?
        #     ### sina
        # elif self.opt.use_noise:
        #     self.fc = nn.Linear(opt.z_dim, 16 * nf * self.sw * self.sh)
        #     ### sina
        # else:
        #     # Otherwise, we make the network deterministic by starting with
        #     # downsampled segmentation map instead of random z


        if self.opt.crop_size == 256:
            in_fea = 2 * 16
            self.opt.num_upsampling_layers = 'most'
        if self.opt.crop_size == 128:
            in_fea = 1 * 16
        
        self.fc_img = nn.Linear(in_fea * nf * 8 * 8, in_fea * nf)
        self.fc_img2 = nn.Linear(in_fea * nf, in_fea * nf * 8 * 8)
        self.fc = nn.Conv2d(self.opt.semantic_nc, in_fea * nf, 3, padding=1)

        self.head_0 = SPADEResnetBlock(in_fea * nf, in_fea * nf, opt)

        self.G_middle_0 = SPADEResnetBlock(in_fea * nf, in_fea * nf, opt)
        self.G_middle_1 = SPADEResnetBlock(in_fea * nf, in_fea * nf, opt)

        self.up_0 = SPADEResnetBlock(in_fea * nf, 8 * nf, opt)
        self.up_1 = SPADEResnetBlock(8 * nf, 4 * nf, opt)
        self.up_2 = SPADEResnetBlock(4 * nf, 2 * nf, opt)
        self.up_3 = SPADEResnetBlock(2 * nf, 1 * nf, opt)

        final_nc = nf

        if self.opt.num_upsampling_layers == 'most':
            self.up_4 = SPADEResnetBlock(1 * nf, nf // 2, opt)
            final_nc = nf // 2

        self.conv_img = nn.Sequential(
            nn.Conv2d(final_nc, opt.output_nc, 3, padding=1),  # sina changing the number of the channels for the output conv layer from 3 to opt.output_nc
            nn.Tanh()
        )

        self.up = nn.Upsample(scale_factor=2)

    # def compute_latent_vector_size(self, opt):
    #     if opt.num_upsampling_layers == 'few':
    #         num_up_layers = 4
    #     elif opt.num_upsampling_layers == 'normal':
    #         num_up_layers = 5
    #     elif opt.num_upsampling_layers == 'more':
    #         num_up_layers = 6
    #     elif opt.num_upsampling_layers == 'most':
    #         num_up_layers = 7
    #     else:
    #         raise ValueError('opt.num_upsampling_layers [%s] not recognized' %
    #                          opt.num_upsampling_layers)

    #     sw = opt.crop_size // (2**num_up_layers)
    #     sh = round(sw / opt.aspect_ratio)

    #     return sw, sh

    def forward(self, input, image, input_dist=None):
        seg = input
        image = image

        # if self.opt.use_vae:
        #     # we sample z from unit normal and reshape the tensor
        #     if z is None:
        #         z = torch.randn(input.size(0), self.opt.z_dim,
        #                         dtype=torch.float32, device=input.get_device())
        #     x = self.fc(z)
        #     x = x.view(-1, 16 * self.opt.ngf, self.sh, self.sw)
        #     ### sina
        # elif self.opt.use_noise:
        #     # print('yes got the noise')
        #     z = torch.randn(input.size(0), self.opt.z_dim,
        #                         dtype=torch.float32, device=input.get_device())
        #     x = self.fc(z)
        #     x = x.view(-1, 16 * self.opt.ngf, self.sh, self.sw)
        #     ### sina
        # else:
        #     # we downsample segmap and run convolution
        #     x = F.interpolate(seg, size=(self.sh, self.sw))
        #     x = self.fc(x)
        x = self.model(image)
        # seg = F.interpolate(seg, size=(x.shape[-1], x.shape[-2]))
        # seg = self.fc(seg)
        
        x = x.view(x.size(0), -1)
        

        x = self.fc_img(x)
        x = self.fc_img2(x)

        if self.opt.crop_size == 256:
            in_fea = 2 * 16
        if self.opt.crop_size == 128:
            in_fea = 1 * 16
   
        x = x.view(-1, in_fea * self.opt.ngf, 8, 8)

        x = self.head_0(x, seg, input_dist)

        # if self.opt.num_upsampling_layers != 'few':
            
        #     x = self.up(x)
        x = self.G_middle_0(x, seg, input_dist)

        # if self.opt.num_upsampling_layers == 'more' or \
        #    self.opt.num_upsampling_layers == 'most':
        #     x = self.up(x)

        # x = self.G_middle_1(x, seg, input_dist)

        x = self.up(x)
        x = self.up_0(x, seg, input_dist)
        x = self.up(x)
        x = self.up_1(x, seg, input_dist)
        x = self.up(x)
        x = self.up_2(x, seg, input_dist)
        x = self.up(x)
        x = self.up_3(x, seg, input_dist)

        if self.opt.num_upsampling_layers == 'most':
            x = self.up(x)
            x = self.up_4(x, seg, input_dist)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        # x = nn.Tanh(x)

        return x