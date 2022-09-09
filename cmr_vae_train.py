# %%
import numpy as np

import matplotlib.pyplot as plt
from torch.functional import norm
import torch.nn.functional as F
import data
import argparse
import torch
import os
import sys
from util.iter_counter import IterationCounter
from models.networks.vae import VAE
from models.networks import vae_loss
import torch.optim as optim 
import time
from util.plot_loss import Plot_loss
from torchvision.utils import make_grid , save_image
from util.util import one_hot, save_network, show_gen, print_current_errors, set_seed, combined_loss, combined_loss_beta_VAE, load_network_vae 
import random
from torch.utils.tensorboard import SummaryWriter
# %%

plt.rcParams["figure.figsize"] = (16, 10)
parser = argparse.ArgumentParser(description='Add some arguments for the model')
### ================================================================================  options starts
parser.add_argument('--gpu_ids', type=str, default='1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

parser.add_argument('--label_dir', type=str, required=False, default = "/data/sina/dataset/cmrVAE/mms1/training_crop_BB/vendors/Vendor_A/Mask/",
                    help='path to the directory that contains label images')
parser.add_argument('--label_dir_B', type=str, required=False, default = "/data/sina/dataset/cmrVAE/mms1/training_crop_BB/vendors/Vendor_B/Mask/",
                    help='path to the directory that contains label images')
parser.add_argument('--image_dir', type=str, required=False, default ="/data/sina/dataset/cmrVAE/mms1/training_crop_BB/vendors/Vendor_A/Image/" ,
                    help='path to the directory that contains photo images')
parser.add_argument('--image_dir_B', type=str, required=False, default ="/data/sina/dataset/cmrVAE/mms1/training_crop_BB/vendors/Vendor_B/Image/" ,
                    help='path to the directory that contains photo images')
parser.add_argument('--image_dir_mms2', type=str, required=False, default ="/data/sina/dataset/cmrVAE/mms1/training_crop_BB/vendors/Vendor_B/Image/" ,
                    help='path to the directory that contains photo images')
parser.add_argument('--acdc_dir', type=str, required=False, default = "/data/sina/dataset/ACDC/pathology_crop_noBA_NR_C128/",
                            help='path to the directory that contains label images')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--crop_size', type=int, default=128, help='Crop to the width of crop_size (after initially scaling the images to load_size.)')
parser.add_argument('--target_res', type=int, default=1.25, help='Crop to the width of crop_size (after initially scaling the images to load_size.)')

parser.add_argument('--label_nc', type=int, default=4, help='# of input label classes.')
parser.add_argument('--output_nc', type=int, default=4, help='# of output image channels')
parser.add_argument('--input_nc', type=int, default=1, help='# of input image channels')
parser.add_argument('--which_epoch', type=int, default=100, help='# of output image channels')
parser.add_argument('--dataset_mode', type=str, default='mms2BB')
parser.add_argument('--vendor', type=str, default='All_SA', help='selects a vendor for training [Philips_LA, Philips_SA, Siemens_LA, Siemens_SA, All_SA]')
parser.add_argument('--phase', type=str, default='train', help='train validation test')
parser.add_argument('--serial_batches', action='store_true', default=False, help='')
parser.add_argument('--isTrain', action='store_true', default=True, help='')

parser.add_argument('--nThreads', default=0, type=int, help='# threads for loading data')
parser.add_argument('--print_freq', default=100, type=int, help='#print frequency')
parser.add_argument('--niter', type=int, default=300, help='# of iter at starting learning rate. This is NOT the total #epochs. Totla #epochs is niter + niter_decay')
parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
parser.add_argument('--web_dir', type=str, help='models are saved here')
parser.add_argument('--name', type=str, default='labelmanipulation', help='name of the experiment. It decides where to store samples and models')
parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')

parser.add_argument('--rec_loss', type=str, default='Dice', help='recontrcution loss for VAE: choose between BCE  MSE L1 L1F Dice')
parser.add_argument('--seed', type=int, default=1220,
                        help='Random seed. Can be `None` for stochastic behavior.')
parser.add_argument('--zdim', default=32, type=int, help='# size of latent vector')
parser.add_argument('--corrected_labels', action='store_true', help='continue training: load the latest model')
parser.add_argument('--VAE_altered_anatomy', action='store_true', help='continue training: load the latest model')
parser.add_argument('--selected_labels', action='store_true', help='continue training: load the latest model')
parser.add_argument('--no_instance', action='store_true', help='continue training: load the latest model')
parser.add_argument('--add_dist', action='store_true', help='continue training: load the latest model')
parser.add_argument('--what_data', type=str,default='all', help='what data to load: acdc, mms')


### this is for parsing arguments on jupyter notebook and resolving its issue
import sys
sys.argv=['']
del sys
# %%


### parsing options
opt = parser.parse_args()
opt.batchSize = 20
opt.output_nc = 1
opt.input_nc = 4
opt.label_nc = 4
opt.rec_loss = 'CE'
opt.add_dist = False
# opt.rec_loss = 'CEDice'
opt.crop_size = 128
# opt.target_res = 1.5

### experiments using mms-1 data
opt.label_dir = "/data/sina/dataset/cmrVAE/mms1/training_crop_noBA_NR/vendors/Vendor_A/Mask/"
opt.image_dir = "/data/sina/dataset/cmrVAE/mms1/training_crop_noBA_NR/vendors/Vendor_A/Image/"
opt.label_dir_B = "/data/sina/dataset/cmrVAE/mms1/training_crop_noBA_NR/vendors/Vendor_B/Mask/"
opt.image_dir_B = "/data/sina/dataset/cmrVAE/mms1/training_crop_noBA_NR/vendors/Vendor_B/Image/"

opt.image_dir_mms2 = "/data/sina/dataset/cmrVAE/mms2_sorted/PerDisease_crop_noBA_R_128/"
opt.acdc_dir = "/data/sina/dataset/cmrVAE/acdc/"




### for paper dont use the mms2 data, the heart for mms2 case positined differently
opt.name = '220909_mms1_acdc_z16_128_beta15'
writer = SummaryWriter("runs/" + opt.name , comment=opt.name)
opt.zdim = 16
lamda_kld = 15

opt.isTrain = True
opt.continue_train = False

### setting the GPU ID
torch.cuda.set_device('cuda:' + str(opt.gpu_ids[0]))
device = torch.device('cuda:' + str(opt.gpu_ids[0]))

### creating the dataloader

opt.dataset_mode = 'mms1acdcBB'
opt.what_data = 'all'

dataloader = data.create_dataloader(opt)
### create tool for counting iterations
iter_counter = IterationCounter(opt, len(dataloader))

### create a webpage that summarizes the all results
web_dir = os.path.join(opt.checkpoints_dir, opt.name,
                           '%s' % (opt.phase) + '/')
os.makedirs(os.path.join(web_dir, 'images'), exist_ok=True)
os.makedirs(web_dir, exist_ok=True)    
opt.web_dir = web_dir


### ================================================================================  training parameters starts
max_epoch = 100000 # maximum number of epochs, break
max_image = 100000 # maximum number of images per epoch, break
input_labels_list = []
# init_lr = 0.0000025
init_lr = 0.00002
# init_lr = 0.0002
# init_lr_sch = [0.002, 0.0002, 0.00002, 0.000002, 0.000001, 0.0000001]
# init_lr = 0.00006
# lamda_kld = 0.5 
train_losses = []


### ================================================================================  create VAE model and optimizer starts

vae = VAE(opt)

opt.which_epoch = 50
if opt.continue_train:
    vae = load_network_vae(vae, opt.which_epoch, opt)


vae.cuda()

optimizer = optim.Adam(vae.parameters(), lr=init_lr)
# adjusting learning rate source: https://pytorch.org/docs/stable/optim.html
# scheduler =  torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
lambda1 = lambda epoch: 0.96 ** epoch
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[70,140], gamma=0.1)

opt.seed = 1220
set_seed(opt.seed)
model_to_tensorboard = False # something is wrong here! there is a warninig when this line is activate - not needed for experiments though 
# just turn it on once when there is a change in the network architecture to visualize that
if model_to_tensorboard:
    data_i = next(iter(dataloader))
    # send image to cuda
    input_image = data_i['label'].cuda().type(torch.cuda.FloatTensor)
    # create one-hot vector for label map 
    size = input_image.size()
    oneHot_size = (size[0], opt.label_nc, size[2], size[3])
    input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
    input_label = input_label.scatter_(1, input_image.data.long().cuda(), 1.0)
    writer.add_graph(vae, input_label)
    writer.flush()

### ================================================================================ training procedure starts
# %%
for epoch in iter_counter.training_epochs():
    if epoch > max_epoch:
            break
    if opt.continue_train:
        epoch = epoch + opt.which_epoch
    iter_counter.record_epoch_start(epoch)
    # train mode
    vae.train()
    
    # statistics
    running_loss = 0.0
    
    # iteration over training data
    for i, data_i in enumerate(dataloader, start=iter_counter.epoch_iter):
        if i>max_image:
            break

        iter_counter.record_one_iteration()
        input_labels_list.append(data_i['label'])
        
        # send image to cuda
        input_image = data_i['label'].cuda().type(torch.cuda.FloatTensor)
        ############################
        # create one-hot vector for label map 
        size = input_image.size()
        oneHot_size = (size[0], opt.label_nc, size[2], size[3])
        input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
        input_label = input_label.scatter_(1, input_image.data.long().cuda(), 1.0)


        optimizer.zero_grad()

        # writer.add_graph(vae, input_label) gives me some warninigs and slows the training substantially
        # writer.close()
        
        # forward + backward + optimize
        reconstructed, mu, logvar = vae(input_label)

        loss, losses_dic = combined_loss_beta_VAE(reconstructed, input_image, mu, logvar, type = opt.rec_loss, lamda_kld = lamda_kld, n_train_steps = iter_counter.total_steps_so_far)
        
        loss.backward()
        optimizer.step()
        
        # statistics
        running_loss += loss.item()
        # Visualizations
        if iter_counter.needs_printing():
            losses = losses_dic
            print_current_errors(epoch, iter_counter.epoch_iter,
                                                losses, iter_counter.time_per_iter, web_dir)
            opt.log_file = web_dir +  '/loss_log.txt'
            Plot_loss.plot_loss(opt)

    # adjusting learning rate
    if  epoch >= 150:
        scheduler.step()
    print('the new learning rate is {}'.format(round(scheduler.get_last_lr()[0], 9)))
    running_loss /= (len(dataloader) // opt.batchSize)
    
    train_losses.append(running_loss)
    # output
    print('Epoch {} -- loss: {:.4f}'.format(epoch + 1, running_loss))
    
    # visualize reconstruction and synthesis
    if(epoch==1) or (epoch%5==0):
        print("Saving training examples . . . ")
        reverted_recon = torch.argmax(reconstructed, dim=1).cuda().type(torch.cuda.FloatTensor).unsqueeze(dim=1)
        rec_img_grid = make_grid(reverted_recon, nrow=opt.batchSize//2, padding=12, pad_value=-1, normalize=True)
        real_img_grid = make_grid(input_image, nrow=opt.batchSize//2, padding=12, pad_value=-1, normalize=True)
        real_name= os.path.join(web_dir, 'images/') + str(epoch) + '_real_image.png'
        rec_name = os.path.join(web_dir, 'images/') +  str(epoch) + '_rec_image.png'
        save_image(rec_img_grid[0].detach().cpu(), rec_name)
        save_image(real_img_grid[0].detach().cpu(), real_name)
    
    if epoch%2==0:
        ## saving images in tensorboard:
        reverted_recon = torch.argmax(reconstructed, dim=1).cuda().type(torch.cuda.FloatTensor).unsqueeze(dim=1)
        rec_img_grid = make_grid(reverted_recon, padding=12, pad_value=-1, normalize=True)
        real_img_grid = make_grid(input_image,  padding=12, pad_value=-1, normalize=True)
        writer.add_image('reconstructed', rec_img_grid, epoch)
        writer.add_image('input', real_img_grid, epoch)
        writer.add_scalar("Loss/train.all", loss, epoch)
        writer.add_scalar("Loss/train.KLD", losses_dic['KLD'], epoch)
        writer.add_scalar("Loss/train.REC", losses_dic['Rec'], epoch)
        
        writer.add_histogram("model/encoder_mu.weight", vae.encoder_mu.weight, epoch)
        writer.add_histogram("model/encoder_logvar.weight", vae.encoder_logvar.weight, epoch)
        writer.add_histogram("latent/mu", mu, epoch)
        writer.add_histogram("latent/logvar", logvar, epoch)
        

        writer.flush()
        
        
        
    
    if  epoch%50==0:
        mode_name = web_dir + 'VAE_net' + str(epoch) + '.pth'
        torch.save(vae.state_dict(), mode_name)
writer.close()


    
    
        
        
        
# %%
