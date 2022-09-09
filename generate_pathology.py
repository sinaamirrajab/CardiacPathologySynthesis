# %%
import torch
import os
import numpy as np
import pandas as pd
from scipy import interpolate
import scipy.stats as stats
import torchvision.utils
import ipywidgets as wg
import data
from util.iter_counter import IterationCounter
from models.networks.vae import VAE
from util.util import load_network_vae
from util.util import set_seed, show_gen
import matplotlib.pyplot as plt
from torchvision.utils import make_grid , save_image
from models.pix2pix_model import Pix2PixModel
from util.visualizer import Visualizer
from util import html
import sys
sys.argv=['']
del sys
# %%
from options.test_options import TestOptions
opt = TestOptions().parse()

# %%



### parsing options

opt.batchSize = 1
opt.selected_labels = True

opt.label_dir = "/data/sina/dataset/cmrVAE/mms1/training_crop_noBA_NR/vendors/Vendor_A/Mask/"
opt.image_dir = "/data/sina/dataset/cmrVAE/mms1/training_crop_noBA_NR/vendors/Vendor_A/Image/"
opt.label_dir_B = "/data/sina/dataset/cmrVAE/mms1/training_crop_noBA_NR/vendors/Vendor_B/Mask/"
opt.image_dir_B = "/data/sina/dataset/cmrVAE/mms1/training_crop_noBA_NR/vendors/Vendor_B/Image/"

# opt.acdc_dir = "/data/sina/dataset/ACDC/pathology_crop_noBA_NR_C128_correct_pixdim/"
opt.acdc_dir = "/data/sina/dataset/cmrVAE/acdc/"
# opt.name = '220127_mms1_acdc_z32_128_beta10'
opt.name = '220119_mms1_acdc_z16_128_beta15'
opt.zdim = 16
opt.serial_batches = True

### setting the GPU ID
torch.cuda.set_device('cuda:' + str(opt.gpu_ids[0]))
device = torch.device('cuda:' + str(opt.gpu_ids[0]))

### creating the dataloader
# opt.dataset_mode = 'mms1BB'
opt.serial_batches = True
opt.isTrain = False
opt.what_data = 'acdc'
opt.dataset_mode = 'mms1acdcBB'
dataloader = data.create_dataloader(opt)
### create tool for counting iterations
# iter_counter = IterationCounter(opt, len(dataloader))

### create a webpage that summarizes the all results
web_dir = os.path.join(opt.checkpoints_dir, opt.name,
                           '%s' % (opt.phase) + '/')
os.makedirs(os.path.join(web_dir, 'images'), exist_ok=True)
os.makedirs(web_dir, exist_ok=True)
os.makedirs(os.path.join(opt.result_dir,opt.name ), exist_ok=True)

opt.web_dir = web_dir
print(device)

# %%
vae = VAE(opt).to(device)
opt.isTrain = False
opt.continue_train = True
opt.which_epoch = 200
if opt.continue_train or not opt.isTrain :
    vae = load_network_vae(vae, opt.which_epoch, opt, device)

vae.to(device)
opt.seed = 1220
set_seed(opt.seed)
# %%

opt.results_dir = './results/Synthetic_pathology'
opt.name = '211114_GAN_cmr_mms1_acdc_noBA_NR_128'
opt.which_epoch = 'latest' 
model = Pix2PixModel(opt)
model.eval()

web_dir = os.path.join(opt.results_dir, opt.name,
                       '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir,
                    'Experiment = %s, Phase = %s, Epoch = %s' %
                    (opt.name, opt.phase, opt.which_epoch))
# %%

# to synthesize 100 samples for checking the quality
generated_images = []
real_images = []
real_labels = []
how_many = 20 
data_loader = dataloader


for i, data_i in enumerate(data_loader):
    if i * opt.batchSize >= how_many:
        break
#     data_i['image'] = torch.transpose(data_i['image'],2,3)
#     data_i['label'] = torch.transpose(data_i['label'],2,3)
#     data_i['image'] = torch.rot90(data_i['image'], 1,(2,3))
#     data_i['label'] = torch.rot90(data_i['label'], 1,(2,3))
    data_i['label']= torch.round(data_i['label'])

    real_images+=data_i['image']
    # real_labels+=torch.ceil(data_i['label'])
    real_labels+=torch.round(data_i['label'])
    # data_i['label'] = torch.round(data_i['label'])
    # print(data_i['label'].dtype)
    generated = model(data_i, mode='inference')
    generated_images+=generated
# %%
show_gen(generated_images[:], title = 'Synthesized Image')
show_gen(real_labels[:], title= 'Input Label')
show_gen(real_images[:],title = 'Real Image' )
# %%
embedings = {'subject':[],
             'pathology':[], 'image': [], 'label': []}
for i in range(opt.zdim):
    name = 'z' + str(i)
    embedings.update({name: []})   
    
max_image = 50000
for i, data_i in enumerate(dataloader):
    if i> max_image:
        break
#     segpair_slice = data_i['segpair_slice'][0]
    
    subject = data_i['path'][0].split('/')[-1]
#     print(subject)


    pathology = str(data_i['path'][0].split('/')[-3])
    image = data_i['image']
    label = data_i['label']
    embedings['subject'].append(subject)
    embedings['pathology'].append(pathology)
    embedings['image'].append(image)
    embedings['label'].append(label)
    # create one-hot vector for label map 
    size = image.size()
    oneHot_size = (size[0], opt.label_nc, size[2], size[3])
    input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
    
    input_label = input_label.scatter_(1, label.data.long().cuda(), 1.0)

    reconstructed, mu, logvar = vae(input_label)
    z = vae.reparameterize(mu, logvar)
    z_np  = z.cpu().detach().numpy().squeeze()
#     z_norm = np.abs((z_np - z_np.min())/(z_np.max()-z_np.min()))
    
    for i in range(opt.zdim):
        name = 'z' + str(i)
        embedings[name].append(pd.to_numeric(z_np[i], downcast='float'))


        

# %%
df = pd.DataFrame(embedings)
df.head(15)
# %%
os.makedirs('./pickle',exist_ok=True)
df.to_pickle('./pickle/acdc_pickle.pkl')
# df = pd.read_pickle('./pickle/acdc_pickle.pkl')
# df.head(15)
# %%
def find_ED_ES_of_subjects(sujects_embedings):
    all_subjects = list(sujects_embedings['subject'].drop_duplicates(keep='first'))
    ED_sujects = []
    ES_sujects = []
    for idx, subject in enumerate(all_subjects):
        if idx%2==0:
            ED_sujects.append(subject)
        else:
            ES_sujects.append(subject)
    assert len(ED_sujects)==len(ES_sujects)
    # check the subjects are the same and correctly chosen as ED and ES
    for i in range(len(ED_sujects)):
        assert ED_sujects[i].split('_')[0]== ED_sujects[i].split('_')[0]
    return ED_sujects, ES_sujects

def condition_df(list_sujects, sujects_embedings):
    phase_df = []
    for idx , sub in enumerate(sujects_embedings['subject']):
        if sub in list_sujects:
            phase_df.append(True)
    #         ED_df.append(sujects_embedings.iloc[idx][0])
        else:
            phase_df.append(False)
    phase_df = pd.DataFrame({'subject': phase_df})
    condition = phase_df['subject'] == True
    return condition

def interpolate_slices(sub_slice_np):
    n_slice = sub_slice_np.shape[0]
    x = np.arange(0, n_slice)
    fc = interpolate.interp1d(x, sub_slice_np, kind='cubic', axis=0)
    xnew = np.arange(0, n_slice-1, (n_slice-1)/31) # for upsampling to 16 slices
    xnew = np.append(xnew,n_slice-1)
    ynew = fc(xnew)
    return ynew.astype('float32')

    
def interpolate_slices_all_subjects(df):
    all_sub_np = []
    sub_old = ''
    for sub in df['subject']:
        sub_new = sub
        sub_name = sub.split('_')[0]
        
        if not sub_old == sub_new:

            sub_slice = df[df['subject']==sub]
            sub_slice_np = sub_slice[['z'+str(i) for i in range(opt.zdim)]].to_numpy()

            sub_slice_np_interp = interpolate_slices(sub_slice_np)

            if sub_slice_np_interp.shape [0] ==32:
                all_sub_np.append(sub_slice_np_interp)
                
            else:
                print('subjcet {} is excluded because has more slices than 32 after interpolation'.format(sub))
            sub_old = sub_new

    return np.array(all_sub_np)

def interpolate_slices_and_return_mean_std_min_max(df_sub):
    subject_np = interpolate_slices_all_subjects(df_sub)
    mean = np.mean(subject_np, axis =0)
    std = np.std(subject_np, axis =0)
    min = np.min(subject_np, axis =0)
    max = np.max(subject_np, axis =0)
    return mean, std, min, max


def truncated_normal(mu, sigma, lower, upper):
    #lower, upper, mu, and sigma are four parameters
    #     lower, upper = -2*sigma, 2*sigma
    #instantiate an object X using the above four parameters,
    X = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    samples = X.rvs(sigma.shape)
    return samples
def get_images(df, sub):
    keys = ['subject', 'pathology', 'image', 'label']
    sub_images = df[df['subject']==sub][keys]
#     sub_images_np = sub_images['image']
    return sub_images
    

def interpolate_subject(df, sub, zdim):
    sub_slice = df[df['subject']==sub]
    sub_slice_np = sub_slice[['z'+str(i) for i in range(zdim)]].to_numpy()
    sub_slice_np_interp = interpolate_slices(sub_slice_np)
    return sub_slice_np_interp


def interpolate_NOR_to_Pathology(NOR_sub_interp, pathology_psuedo, how_many=10):
    n_slice = 2
#     how_many = 20
    x = np.arange(0, n_slice)
    concat = np.concatenate((NOR_sub_interp[np.newaxis,:], pathology_psuedo[np.newaxis,:]), axis=0 )
#     print(concat.shape)
    fc = interpolate.interp1d(x, concat, kind='linear', axis=0)
    xnew = np.arange(0, n_slice-1, (n_slice-1)/(how_many-1)) # for upsampling to 16 slices
    xnew = np.append(xnew,n_slice-1)
    ynew = fc(xnew)
    return ynew.astype('float32')

def concatinate_interpolate(sub_image):
    one_subject = []
    for idx in range(len(sub_image['image'].index)):
        one_image = sub_image['image'].iloc[idx].numpy().squeeze()
        one_subject.append(one_image)

    one_subject_np = np.array(one_subject)
    print('original size: ', one_subject_np.shape)
    one_subject_np_interp = interpolate_slices(one_subject_np)
    print('interpolated size: ',one_subject_np_interp.shape)
    return one_subject_np_interp
def make_int_slider(z,i):
    z_slider = wg.IntSlider(
        value=0,
        min=0,
        max=z,
        step=1,
        description='z'+str(i),
        disabled=False,
        continuous_update=False,
        readout_format='d'
    )
    return z_slider
def to_img(x):
    x = x.clamp(0, 1)
    return x
def show_image(img):
    img = to_img(img)
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
def show_image_from_z_synthesize(latent_z_np, input_image):
    latent_z_t = torch.from_numpy(latent_z_np).cuda()
    label = vae.decode(latent_z_t)
    label_recon = label.cpu()
    label_recon = torch.argmax(label_recon, dim=1).cuda().type(torch.cuda.FloatTensor)
    data_i = {}
    data_i['label'] = label_recon.cuda().type(torch.cuda.FloatTensor).unsqueeze(dim=0)
    data_i['image'] = torch.from_numpy(input_image).cuda().type(torch.cuda.FloatTensor).unsqueeze(dim=0).unsqueeze(dim=0)
    data_i['instance'] = torch.tensor([0])
    data_i['dist'] = torch.tensor([0])
    generated = model(data_i, mode='inference')
    fig, ax = plt.subplots( figsize=(6, 6))
    show_image(torchvision.utils.make_grid(generated[:], 10, 5, normalize=True))
    plt.axis('off')
    plt.show()
    fig, ax = plt.subplots( figsize=(6, 6))
    show_image(torchvision.utils.make_grid(label_recon[:], 10, 5, normalize=True))
    plt.axis('off')
    plt.show()

def show_NOR_to_Path_synthesize(NOR_10_DCM, input_image, interp_idx_total, slice_total):
    z0 = make_int_slider(interp_idx_total, 0)
    z1 = make_int_slider(slice_total, 1)

    ui = wg.VBox([z0, z1] )
    def f(z0, z1):

        show_image_from_z_synthesize(NOR_10_DCM[z0 ,z1,:], input_image[z1,:])
    out = wg.interactive_output(f, {'z0': z0, 'z1': z1})

    display(ui, out)

def correlate_random_sub(NOR_interp_np, psuedo_random_sub, all_pathological_df):
    
    corr_pathology = all_pathological_df.corr(method='kendall').to_numpy() # this is important to correlate across latent codes for all subjects
    chol_pathology = np.linalg.cholesky(corr_pathology)
    correlated_data = np.matmul(chol_pathology, psuedo_random_sub.T)
    corr_slice = pd.DataFrame(NOR_interp_np.T).corr(method='kendall').to_numpy() # this is important to correlate across slices for one subject
    chol_slice = np.linalg.cholesky(corr_slice)
    correlated_data = np.matmul(chol_slice, correlated_data.T)
    return correlated_data

# %%
keys = ['subject', 'pathology'] + ['z'+str(i) for i in range(opt.zdim)]
sujects_embedings = df[keys]
sujects_embedings
# %%
ED_sujects, ES_sujects = find_ED_ES_of_subjects(sujects_embedings)
condition_ED = condition_df(ED_sujects, sujects_embedings)  
condition_ES = condition_df(ES_sujects, sujects_embedings)  
# %%
NOR_ED = sujects_embedings[(sujects_embedings['pathology'] == 'NOR') &  condition_ED]
NOR_ES = sujects_embedings[(sujects_embedings['pathology'] == 'NOR') &  condition_ES]

DCM_ED = sujects_embedings[(sujects_embedings['pathology'] == 'DCM') &  condition_ED]
DCM_ES = sujects_embedings[(sujects_embedings['pathology'] == 'DCM') &  condition_ES]

HCM_ED = sujects_embedings[(sujects_embedings['pathology'] == 'HCM') &  condition_ED]
HCM_ES = sujects_embedings[(sujects_embedings['pathology'] == 'HCM') &  condition_ES]

RV_ED = sujects_embedings[(sujects_embedings['pathology'] == 'RV') &  condition_ED]
RV_ES = sujects_embedings[(sujects_embedings['pathology'] == 'RV') &  condition_ES]

NOR_ED.head(5)
# %%

NOR_ED_mean, NOR_ED_std, NOR_ED_min, NOR_ED_max = interpolate_slices_and_return_mean_std_min_max(NOR_ED)
NOR_ES_mean, NOR_ES_std, NOR_ES_min, NOR_ES_max = interpolate_slices_and_return_mean_std_min_max(NOR_ES)

HCM_ED_mean, HCM_ED_std, HCM_ED_min, HCM_ED_max = interpolate_slices_and_return_mean_std_min_max(HCM_ED)
HCM_ES_mean, HCM_ES_std, HCM_ES_min, HCM_ES_max = interpolate_slices_and_return_mean_std_min_max(HCM_ES)

DCM_ED_mean, DCM_ED_std, DCM_ED_min, DCM_ED_max = interpolate_slices_and_return_mean_std_min_max(DCM_ED)
DCM_ES_mean, DCM_ES_std, DCM_ES_min, DCM_ES_max = interpolate_slices_and_return_mean_std_min_max(DCM_ES)

RV_ED_mean, RV_ED_std, RV_ED_min, RV_ED_max = interpolate_slices_and_return_mean_std_min_max(RV_ED)
RV_ES_mean, RV_ES_std, RV_ES_min, RV_ES_max = interpolate_slices_and_return_mean_std_min_max(RV_ES)
# %%
DCM_ED_psuedo_tn = truncated_normal(DCM_ED_mean, DCM_ED_std, DCM_ED_min, DCM_ED_max)
DCM_ES_psuedo_tn = truncated_normal(DCM_ES_mean, DCM_ES_std, DCM_ES_min, DCM_ES_max)

HCM_ED_psuedo_tn = truncated_normal(HCM_ED_mean, HCM_ED_std, HCM_ED_min, HCM_ED_max)
HCM_ES_psuedo_tn = truncated_normal(HCM_ES_mean, HCM_ES_std, HCM_ES_min, HCM_ES_max)

RV_ED_psuedo_tn = truncated_normal(RV_ED_mean, RV_ED_std, RV_ED_min, RV_ED_max)
RV_ES_psuedo_tn = truncated_normal(RV_ES_mean, RV_ES_std, RV_ES_min, RV_ES_max)
DCM_ED_psuedo_tn.shape
# %%
NOR_sub_ED = 'patient061_frame01.nii.gz'
NOR_sub_ES = 'patient061_frame10.nii.gz'
# NOR_sub_ED = 'patient064_frame01.nii.gz'
# NOR_sub_ES = 'patient064_frame12.nii.gz'
NOR_ED_sub_interp = interpolate_subject(NOR_ED, NOR_sub_ED, opt.zdim)
NOR_ES_sub_interp = interpolate_subject(NOR_ES, NOR_sub_ES, opt.zdim)
NOR_sub_image_ED = get_images(df, NOR_sub_ED)
NOR_sub_image_ES = get_images(df, NOR_sub_ES)
NOR_ED_sub_interp.shape
# %%
DCM_ED_psuedo_tn_correlated = correlate_random_sub(NOR_ED_sub_interp, DCM_ED_psuedo_tn, DCM_ED)
DCM_ES_psuedo_tn_correlated = correlate_random_sub(NOR_ES_sub_interp, DCM_ES_psuedo_tn, DCM_ES)

HCM_ED_psuedo_tn_correlated = correlate_random_sub(NOR_ED_sub_interp, HCM_ED_psuedo_tn, HCM_ED)
HCM_ES_psuedo_tn_correlated = correlate_random_sub(NOR_ES_sub_interp, HCM_ES_psuedo_tn, HCM_ES)

RV_ED_psuedo_tn_correlated = correlate_random_sub(NOR_ED_sub_interp, RV_ED_psuedo_tn, RV_ED)
RV_ES_psuedo_tn_correlated = correlate_random_sub(NOR_ES_sub_interp, RV_ES_psuedo_tn, RV_ES)
NOR_20_HCM_ED_not_corr = interpolate_NOR_to_Pathology(NOR_ED_sub_interp, HCM_ED_psuedo_tn, 20)
NOR_20_HCM_ES_not_corr = interpolate_NOR_to_Pathology(NOR_ES_sub_interp, HCM_ES_psuedo_tn, 20)

DCM_20_HCM_ED_not_corr = interpolate_NOR_to_Pathology(NOR_ED_sub_interp, DCM_ED_psuedo_tn,20)
DCM_20_HCM_ES_not_corr = interpolate_NOR_to_Pathology(NOR_ES_sub_interp, DCM_ES_psuedo_tn,20)

RV_20_HCM_ED_not_corr = interpolate_NOR_to_Pathology(NOR_ED_sub_interp, RV_ED_psuedo_tn,20)
RV_20_HCM_ES_not_corr = interpolate_NOR_to_Pathology(NOR_ES_sub_interp, RV_ES_psuedo_tn,20)

##### comparing correlated anc uncorrelated sampling for NOR to HCM #####

NOR_20_HCM_ED_corr = interpolate_NOR_to_Pathology(NOR_ED_sub_interp, HCM_ED_psuedo_tn_correlated,20)
NOR_20_HCM_ES_corr = interpolate_NOR_to_Pathology(NOR_ES_sub_interp, HCM_ES_psuedo_tn_correlated,20)

DCM_20_HCM_ED_corr = interpolate_NOR_to_Pathology(NOR_ED_sub_interp, DCM_ED_psuedo_tn_correlated,20)
DCM_20_HCM_ES_corr = interpolate_NOR_to_Pathology(NOR_ES_sub_interp, DCM_ES_psuedo_tn_correlated,20)

RV_20_HCM_ED_corr = interpolate_NOR_to_Pathology(NOR_ED_sub_interp, RV_ED_psuedo_tn_correlated,20)
RV_20_HCM_ES_corr = interpolate_NOR_to_Pathology(NOR_ES_sub_interp, RV_ES_psuedo_tn_correlated,20)

RV_20_HCM_ES_corr.shape
# %%
one_subject_np_interp_ED = concatinate_interpolate(NOR_sub_image_ED)
one_subject_np_interp_ES = concatinate_interpolate(NOR_sub_image_ES)

def visual_interactive(NOR_sub_image_ED, NOR_20_HCM_ED_corr ):
    one_subject_np_interp_ED = concatinate_interpolate(NOR_sub_image_ED)
    how_many = 20-1
    show_NOR_to_Path_synthesize(NOR_20_HCM_ED_corr,one_subject_np_interp_ED, how_many, NOR_20_HCM_ED_corr.shape[1]-1)
    
    
visual_interactive(NOR_sub_image_ES, NOR_20_HCM_ES_corr )
# %%
visual_interactive(NOR_sub_image_ES, DCM_20_HCM_ES_not_corr )
# %%
