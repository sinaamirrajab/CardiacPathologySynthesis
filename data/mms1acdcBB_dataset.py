"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from data.base_dataset import BaseDataset



import os
import nibabel as nib
import util.cmr_dataloader as cmr
import util.cmr_transform as cmr_tran
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

# TR_CLASS_MAP_MMS_SRS= {'MYO': 2,'LV_Blood': 1, 'Scar': 3,'BG': 0,'NO_reflow': 4}
# TR_CLASS_MAP_MMS_DES= {'MYO': 2,'LV_Blood': 1, 'Scar': 3,'BG': 0,'NO_reflow': 4}
# TR_CLASS_MAP_MMS_SRS= {'MYO': 2,'LV_Blood': 1, 'Scar': 3,'NO_reflow': 4}
# TR_CLASS_MAP_MMS_DES= {'MYO': 2,'LV_Blood': 1, 'Scar': 3,'NO_reflow': 0}

# TR_CLASS_MAP_MMS_SRS= {'MYO': 2,'LV_Blood': 1, 'Scar': 3,'NO_reflow': 4}
# TR_CLASS_MAP_MMS_DES= {'MYO': 0,'LV_Blood': 1, 'Scar': 2,'NO_reflow': 3}
# sina feb 2021 for the heart with separated  labels
# TR_CLASS_MAP_MMS_SRS= {'BG': 0,'LV_Bloodpool': 8, 'LV_Myocardium': 9,'RV_Bloodpool': 10,'abdomen': 4,'Body_fat': 2,'vessel': 6, 'extra_heart': 1, 'Lung': 5,'Skeletal': 3}
# TR_CLASS_MAP_MMS_DES= {'BG': 0,'LV_Bloodpool': 1, 'LV_Myocardium': 2,'RV_Bloodpool': 3,'abdomen': 4,'Body_fat': 5,'vessel': 6, 'extra_heart': 7, 'Lung': 8 ,'Skeletal': 9}

TR_CLASS_MAP_MMS_SRS= {'BG': 0,'LV_Bloodpool': 1, 'LV_Myocardium': 2,'RV_Bloodpool': 3}
TR_CLASS_MAP_MMS_DES= {'BG': 0,'LV_Bloodpool': 1, 'LV_Myocardium': 2,'RV_Bloodpool': 3}

class Mms1acdcBBDataset(BaseDataset):
    """ Dataset that loads images from directories
        Use option --label_dir, --image_dir, --instance_dir to specify the directories.
        The images in the directories are sorted in alphabetical order and paired in order.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = BaseDataset.modify_commandline_options(parser, is_train)
       
        # parser.set_defaults(label_nc=4)
        parser.set_defaults(output_nc=1)
        # parser.set_defaults(crop_size=128)
        parser.set_defaults(no_instance=True)
        parser.set_defaults(add_dist=False)
        

        parser.add_argument('--label_dir', type=str, required=False, default = "/data/sina/dataset/cmrVAE/mms1/training_crop_noBA_NR/vendors/Vendor_A/Mask/",
                            help='path to the directory that contains label images')
        parser.add_argument('--label_dir_B', type=str, required=False, default = "/data/sina/dataset/cmrVAE/mms1/training_crop_noBA_NR/vendors/Vendor_B/Mask/",
                            help='path to the directory that contains label images')
        parser.add_argument('--image_dir', type=str, required=False, default ="/data/sina/dataset/cmrVAE/mms1/training_crop_noBA_NR/vendors/Vendor_A/Image/" ,
                            help='path to the directory that contains photo images')
        parser.add_argument('--image_dir_B', type=str, required=False, default ="/data/sina/dataset/cmrVAE/mms1/training_crop_noBA_NR/vendors/Vendor_B/Image/" ,
                            help='path to the directory that contains photo images')
        parser.add_argument('--instance_dir', type=str, default='',
                            help='path to the directory that contains instance maps. Leave black if not exists')
        parser.add_argument('--acdc_dir', type=str, required=False, default = "/data/sina/dataset/ACDC/pathology_crop_noBA_NR_C128/",
                            help='path to the directory that contains label images')
                        
        return parser

    def get_paths(self, opt):
        """
        To prepare and get the list of files
        """

        SA_image_list = sorted(os.listdir(os.path.join(opt.image_dir)))
        SA_mask_list = sorted(os.listdir(os.path.join(opt.label_dir)))

        SA_image_list_B = sorted(os.listdir(os.path.join(opt.image_dir_B)))
        SA_mask_list_B = sorted(os.listdir(os.path.join(opt.label_dir_B)))

        pathologies = sorted(os.listdir(os.path.join(opt.acdc_dir)))
        
        

        assert len(SA_mask_list_B) == len(SA_image_list_B) 
        assert len(SA_image_list) == len(SA_mask_list)


        SA_filename_pairs = [] 
        SA_filename_pairs_B = []

        SA_filename_pairs_acdc = [] 
        SA_image_list_acdc_all = []
        SA_mask_list_acdc_all = []

        what_pathology = 'all'
        if what_pathology == 'all':
            for pathology in pathologies:
                SA_image_list_acdc = sorted(os.listdir(os.path.join(opt.acdc_dir, pathology, 'Image')))
                SA_mask_list_acdc = sorted(os.listdir(os.path.join(opt.acdc_dir, pathology, 'Label_c')))
                SA_image_list_acdc_all += SA_image_list_acdc
                SA_mask_list_acdc_all += SA_mask_list_acdc
                for i in range(len(SA_image_list_acdc)):
                    SA_filename_pairs_acdc += [(os.path.join(opt.acdc_dir, pathology, 'Image',SA_image_list_acdc[i]), os.path.join(opt.acdc_dir, pathology, 'Label_c', SA_mask_list_acdc[i]))]



        # for i in range(len(LA_image_list)):
        #     LA_filename_pairs += [(os.path.join(opt.main_dir,str(LA_image_list[i].split('_')[0]),LA_image_list[i]), os.path.join(opt.main_dir,str(LA_image_list[i].split('_')[0]),LA_mask_list[i]) )]
        #     if not opt.no_Short_axis:
        #         SA_filename_pairs += [(os.path.join(opt.main_dir,str(SA_image_list[i].split('_')[0]),SA_image_list[i]), os.path.join(opt.main_dir,str(SA_image_list[i].split('_')[0]),SA_mask_list[i]) )]
        # print('the size of the image list', len(SA_image_list))
        for i in range(len(SA_image_list)):
            SA_filename_pairs += [(os.path.join(opt.image_dir,SA_image_list[i]), os.path.join(opt.label_dir, SA_mask_list[i]))]

        for i in range(len(SA_image_list_B)):
            SA_filename_pairs_B += [(os.path.join(opt.image_dir_B,SA_image_list_B[i]), os.path.join(opt.label_dir_B, SA_mask_list_B[i]))]

        # for i in range(len(LA_image_list)):
        #     LA_filename_pairs += [(os.path.join(opt.main_dir, 'Image',LA_image_list[i]), os.path.join(opt.main_dir, label, LA_mask_list[i]))]
                

        imglist = []
        msklist = []
        filename_pairs = []
        if not opt.VAE_altered_anatomy: # use the VAE deformed version of the labels
            imglist = SA_image_list + SA_image_list_B
            msklist = SA_mask_list + SA_mask_list_B
            filename_pairs = SA_filename_pairs + SA_filename_pairs_B

        if not opt.selected_labels:
            imglist = SA_image_list + SA_image_list_B
            msklist = SA_mask_list + SA_mask_list_B
            filename_pairs = SA_filename_pairs + SA_filename_pairs_B
        else:
            imglist = SA_image_list 
            msklist = SA_mask_list 
            filename_pairs = SA_filename_pairs

        if opt.what_data == 'acdc':
            self.img_list = SA_image_list_acdc_all
            self.msk_list = SA_mask_list_acdc_all
            self.filename_pairs = SA_filename_pairs_acdc
        else:
            self.img_list = imglist + SA_image_list_acdc_all
            self.msk_list = msklist + SA_mask_list_acdc_all
            self.filename_pairs = filename_pairs + SA_filename_pairs_acdc
        
    
        return self.filename_pairs, self.img_list, self.msk_list



    def initialize(self, opt):
        self.opt = opt

        self.filename_pairs, _, _  = self.get_paths(self.opt)


        print('the size of the image list', len(self.filename_pairs))
   

        if opt.isTrain:
            train_transforms = Compose([
                # cmr_tran.Resample(self.opt.target_res,self.opt.target_res), #1.33
                # cmr_tran.CenterCrop2D((self.opt.crop_size,self.opt.crop_size)),
                
                # cmr_tran.RandomRotation(degrees=90),
                # cmr_tran.RandomRotation(p=0.5),
                cmr_tran.RandomRotation90(p=0.7),
                
                cmr_tran.ToTensor(),
                # cmr_tran.NormalizeLabel(),

                # cmr_tran.PercentileBasedRescaling(out_min_max=(-1,1), percentiles=(1,99)),  #TODO: make sure the normalization is performed on the volume data not slice-by-slice
                # cmr_tran.RandomElasticTorchio(num_control_points  = (8, 8, 4), max_displacement  = (20, 20, 0), p=0.5),
                # cmr_tran.ClipScaleRange(min_intensity= 0, max_intensity=1),
                # cmr_tran.ClipTanh(),
                # cmr_tran.ClipScaleRange(),
                # cmr_tran.ClipNormalize(min_intensity= 0, max_intensity=4000),
                # cmr_tran.ClipZscoreMinMax(min_intensity= 0, max_intensity=4000),
                
                cmr_tran.RandomHorizontalFlip2D(p=0.7),
                cmr_tran.RandomVerticalFlip2D(p=0.7),
                # cmr_tran.UpdateLabels(source=TR_CLASS_MAP_MMS_SRS, destination=TR_CLASS_MAP_MMS_DES)

            ])
        else:
            train_transforms = Compose([
                # cmr_tran.Resample(self.opt.target_res,self.opt.target_res), #1.33
                # cmr_tran.CenterCrop2D((self.opt.crop_size,self.opt.crop_size)),
                # cmr_tran.RandomDilation_label_only(kernel_shape ='elliptical', kernel_size = 3, iteration_range = (1,2) , p=0.5),
                # cmr_tran.RandomRotation(degrees=90),
                # cmr_tran.RandomRotation(p=0.5),
                cmr_tran.ToTensor(),
                # cmr_tran.PercentileBasedRescaling(out_min_max=(-1,1), percentiles=(1,99)),  #TODO: make sure the normalization is performed on the volume data not slice-by-slice
                # cmr_tran.RandomElasticTorchio_label_only(num_control_points  = (8, 8, 4), max_displacement  = (14, 14, 1), p=1),
                
                # cmr_tran.RandomElasticTorchio(num_control_points  = (8, 8, 4), max_displacement  = (20, 20, 0), p=0.5),
                # cmr_tran.ClipScaleRange(min_intensity= 0, max_intensity=4000),
                # cmr_tran.ClipTanh(),
                # cmr_tran.ClipScaleRange(),
                # cmr_tran.RandomHorizontalFlip2D(p=0.5),
                # cmr_tran.RandomVerticalFlip2D(p=0.5),
                # cmr_tran.UpdateLabels(source=TR_CLASS_MAP_MMS_SRS, destination=TR_CLASS_MAP_MMS_DES)

            ])
        
        self.cmr_dataset = cmr.MRI2DSegmentationDataset(self.filename_pairs, transform = train_transforms, slice_axis=2, canonical = False)
        
        
        size = len(self.cmr_dataset)
        self.dataset_size = size


    def __getitem__(self, index):
        # Label Image
        data_input = self.cmr_dataset[index]
        # if using instance maps
        if self.opt.no_instance:
            instance_tensor = 0
        else:
            instance_tensor = data_input["gt"] # the label map equals the instance map for this dataset
        if not self.opt.add_dist:
            dist_tensor = 0
        input_dict = {'label': data_input['gt'],
                      'image': data_input['input'],
                      'instance': instance_tensor,
                      'dist': dist_tensor,
                      'path': data_input['filename'],
                      'gtname': data_input['gtname'],
                      'index': data_input['index'],
                      'segpair_slice': data_input['segpair_slice'],
                      }

        return input_dict
    
    def __len__(self):
        return self.cmr_dataset.__len__()