import skimage
import numpy as np
import numbers
import torchvision.transforms.functional as F
from torchvision import transforms
from PIL import Image
import torch

from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
# import torchio as tio
import nibabel as nib


class MTTransform(object):

    def __call__(self, sample):
        raise NotImplementedError("You need to implement the transform() method.")

    def undo_transform(self, sample):
        raise NotImplementedError("You need to implement the undo_transform() method.")


class UndoCompose(object):
    def __init__(self, compose):
        self.transforms = compose.transforms

    def __call__(self):
        for t in self.transforms:
            img = t.undo_transform(img)
        return img


class UndoTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        return self.transform.undo_transform(sample)


class ToTensor(MTTransform):
    """Convert a PIL image or numpy array to a PyTorch tensor."""

    def __init__(self, labeled=True):
        self.labeled = labeled

    def __call__(self, sample):
        rdict = {}
        input_data = sample['input']

        if isinstance(input_data, list):
            ret_input = [F.to_tensor(item)
                         for item in input_data]
        else:
            ret_input = F.to_tensor(input_data)

        rdict['input'] = ret_input

        if self.labeled:
            gt_data = sample['gt']
            if gt_data is not None:
                if isinstance(gt_data, list):
                    ret_gt = [F.to_tensor(item)
                              for item in gt_data]
                else:
                    ret_gt = F.to_tensor(gt_data)

                rdict['gt'] = ret_gt
        sample.update(rdict)
        return sample


class ToPIL(MTTransform):
    def __init__(self, labeled=True):
        self.labeled = labeled

    def sample_transform(self, sample_data):
        # Numpy array
        if not isinstance(sample_data, np.ndarray):
            input_data_npy = sample_data.numpy()
        else:
            input_data_npy = sample_data

        input_data_npy = np.transpose(input_data_npy, (1, 2, 0))
        input_data_npy = np.squeeze(input_data_npy, axis=2)
        input_data = Image.fromarray(input_data_npy, mode='F')
        return input_data

    def __call__(self, sample):
        rdict = {}
        input_data = sample['input']

        if isinstance(input_data, list):
            ret_input = [self.sample_transform(item)
                         for item in input_data]
        else:
            ret_input = self.sample_transform(input_data)

        rdict['input'] = ret_input

        if self.labeled:
            gt_data = sample['gt']

            if isinstance(gt_data, list):
                ret_gt = [self.sample_transform(item)
                          for item in gt_data]
            else:
                ret_gt = self.sample_transform(gt_data)

            rdict['gt'] = ret_gt

        sample.update(rdict)
        return sample


class UnCenterCrop2D(MTTransform):
    def __init__(self, size, segmentation=True):
        self.size = size
        self.segmentation = segmentation

    def __call__(self, sample):
        input_data, gt_data = sample['input'], sample['gt']
        input_metadata, gt_metadata = sample['input_metadata'], sample['gt_metadata']

        (fh, fw, w, h) = input_metadata["__centercrop"]
        (fh, fw, w, h) = gt_metadata["__centercrop"]

        return sample


class CenterCrop2D(MTTransform):
    """Make a center crop of a specified size.

    :param segmentation: if it is a segmentation task.
                         When this is True (default), the crop
                         will also be applied to the ground truth.
    """
    def __init__(self, size, labeled=True):
        self.size = size
        self.labeled = labeled

    @staticmethod
    def propagate_params(sample, params):
        input_metadata = sample['input_metadata']
        input_metadata["__centercrop"] = params
        return input_metadata

    @staticmethod
    def get_params(sample):
        input_metadata = sample['input_metadata']
        return input_metadata["__centercrop"]

    def __call__(self, sample):
        rdict = {}
        input_data = sample['input']

        w, h = input_data.size
        th, tw = self.size
        fh = int(round((h - th) / 2.))
        fw = int(round((w - tw) / 2.))

        params = (fh, fw, w, h)
        self.propagate_params(sample, params)

        input_data = F.center_crop(input_data, self.size)
        rdict['input'] = input_data

        if self.labeled:
            gt_data = sample['gt']
            gt_metadata = sample['gt_metadata']
            gt_data = F.center_crop(gt_data, self.size)
            gt_metadata["__centercrop"] = (fh, fw, w, h)
            rdict['gt'] = gt_data


        sample.update(rdict)
        return sample

    def undo_transform(self, sample):
        rdict = {}
        input_data = sample['input']
        fh, fw, w, h = self.get_params(sample)
        th, tw = self.size

        pad_left = fw
        pad_right = w - pad_left - tw
        pad_top = fh
        pad_bottom = h - pad_top - th

        padding = (pad_left, pad_top, pad_right, pad_bottom)
        input_data = F.pad(input_data, padding)
        rdict['input'] = input_data

        sample.update(rdict)
        return sample


class Normalize(MTTransform):
    """Normalize a tensor image with mean and standard deviation.

    :param mean: mean value.
    :param std: standard deviation value.
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        input_data = sample['input']

        input_data = F.normalize(input_data, self.mean, self.std)

        rdict = {
            'input': input_data,
        }
        sample.update(rdict)
        return sample




# class PercentileBasedRescaling(MTTransform):

#     def __init__(self, out_min_max = (0, 1), percentiles=(5,95), masking_method=None, p=1.0, labeled=False):

#         '''
#         Rescale intensity values to a certain range.
        
#         Parameters:	
#             -out_min_max – Range (nmin,nmax) of output intensities. If only one value d is provided, (nmin,nmax)=(−d,d).
#             -percentiles – Percentile values of the input image that will be mapped to (nmin,nmax). They can be used for contrast stretching, as in this scikit-image example. For example, Isensee et al. use (0.5, 99.5) in their nn-UNet paper. If only one value d is provided, (nmin,nmax)=(0,d).
#             -masking_method – See NormalizationTransform.
#             -p – Probability that this transform will be applied.

#         Calling the function: PercentileBasedRescaling((0,1), percentiles=(1,99))
#         '''

#         self.out_min_max=out_min_max
#         self.percentiles=percentiles
#         self.masking_method=masking_method
#         self.p=p
#         self.labeled=labeled 

#     def __call__(self, sample):

#         rdict={}
#         input_data=sample["input"]
#         input_data=input_data.unsqueeze(0)

#         rescale=tio.transforms.RescaleIntensity(self.out_min_max, self.percentiles, self.masking_method)
#         input_data=rescale(input_data)

#         input_data=input_data.squeeze(0)
#         rdict['input'] = input_data

#         if self.labeled:
#             gt_data = sample['gt']
#             if gt_data is not None:
#                 gt_data=gt_data.unsqueeze(0)
#                 gt_data=rescale(gt_data)
#                 gt_data=gt_data.squeeze(0)

#                 rdict["gt"]=gt_data

#         sample.update(rdict)
#         return sample

class NormalizeInstance(MTTransform):
    """Normalize a tensor image with mean and standard deviation estimated
    from the sample itself.

    :param mean: mean value.
    :param std: standard deviation value.
    """
    def __call__(self, sample):
        input_data = sample['input']

        mean, std = input_data.mean(), input_data.std()
        input_data = F.normalize(input_data, [mean], [std])

        rdict = {
            'input': input_data,
        }
        sample.update(rdict)
        return sample

class NormalizeInstance3D(MTTransform):
    """Normalize a tensor volume with mean and standard deviation estimated
    from the sample itself.

    :param mean: mean value.
    :param std: standard deviation value.
    """
    def __call__(self, sample):
        input_data = sample['input']

        mean, std = input_data.mean(), input_data.std()

        if mean != 0 or std != 0:
            input_data_normalized = F.normalize(input_data,
                                    [mean for _ in range(0,input_data.shape[0])],
                                    [std for _ in range(0,input_data.shape[0])])

            rdict = {
                'input': input_data_normalized,
            }
            sample.update(rdict)
        return sample


class RandomRotation3D(MTTransform):
    """Make a rotation of the volume's values.

    :param degrees: Maximum rotation's degrees.
    :param axis: Axis of the rotation.
    """
    def __init__(self, degrees, axis=0, labeled=True):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees
        self.labeled = labeled
        self.axis = axis

    @staticmethod
    def get_params(degrees):
        angle = np.random.uniform(degrees[0], degrees[1])
        return angle

    def __call__(self, sample):
        rdict = {}
        input_data = sample['input']
        if len(sample['input'].shape) != 3:
            raise ValueError("Input of RandomRotation3D should be a 3 dimensionnal tensor.")

        angle = self.get_params(self.degrees)
        input_rotated = np.zeros(input_data.shape, dtype=input_data.dtype)
        gt_data = sample['gt'] if self.labeled else None
        gt_rotated = np.zeros(gt_data.shape, dtype=gt_data.dtype) if self.labeled else None

        # TODO: Would be faster with only one vectorial operation
        # TODO: Use the axis index for factoring this loop
        for x in range(input_data.shape[self.axis]):
            if self.axis == 0:
                input_rotated[x,:,:] = F.rotate(Image.fromarray(input_data[x,:,:], mode='F'), angle)
                if self.labeled:
                    gt_rotated[x,:,:] = F.rotate(Image.fromarray(gt_data[x,:,:], mode='F'), angle)
            if self.axis == 1:
                input_rotated[:,x,:] = F.rotate(Image.fromarray(input_data[:,x,:], mode='F'), angle)
                if self.labeled:
                    gt_rotated[:,x,:] = F.rotate(Image.fromarray(gt_data[:,x,:], mode='F'), angle)
            if self.axis == 2:
                input_rotated[:,:,x] = F.rotate(Image.fromarray(input_data[:,:,x], mode='F'), angle)
                if self.labeled:
                    gt_rotated[:,:,x] = F.rotate(Image.fromarray(gt_data[:,:,x], mode='F'), angle)

        rdict['input'] = input_rotated
        if self.labeled : rdict['gt'] = gt_rotated
        sample.update(rdict)

        return sample

class RandomReverse3D(MTTransform):
    """Make a symmetric inversion of the different values of each dimensions.
    (randomized)
    """
    def __init__(self, labeled=True):
        self.labeled = labeled

    def __call__(self, sample):
        rdict = {}
        input_data = sample['input']
        gt_data = sample['gt'] if self.labeled else None
        if np.random.randint(2) == 1:
            input_data = np.flip(input_data,axis=0).copy()
            if self.labeled: gt_data = np.flip(gt_data,axis=0).copy()
        if np.random.randint(2) == 1:
            input_data = np.flip(input_data,axis=1).copy()
            if self.labeled: gt_data = np.flip(gt_data,axis=1).copy()
        if np.random.randint(2) == 1:
            input_data = np.flip(input_data,axis=2).copy()
            if self.labeled: gt_data = np.flip(gt_data,axis=2).copy()

        rdict['input'] = input_data
        if self.labeled : rdict['gt'] = gt_data

        sample.update(rdict)
        return sample

class RandomAffine(MTTransform):
    def __init__(self, degrees, translate=None,
                 scale=None, shear=None,
                 resample=False, fillcolor=0,
                 labeled=True):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
                "degrees should be a list or tuple and it must be of length 2."
            self.degrees = degrees

        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            if isinstance(shear, numbers.Number):
                if shear < 0:
                    raise ValueError("If shear is a single number, it must be positive.")
                self.shear = (-shear, shear)
            else:
                assert isinstance(shear, (tuple, list)) and len(shear) == 2, \
                    "shear should be a list or tuple and it must be of length 2."
                self.shear = shear
        else:
            self.shear = shear

        self.resample = resample
        self.fillcolor = fillcolor
        self.labeled = labeled

    @staticmethod
    def get_params(degrees, translate, scale_ranges, shears, img_size):
        """Get parameters for affine transformation
        Returns:
            sequence: params to be passed to the affine transformation
        """
        angle = np.random.uniform(degrees[0], degrees[1])
        if translate is not None:
            max_dx = translate[0] * img_size[0]
            max_dy = translate[1] * img_size[1]
            translations = (np.round(np.random.uniform(-max_dx, max_dx)),
                            np.round(np.random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = np.random.uniform(scale_ranges[0], scale_ranges[1])
        else:
            scale = 1.0

        if shears is not None:
            shear = np.random.uniform(shears[0], shears[1])
        else:
            shear = 0.0

        return angle, translations, scale, shear

    def sample_augment(self, input_data, params):
        input_data = F.affine(input_data, *params, resample=self.resample,
                              fillcolor=self.fillcolor)
        return input_data

    def label_augment(self, gt_data, params):
        gt_data = self.sample_augment(gt_data, params)
        np_gt_data = np.array(gt_data)
        np_gt_data[np_gt_data >= 0.5] = 1.0
        np_gt_data[np_gt_data < 0.5] = 0.0
        gt_data = Image.fromarray(np_gt_data, mode='F')
        return gt_data

    def __call__(self, sample):
        """
            img (PIL Image): Image to be transformed.
        Returns:
            PIL Image: Affine transformed image.
        """
        rdict = {}
        input_data = sample['input']

        if isinstance(input_data, list):
            input_data_size = input_data[0].size
        else:
            input_data_size = input_data.size

        params = self.get_params(self.degrees, self.translate, self.scale,
                                 self.shear, input_data_size)

        if isinstance(input_data, list):
            ret_input = [self.sample_augment(item, params)
                         for item in input_data]
        else:
            ret_input = self.sample_augment(input_data, params)

        rdict['input'] = ret_input

        if self.labeled:
            gt_data = sample['gt']
            if isinstance(gt_data, list):
                ret_gt = [self.label_augment(item, params)
                          for item in gt_data]
            else:
                ret_gt = self.label_augment(gt_data, params)

            rdict['gt'] = ret_gt

        sample.update(rdict)
        return sample

class RandomTensorChannelShift(MTTransform):
    def __init__(self, shift_range):
        self.shift_range = shift_range

    @staticmethod
    def get_params(shift_range):
        sampled_value = np.random.uniform(shift_range[0],
                                          shift_range[1])
        return sampled_value

    def sample_augment(self, input_data, params):
        np_input_data = np.array(input_data)
        np_input_data += params
        input_data = Image.fromarray(np_input_data, mode='F')
        return input_data

    def __call__(self, sample):
        input_data = sample['input']
        params = self.get_params(self.shift_range)

        if isinstance(input_data, list):
            #ret_input = [self.sample_augment(item, params)
            #             for item in input_data]

            # Augment just the image, not the mask
            # TODO: fix it later
            ret_input = []
            ret_input.append(self.sample_augment(input_data[0], params))
            ret_input.append(input_data[1])
        else:
            ret_input = self.sample_augment(input_data, params)

        rdict = {
            'input': ret_input,
        }

        sample.update(rdict)
        return sample


class ElasticTransform(MTTransform):
    def __init__(self, alpha_range, sigma_range,
                 p=0.5, labeled=True):
        self.alpha_range = alpha_range
        self.sigma_range = sigma_range
        self.labeled = labeled
        self.p = p

    @staticmethod
    def get_params(alpha, sigma):
        alpha = np.random.uniform(alpha[0], alpha[1])
        sigma = np.random.uniform(sigma[0], sigma[1])
        return alpha, sigma

    @staticmethod
    def elastic_transform(image, alpha, sigma):
        shape = image.shape
        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1),
                             sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1),
                             sigma, mode="constant", cval=0) * alpha

        x, y = np.meshgrid(np.arange(shape[0]),
                           np.arange(shape[1]), indexing='ij')
        indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))
        return map_coordinates(image, indices, order=1).reshape(shape)

    def sample_augment(self, input_data, params):
        param_alpha, param_sigma = params

        np_input_data = np.array(input_data)
        np_input_data = self.elastic_transform(np_input_data,
                                               param_alpha, param_sigma)
        input_data = Image.fromarray(np_input_data, mode='F')
        return input_data

    def label_augment(self, gt_data, params):
        param_alpha, param_sigma = params

        np_gt_data = np.array(gt_data)
        np_gt_data = self.elastic_transform(np_gt_data,
                                            param_alpha, param_sigma)
        np_gt_data[np_gt_data >= 0.5] = 1.0
        np_gt_data[np_gt_data < 0.5] = 0.0
        gt_data = Image.fromarray(np_gt_data, mode='F')

        return gt_data

    def __call__(self, sample):
        rdict = {}

        if np.random.random() < self.p:
            input_data = sample['input']
            params = self.get_params(self.alpha_range,
                                     self.sigma_range)

            if isinstance(input_data, list):
                ret_input = [self.sample_augment(item, params)
                             for item in input_data]
            else:
                ret_input = self.sample_augment(input_data, params)

            rdict['input'] = ret_input

            if self.labeled:
                gt_data = sample['gt']
                if isinstance(gt_data, list):
                    ret_gt = [self.label_augment(item, params)
                              for item in gt_data]
                else:
                    ret_gt = self.label_augment(gt_data, params)

                rdict['gt'] = ret_gt

        sample.update(rdict)
        return sample


# TODO: Resample should keep state after changing state.
#       By changing pixel dimensions, we should be
#       able to return later to the original space.
class Resample(MTTransform):
    def __init__(self, wspace, hspace,
                 interpolation=Image.BILINEAR,
                 labeled=True):
        self.hspace = hspace
        self.wspace = wspace
        self.interpolation = interpolation
        self.labeled = labeled

    def __call__(self, sample):
        rdict = {}
        input_data = sample['input']
        input_metadata = sample['input_metadata']

        # Voxel dimension in mm
        hzoom, wzoom = input_metadata["zooms"]
        hshape, wshape = input_metadata["data_shape"]

        hfactor = hzoom / self.hspace
        wfactor = wzoom / self.wspace

        hshape_new = int(hshape * hfactor)
        wshape_new = int(wshape * wfactor)

        input_data = input_data.resize((wshape_new, hshape_new),
                                       resample=self.interpolation)   
        rdict['input'] = input_data

        if self.labeled:
            gt_data = sample['gt']
            gt_metadata = sample['gt_metadata']
            gt_data = gt_data.resize((wshape_new, hshape_new),
                                     resample=Image.NEAREST) # sina: replacing the interpolation with nearest for labels, and commenting the following code
            # np_gt_data = np.array(gt_data)
            # np_gt_data[np_gt_data >= 0.5] = 1.0
            # np_gt_data[np_gt_data < 0.5] = 0.0
            # gt_data = Image.fromarray(np_gt_data, mode='F')
            rdict['gt'] = gt_data

        sample.update(rdict)
        return sample


# sina starts
class UpdateLabels(MTTransform):
    """
    To map the labels of the simulated dataset to ACDC set
    """
    def __init__(self, source=None, destination=None):
        """
        Find the corresponding classes in the reference dictionary and change the label
        :param source: the dictionary of the current set which needs to be updated
        :param destination: the reference dictionary
        """
        self.dest = destination
        self.source = source

    def __call__(self, sample):
        if self.source != self.dest:
            input_data = sample['gt']
            input_data = np.rint(input_data)
            # print(np.unique(input_data))
            input_data_copy = np.copy(input_data)
            gt_data = np.zeros_like(input_data_copy)
            # print("==================")
            for key in self.dest:
                if key in self.source.keys():
                    val_s = self.source[key]
                    val_d = self.dest[key]
                    # print(val_s, "==>", val_d)
                    # input_data_mod[input_data == val_s] = val_d
                    np.putmask(gt_data,input_data_copy == val_s , val_d)

            rdict = {
                # 'gt': input_data,
                'gt': torch.from_numpy(gt_data)
            }
            sample.update(rdict)
        return sample
class ClipScaleRange(MTTransform):
    """
    To clip the intensity values between 0 and +4000
    and scale between 0 and 1 afterwards.
    To be used after ToTensor()

    """
    def __init__(self, min_intensity= 0, max_intensity=4000):
        self.min_intensity = min_intensity
        self.max_intensity = max_intensity

    def __call__(self, sample):
        input_data = sample['input']

        # torch.clamp(input_data,min=self.min_intensity, max=self.max_intensity, out=input_data)
        # range between 0 and 1
        # input_data = input_data / self.max_intensity
        if input_data.max() != 0:
            # input_data = (input_data / input_data.max())
            # scale between -1 and 1
            input_data =( input_data - 0.5 ) / 0.5
        else: RuntimeError('the maximum value of the input data should not be zero')
        rdict = {
            'input': input_data,
        }
        sample.update(rdict)
        return sample



class ClipTanh(MTTransform):
    """
    To clip the intensity values between 0 and +4000
    and scale between 0 and 1 afterwards.
    To be used after ToTensor()

    """
    def __call__(self, sample):
        input_data = sample['input']
        

        # torch.clamp(input_data,min=self.min_intensity, max=self.max_intensity, out=input_data)

        # a_min_intensity = input_data[input_data!=0].min()
        # a_max_intensity = input_data.max()
        # mean = input_data[input_data!=0].mean()
        # std = input_data[input_data!=0].std()
        

        # range between 0 and 1
        # input_data = input_data / self.max_intensity
        if input_data.max() != 0:
            # if std !=0:
                
                # input_data = (input_data - mean)/std
                # minv = input_data[input_data!=0].min()
                # maxv = input_data[input_data!=0].max()
                # input_data = (input_data - minv)/(maxv-minv)
                # input_data = F.normalize(input_data, mean=0.5, std=0.5)
            # input_data = (input_data-a_min_intensity)/(a_max_intensity-a_min_intensity)
            # input_data = (input_data / input_data.max())
            # scale between -1 and 1
            input_data =( input_data - 0.5 ) / 0.5
            tan = torch.nn.Tanh()
            input_data=tan(input_data)

        else: RuntimeError('the maximum value of the input data should not be zero')
        rdict = {
            'input': input_data,
        }
        sample.update(rdict)
        return sample



class ClipZscoreMinMax(MTTransform):
    """
    To clip the intensity values between 0 and +4000
    and scale between 0 and 1 afterwards.
    To be used after ToTensor()

    """
    def __init__(self, min_intensity= 0, max_intensity=4000):
        self.min_intensity = min_intensity
        self.max_intensity = max_intensity

    def __call__(self, sample):
        input_data = sample['input']
        

        torch.clamp(input_data,min=self.min_intensity, max=self.max_intensity, out=input_data)

        # a_min_intensity = input_data[input_data!=0].min()
        # a_max_intensity = input_data.max()
        mean = input_data[input_data!=0].mean()
        std = input_data[input_data!=0].std()
        

        # range between 0 and 1
        # input_data = input_data / self.max_intensity
        if input_data.max() != 0:
            if std !=0:
                input_data = (input_data - mean)/std
                # minv = input_data[input_data!=0].min()
                # maxv = input_data[input_data!=0].max()
                # input_data = (input_data - minv)/(maxv-minv)
                # input_data = F.normalize(input_data, mean=0.5, std=0.5)
            # input_data = (input_data-a_min_intensity)/(a_max_intensity-a_min_intensity)
            # input_data = (input_data / input_data.max())
            # scale between -1 and 1
                # input_data =( input_data - 0.5 ) / 0.5
        else: RuntimeError('the maximum value of the input data should not be zero')
        rdict = {
            'input': input_data,
        }
        sample.update(rdict)
        return sample

class ClipNormalize(MTTransform):
    """
    To clip the intensity values between 0 and +4000
    and scale between 0 and 1 afterwards.
    To be used after ToTensor()

    """
    def __init__(self, min_intensity= 0, max_intensity=4000):
        self.min_intensity = min_intensity
        self.max_intensity = max_intensity

    def __call__(self, sample):
        input_data = sample['input']

        torch.clamp(input_data,min=self.min_intensity, max=self.max_intensity, out=input_data)
        # range between 0 and 1
        # input_data = input_data / self.max_intensity
        if input_data.max() != 0:
            # input_data = F.normalize(input_data, mean=0.5, std=0.5)
            input_data = F.normalize(input_data, mean=0.5, std=0.5)
            # input_data = (input_data / input_data.max())
            # scale between -1 and 1
            # input_data =( input_data - 0.5 ) / 0.5
        else: RuntimeError('the maximum value of the input data should not be zero')
        rdict = {
            'input': input_data,
        }
        sample.update(rdict)
        return sample

# sina ends

class AdditiveGaussianNoise(MTTransform):
    def __init__(self, mean=0.0, std=0.01):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        rdict = {}
        input_data = sample['input']

        noise = np.random.normal(self.mean, self.std, input_data.size)
        noise = noise.astype(np.float32)

        np_input_data = np.array(input_data)
        np_input_data += noise
        input_data = Image.fromarray(np_input_data, mode='F')
        rdict['input'] = input_data

        sample.update(rdict)
        return sample

class Clahe(MTTransform):
    def __init__(self, clip_limit=3.0, kernel_size=(8, 8)):
        # Default values are based upon the following paper:
        # https://arxiv.org/abs/1804.09400 (3D Consistent Cardiac Segmentation)

        self.clip_limit = clip_limit
        self.kernel_size = kernel_size
    
    def __call__(self, sample):
        if not isinstance(sample, np.ndarray):
            raise TypeError("Input sample must be a numpy array.")
        input_sample = np.copy(sample)
        array = skimage.exposure.equalize_adapthist(
            input_sample,
            kernel_size=self.kernel_size,
            clip_limit=self.clip_limit
        )
        return array


class HistogramClipping(MTTransform):
    def __init__(self, min_percentile=5.0, max_percentile=95.0):
        self.min_percentile = min_percentile
        self.max_percentile = max_percentile

    def __call__(self, sample):
        array = np.copy(sample)
        percentile1 = np.percentile(array, self.min_percentile)
        percentile2 = np.percentile(array, self.max_percentile)
        array[array <= percentile1] = percentile1
        array[array >= percentile2] = percentile2
        return array

import kornia
class RandomHorizontalFlip2D(MTTransform):

  def __init__(self, return_transform=False, same_on_batch=False, p=0.5, p_batch=1.0):
    self.return_transform=return_transform
    self.same_on_batch=same_on_batch
    self.p=p 
    self.p_batch=p_batch 

  def __call__(self, sample):

    if torch.rand(1) < self.p:
        rdict = {}
        input_data = sample['input']
        input_mask = sample['gt']
        input_data=input_data.unsqueeze(0)
        input_mask=input_mask.unsqueeze(0)

        aug=kornia.augmentation.RandomHorizontalFlip(self.return_transform, self.same_on_batch, 1, self.p_batch)
        input_data=aug(input_data)
        input_mask=aug(input_mask)

        input_data=input_data.squeeze(0)
        input_mask=input_mask.squeeze(0)

        rdict = {
                'input': input_data,
                'gt':input_mask
            }
        sample.update(rdict)
    return sample


class RandomVerticalFlip2D(MTTransform):

  def __init__(self, return_transform=False, same_on_batch=False, p=0.5, p_batch=1.0):
    self.return_transform=return_transform
    self.same_on_batch=same_on_batch
    self.p=p 
    self.p_batch=p_batch 

  def __call__(self, sample):
    if torch.rand(1) < self.p:
        rdict = {}
        input_data = sample['input']
        input_mask = sample['gt']
        input_data=input_data.unsqueeze(0)
        input_mask=input_mask.unsqueeze(0)

        aug=kornia.augmentation.RandomVerticalFlip(self.return_transform, self.same_on_batch, 1, self.p_batch)
        input_data=aug(input_data)
        input_mask=aug(input_mask)

        input_data=input_data.squeeze(0)
        input_mask=input_mask.squeeze(0)

        rdict = {
                'input': input_data,
                'gt':input_mask
            }
        sample.update(rdict)
    return sample


class RandomRotation(MTTransform):
    def __init__(self, degrees= 180,p=0.5, resample=False,
                 expand=False, center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center
        self.p = p

    @staticmethod
    def get_params(degrees):
        angle = np.random.uniform(degrees[0], degrees[1])
        return angle

    def __call__(self, sample):
        if torch.rand(1) < self.p:
            rdict = {}
            input_data = sample['input']
            angle = self.get_params(self.degrees)
            # print('the rotation angle is ', angle)
            input_data = F.rotate(input_data, angle,
                                self.resample, self.expand,
                                self.center)
            rdict['input'] = input_data

            
            gt_data = sample['gt']
            gt_data = F.rotate(gt_data, angle,
                            0, self.expand,
                            self.center)
            rdict['gt'] = gt_data

            sample.update(rdict)
        return sample

class RandomRotation90(MTTransform):
    def __init__(self, degrees= 90,p=0.5, resample=False,
                 expand=False, center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center
        self.p = p

    @staticmethod
    def get_params(degrees):
        angle = int(np.random.randint(1, 5) * degrees[0])
        return angle

    def __call__(self, sample):
        if torch.rand(1) < self.p:
            rdict = {}
            input_data = sample['input']
            angle = self.get_params(self.degrees)
            # print('the rotation angle is ', angle)
            input_data = F.rotate(input_data, angle,
                                self.resample, self.expand,
                                self.center)
            rdict['input'] = input_data

            
            gt_data = sample['gt']
            gt_data = F.rotate(gt_data, angle,
                            0, self.expand,
                            self.center)
            rdict['gt'] = gt_data

            sample.update(rdict)
        return sample

# import torchio as tio

# class RandomElasticTorchio(MTTransform):

#     # apply elastic transfromation on a 2D image using torchio
#     # source: https://torchio.readthedocs.io/transforms/augmentation.html#randomelasticdeformation 

#     def __init__(self, num_control_points  = (8, 8, 4), max_displacement  = (16, 16, 0), image_interpolation = 'nearest', locked_borders  = 1, p=0.5):

#         # num_control_points  = 8, 8, 4 # Number of control points along each dimension of the coarse grid
#         # max_displacement  = 42, 42, 0 # Maximum displacement along each dimension at each control point
#         # image_interpolation = 'nearest'
#         # locked_borders  = 1

#         self.num_control_points = num_control_points
#         self.max_displacement = max_displacement
#         self.image_interpolation = image_interpolation
#         self.locked_borders =locked_borders
#         self.p = p

#     def __call__(self, sample):
#         if torch.rand(1) < self.p:
#             rdict = {}
#             input_data = sample['input']
#             input_mask = sample['gt']
#             # print('shape of image ',sample['input'].shape) 1 , 128, 128

#             input_data=input_data.unsqueeze(-1)
#             input_mask=input_mask.unsqueeze(-1)

#             subject = tio.Subject(
#                 image = tio.ScalarImage(tensor=input_data),
#                 mask = tio.LabelMap(tensor=input_mask)
#             )

#             random_elastic = tio.transforms.RandomElasticDeformation(p=1, max_displacement=self.max_displacement, num_control_points=self.num_control_points, image_interpolation= self.image_interpolation ,locked_borders=self.locked_borders)
#             transformed_subject = random_elastic(subject)

#             input_data=transformed_subject['image'][tio.DATA]
#             input_mask=transformed_subject['mask'][tio.DATA]

#             input_data=input_data.squeeze(-1)
#             input_mask=input_mask.squeeze(-1)

#             rdict = {
#                 'input': input_data,
#                 'gt':input_mask
#             }

#             sample.update(rdict)
#         return sample


# class RandomElasticTorchio_label_only(MTTransform):

#     # apply elastic transfromation on a 2D image using torchio
#     # source: https://torchio.readthedocs.io/transforms/augmentation.html#randomelasticdeformation 

#     def __init__(self, num_control_points  = (8, 8, 4), max_displacement  = (16, 16, 0), image_interpolation = 'nearest', locked_borders  = 1, p=0.5):

#         # num_control_points  = 8, 8, 4 # Number of control points along each dimension of the coarse grid
#         # max_displacement  = 42, 42, 0 # Maximum displacement along each dimension at each control point
#         # image_interpolation = 'nearest'
#         # locked_borders  = 1

#         self.num_control_points = num_control_points
#         self.max_displacement = max_displacement
#         self.image_interpolation = image_interpolation
#         self.locked_borders =locked_borders
#         self.p = p

#     def __call__(self, sample):
#         if torch.rand(1) < self.p:
#             rdict = {}
#             input_data = sample['input']
#             input_mask = sample['gt']
#             # print('shape of image ',sample['input'].shape) 1 , 128, 128

#             input_data=input_data.unsqueeze(-1)
#             input_mask=input_mask.unsqueeze(-1)

#             subject = tio.Subject(
#                 image = tio.ScalarImage(tensor=input_data),
#                 mask = tio.LabelMap(tensor=input_mask)
#             )

#             random_elastic = tio.transforms.RandomElasticDeformation(p=1, max_displacement=self.max_displacement, num_control_points=self.num_control_points, image_interpolation= self.image_interpolation ,locked_borders=self.locked_borders)
#             transformed_subject = random_elastic(subject)

#             input_data=transformed_subject['image'][tio.DATA]
#             input_mask=transformed_subject['mask'][tio.DATA]

#             input_data=input_data.squeeze(-1)
#             input_mask=input_mask.squeeze(-1)

#             # 'input': input_data,

#             rdict = {
                
#                 'gt':input_mask
#             }

#             sample.update(rdict)
#         return sample

# class RandomElasticTorchio_label_only_range(MTTransform):

#     # apply elastic transfromation on a 2D image using torchio
#     # source: https://torchio.readthedocs.io/transforms/augmentation.html#randomelasticdeformation 

#     def __init__(self, num_control_points_range  = (8, 8, 4), max_displacement_range  = (16, 16, 0), image_interpolation = 'nearest', locked_borders  = 1, p=0.5):

#         # num_control_points  = 8, 8, 4 # Number of control points along each dimension of the coarse grid
#         # max_displacement  = 42, 42, 0 # Maximum displacement along each dimension at each control point
#         # image_interpolation = 'nearest'
#         # locked_borders  = 1

#         self.num_control_points_range = num_control_points_range
#         self.max_displacement_range = max_displacement_range
#         self.image_interpolation = image_interpolation
#         self.locked_borders =locked_borders
#         self.p = p

#     def __call__(self, sample):
#         if torch.rand(1) < self.p:
#             rdict = {}
#             input_data = sample['input']
#             input_mask = sample['gt']
#             # print('shape of image ',sample['input'].shape) 1 , 128, 128
#             num_control_x = np.random.randint(self.num_control_points_range[0], self.num_control_points_range[1]+1)
#             num_control_y = np.random.randint(self.num_control_points_range[0], self.num_control_points_range[1]+1)
#             num_control_points = (num_control_x, num_control_y, self.num_control_points_range[2])

#             max_displacement_x = np.random.randint(self.max_displacement_range[0], self.max_displacement_range[1]+1)
#             max_displacement_y = np.random.randint(self.max_displacement_range[0], self.max_displacement_range[1]+1)
#             max_displacement = (max_displacement_x, max_displacement_y, self.max_displacement_range[2])


#             input_data=input_data.unsqueeze(-1)
#             input_mask=input_mask.unsqueeze(-1)

#             subject = tio.Subject(
#                 image = tio.ScalarImage(tensor=input_data),
#                 mask = tio.LabelMap(tensor=input_mask)
#             )

#             random_elastic = tio.transforms.RandomElasticDeformation(p=1, max_displacement=max_displacement, num_control_points=num_control_points, image_interpolation= self.image_interpolation ,locked_borders=self.locked_borders)
#             transformed_subject = random_elastic(subject)

#             input_data=transformed_subject['image'][tio.DATA]
#             input_mask=transformed_subject['mask'][tio.DATA]

#             input_data=input_data.squeeze(-1)
#             input_mask=input_mask.squeeze(-1)

#             # 'input': input_data,

#             rdict = {
                
#                 'gt':input_mask
#             }

#             sample.update(rdict)
#         return sample

class RandomRotation_label_only(MTTransform):
    def __init__(self, degrees= 20 ,p=0.5, resample=False,
                 expand=False, center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center
        self.p = p

    @staticmethod
    def get_params(degrees):
        angle = np.random.uniform(degrees[0], degrees[1])
        return angle

    def __call__(self, sample):
        if torch.rand(1) < self.p:
            rdict = {}
            # input_data = sample['input']
            angle = self.get_params(self.degrees)
            # # print('the rotation angle is ', angle)
            # input_data = F.rotate(input_data, angle,
            #                     self.resample, self.expand,
            #                     self.center)
            # rdict['input'] = input_data

            
            gt_data = sample['gt']
            gt_data = F.rotate(gt_data, angle,
                            0, self.expand,
                            self.center)
            rdict['gt'] = gt_data

            sample.update(rdict)
        return sample


import cv2
# def labelerosion(label, kernel, iteration=1):
#     # k: kernel size
#     # iteration: number of times erosion is applied
#     # kernel = np.ones((k,k),np.uint8)
#     return cv2.erode(label,kernel,iterations = iteration)



class RandomDilation_label_only(MTTransform):
    # apply it before ToTensor

    # apply morphological dilation on the labels using openCV
    # source: # Structuring Element for kernel https://docs.opencv.org/master/d9/d61/tutorial_py_morphological_ops.html

    def __init__(self, kernel_shape ='elliptical', kernel_size = 3, iteration_range = (1,3) , p=0.5):

        # kernel_shape  = 8, 8, 4 # Number of control points along each dimension of the coarse grid
        # kernel_size  = 42, 42, 0 # Maximum displacement along each dimension at each control point
        # image_interpolation = 'nearest'
        # locked_borders  = 1

        self.kernel_shape = kernel_shape
        self.kernel_size = kernel_size
        self.iteration_range = iteration_range
        self.p = p
    @staticmethod
    def labeldilation(label, kernel, iteration=1):
    # k: kernel size
    # iteration: number of times dilation  is applied
    # kernel = np.ones((k,k),np.uint8)
        return cv2.dilate(label,kernel,iterations = iteration)

    @staticmethod
    def define_kernel(shape = 'elliptical', ks=5):
    # Structuring Element for kernel https://docs.opencv.org/master/d9/d61/tutorial_py_morphological_ops.html
            # Rectangular Kernel
        if shape == 'rectangular':
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(ks,ks))
        elif shape == 'elliptical':
            # Elliptical Kernel
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ks,ks))
        elif shape == 'corss-shaped':
            # Cross-shaped Kernel
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(ks,ks))
        else:
            raise KeyError('choose between these shapes: rectangular, elliptical, cross-shaped')

        return kernel


    def __call__(self, sample):
        if torch.rand(1) < self.p:
            rdict = {}
        
            input_mask = np.array(sample['gt'])
            # input_mask=input_mask.unsqueeze(-1)
            # print('shape of image ',sample['input'].shape) 1 , 128, 128
            defined_kernel = self.define_kernel(self.kernel_shape, self.kernel_size)
            low, high = self.iteration_range
            iteration = np.random.randint(low, high+1)
            input_mask = self.labeldilation(input_mask, defined_kernel, iteration=iteration)
            input_mask = Image.fromarray(input_mask, mode='F')

            rdict = {
                
                'gt':input_mask
            }

            sample.update(rdict)
        return sample



class RandomElasticTorchio_label_only_RV(MTTransform):

    # apply elastic transfromation on a 2D image using torchio
    # source: https://torchio.readthedocs.io/transforms/augmentation.html#randomelasticdeformation 

    def __init__(self, num_control_points  = (8, 8, 4), max_displacement  = (16, 16, 0), image_interpolation = 'nearest', locked_borders  = 1, p=0.5):

        # num_control_points  = 8, 8, 4 # Number of control points along each dimension of the coarse grid
        # max_displacement  = 42, 42, 0 # Maximum displacement along each dimension at each control point
        # image_interpolation = 'nearest'
        # locked_borders  = 1

        self.num_control_points = num_control_points
        self.max_displacement = max_displacement
        self.image_interpolation = image_interpolation
        self.locked_borders =locked_borders
        self.p = p

    def __call__(self, sample):
        if torch.rand(1) < self.p:
            rdict = {}
            input_data = sample['input']
            input_mask = sample['gt']
            input_mask_RV = torch.zeros_like(sample['gt'])
            input_mask_RV[sample['gt']==3] = 3

            # print('shape of image ',sample['input'].shape) 1 , 128, 128

            input_data=input_data.unsqueeze(-1)
            input_mask=input_mask.unsqueeze(-1)
            input_mask_RV=input_mask_RV.unsqueeze(-1)

            subject = tio.Subject(
                image = tio.ScalarImage(tensor=input_data),
                mask = tio.LabelMap(tensor=input_mask),
                mask_RV = tio.LabelMap(tensor=input_mask_RV)
            )

            random_elastic = tio.transforms.RandomElasticDeformation(p=1, max_displacement=self.max_displacement, num_control_points=self.num_control_points, image_interpolation= self.image_interpolation ,locked_borders=self.locked_borders)
            transformed_subject = random_elastic(subject)

            input_data=transformed_subject['image'][tio.DATA]
            input_mask=transformed_subject['mask'][tio.DATA]
            input_mask_RV=transformed_subject['mask_RV'][tio.DATA]

            input_data=input_data.squeeze(-1)
            input_mask=input_mask.squeeze(-1)
            input_mask_RV=input_mask_RV.squeeze(-1)
            input_mask_RV[sample['gt']==2] = 2
            input_mask_RV[sample['gt']==1] = 1

            # 'input': input_data,

            rdict = {
                
                'gt':input_mask_RV
            }

            sample.update(rdict)
        return sample