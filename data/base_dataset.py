"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets.

It also includes common transformation functions (e.g., get_transform, __scale_width), which can be later used in subclasses.
"""
import random
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from abc import ABC, abstractmethod


class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.

    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the class; save the options in the class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        self.root = opt.dataroot

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        pass


def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.preprocess == 'resize_and_crop':
        new_h = new_w = opt.load_size
    elif opt.preprocess == 'scale_width_and_crop':
        new_w = opt.load_size
        new_h = opt.load_size * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
    y = random.randint(0, np.maximum(0, new_h - opt.crop_size))

    flip = random.random() > 0.5

    return {'crop_pos': (x, y), 'flip': flip}


def get_transform(opt, params=None, grayscale=False, ct_domain=False, method=transforms.InterpolationMode.BICUBIC, convert=True):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    if ct_domain:
        transform_list.append(transforms.Lambda(lambda img: Image.fromarray((np.clip(np.array(img), -1000, 600) + 1000) / 1600)))
    if 'resize' in opt.preprocess:
        osize = [opt.load_size, opt.load_size]
        transform_list.append(transforms.Resize(osize, method))
    elif 'scale_width' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size, opt.crop_size, method)))

    if 'crop' in opt.preprocess:
        if params is None or 'crop_pos' not in params:
            transform_list.append(transforms.RandomCrop(opt.crop_size))
        else:
            transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.crop_size)))

    if opt.preprocess == 'none':
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))

    if not opt.no_flip:
        if params is None or 'flip' not in params:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif params['flip']:
            transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    if convert:
        transform_list += [transforms.ToTensor()]
        if grayscale or ct_domain:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def get_transform_sync(opt, params=None, grayscale=False, ct_domain=False, method=transforms.InterpolationMode.BICUBIC, convert=True):
    transform_list = []
    if ct_domain:
        transform_list.append(transforms.Lambda(lambda x: {'image': Image.fromarray((np.clip(np.array(x['image']), -1000, 600) + 1000) / 1600),
                                                           'mask': x['mask']}))
    if 'crop' in opt.preprocess:
        if params is None or 'crop_pos' not in params:
            # transform_list.append(transforms.RandomCrop(opt.crop_size))
            transform_list.append(RandomCropAndPad_Sync(opt.crop_size, normal=True))
        else:
            transform_list.append(transforms.Lambda(lambda x: {'image': __crop(x['image'], params['crop_pos'], opt.crop_size),
                                                               'mask': __crop(x['mask'], params['crop_pos'], opt.crop_size)}))
    if opt.preprocess == 'none':
        transform_list.append(transforms.Lambda(lambda x: {'image': __make_power_2(x['image'], base=4, method=method),
                                                           'mask': __make_power_2(x['mask'], base=4, method=method)}))
        
    if not opt.no_flip:
        if params is None or 'flip' not in params:
            transform_list.append(RandomHorizontalFlip_Sync())
        elif params['flip']:
            transform_list.append(transforms.Lambda(lambda x: {'image':__flip(x['image'], params['flip']),
                                                               'mask': __flip(x['mask'], params['flip'])})) 
    if convert:
        transform_list.append(transforms.Lambda(lambda x: {'image': transforms.ToTensor()(x['image']),
                                                           'mask': transforms.ToTensor()(x['mask'])}))
        if grayscale or ct_domain:
            transform_list.append(transforms.Lambda(lambda x: {'image': transforms.Normalize((0.5,), (0.5,))(x['image']),
                                                               'mask': x['mask']}))
        else:
            transform_list.append(transforms.Lambda(lambda x: {'image': transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(x['image']),
                                                               'mask': x['mask']}))
    return transforms.Compose(transform_list)
    
    
def get_3d_transform(opt, params=None, grayscale=False, ct_domain=False, method=transforms.InterpolationMode.BICUBIC, convert=True):
    transform_list = []
    transform_list.append(transforms.Lambda(lambda x: {'image': x['image'].transpose(2, 0, 1),
                                                       'mask': x['mask'].transpose(2, 0, 1)}))
    if ct_domain:
        transform_list.append(transforms.Lambda(lambda x: {'image': (np.clip(x['image'], -1000, 600) + 1000) / 1600,
                                                           'mask': x['mask']}))
        
    transform_list.append(transforms.Lambda(lambda x: {'image': torch.Tensor(x['image']),
                                                       'mask': torch.Tensor(x['mask'])}))
    
    if 'crop' in opt.preprocess:
        if params is None or 'crop_pos' not in params:
            # transform_list.append(transforms.RandomCrop(opt.crop_size))
            transform_list.append(RandomCropAndPad_Sync(opt.crop_size, normal=False, padding='zeros'))
        else:
            transform_list.append(transforms.Lambda(lambda x: {'image': __crop(x['image'], params['crop_pos'], opt.crop_size),
                                                               'mask': __crop(x['mask'], params['crop_pos'], opt.crop_size)}))
    else:
        # padding
        transform_list.append(transforms.Lambda(lambda x: __ensure_dividable_by_4(x)))
            
    if not opt.no_flip:
        if params is None or 'flip' not in params:
            transform_list.append(RandomHorizontalFlip_Sync())
        elif params['flip']:
            transform_list.append(transforms.Lambda(lambda x: {'image':__flip(x['image'], params['flip']),
                                                               'mask': __flip(x['mask'], params['flip'])})) 

    if convert:
        if grayscale or ct_domain:
            transform_list.append(transforms.Lambda(lambda x: {'image': transforms.Normalize((0.5,), (0.5,))(x['image']),
                                                               'mask': x['mask']}))
        else:
            transform_list.append(transforms.Lambda(lambda x: {'image': transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(x['image']),
                                                               'mask': x['mask']}))
    return transforms.Compose(transform_list)
       
       
def __ensure_dividable_by_4(x):
    image = x['image']
    mask = x['mask']
    sx, sy = image.shape[-2:]
    base = 4
    pad_x = 0 if sx % base == 0 else base - (sx % base) 
    pad_y = 0 if sy % base == 0 else base - (sy % base)
    
    if pad_x > 0 or pad_y > 0:
        pad = (pad_y // 2,
               pad_x // 2, 
               pad_y - pad_y // 2,
               pad_x - pad_x // 2)
        image = F.pad(image, pad, fill=0)
        mask = F.pad(mask, pad, fill=0)
    #print(image.shape)
    return {'image': image, 'mask': mask}    

class RandomCropAndPad_Sync(object):
    def __init__(self, crop_size, normal=False, padding='zeros', crop=True):
        self.crop_size = crop_size
        self.normal = normal
        self.padding = padding
        self.crop = crop
    
    def __call__(self, x):
        image = x['image']
        mask = x['mask']
        # shape_in = image.shape
        
        if self.padding == 'zeros':
            pad_x = max(0, self.crop_size - image.shape[-2])
            pad_y = max(0, self.crop_size - image.shape[-1])

            if pad_x > 0 or pad_y > 0:
                pad = (pad_y // 2,
                       pad_x // 2, 
                       pad_y - pad_y // 2,
                       pad_x - pad_x // 2)
                image = F.pad(image, pad, fill=0)
                mask = F.pad(mask, pad, fill=0) 
            
        # shape_pad = image.shape
        if self.crop:
            if self.normal:
                top = (np.clip(np.random.normal(0.5, 0.25), 0, 1) * (image.shape[-2] - self.crop_size)).astype(int)
                left = (np.clip(np.random.normal(0.5, 0.25), 0, 1) * (image.shape[-1] - self.crop_size)).astype(int)
            else:
                top = random.randint(0, image.shape[-2] - self.crop_size)
                left = random.randint(0, image.shape[-1] - self.crop_size)
            image = F.crop(image, top, left, self.crop_size, self.crop_size)
            mask = F.crop(mask, top, left, self.crop_size, self.crop_size)
        
        # shape_crop = image.shape
        # print(f'in: {shape_in}, pad: {shape_pad}, crop: {shape_crop}, top: {top}, left: {left}, pad_x: {pad_x}, pad_y: {pad_y}')

        return {'image': image, 'mask': mask}
    
    
class RandomHorizontalFlip_Sync(object):
    def __init__(self):
        pass
    
    def __call__(self, x):
        image = x['image']
        mask = x['mask']
        
        if random.random() > 0.5:
            # flip horizontally
            if isinstance(image, torch.Tensor):
                image = torch.flip(image, [2])
                mask = torch.flip(mask, [2])
            elif isinstance(image, Image.Image):
                image = F.hflip(image)
                mask = F.hflip(mask)
        return {'image': image, 'mask': mask}


def __transforms2pil_resize(method):
    mapper = {transforms.InterpolationMode.BILINEAR: Image.BILINEAR,
              transforms.InterpolationMode.BICUBIC: Image.BICUBIC,
              transforms.InterpolationMode.NEAREST: Image.NEAREST,
              transforms.InterpolationMode.LANCZOS: Image.LANCZOS,}
    return mapper[method]


def __make_power_2(img, base, method=transforms.InterpolationMode.BICUBIC):
    method = __transforms2pil_resize(method)
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)


def __scale_width(img, target_size, crop_size, method=transforms.InterpolationMode.BICUBIC):
    method = __transforms2pil_resize(method)
    ow, oh = img.size
    if ow == target_size and oh >= crop_size:
        return img
    w = target_size
    h = int(max(target_size * oh / ow, crop_size))
    return img.resize((w, h), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True
