"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
import os
from data.base_dataset import BaseDataset, get_transform, get_mask_transform, get_transform_sync
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import random
import nibabel as nib


class NiftiDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        #parser.add_argument('--new_dataset_option', type=float, default=1.0, help='new dataset option')
        #parser.set_defaults(max_dataset_size=10, new_dataset_option=2.0)  # specify dataset-specific default values
        parser.add_argument('--vol_A_path', type=str, default='/home/ahaas/airway-seg/vessel_graph_generation/datasets/dataset_4/images/20230907_195116_1fe7dfb7-1f40-4b41-97bf-b1f7b6c0b4ad_volume.nii.gz', help='path to volume A')
        parser.add_argument('--vol_B_path', type=str, default='/home/shared/Data/ATM22/train/images/ATM_253_0000.nii.gz', help='path to volume B')
        parser.add_argument('--mask_A_path', type=str, default='/home/ahaas/data/ATM22_masks/ATM_253_0000_mask_lobes.nii.gz', help='path to mask A')
        parser.add_argument('--mask_B_path', type=str, default='/home/ahaas/data/ATM22_masks/ATM_253_0000_mask_lobes.nii.gz', help='path to mask B')
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        
        self.vol_A = nib.load(opt.vol_A_path).get_fdata()
        self.vol_B = nib.load(opt.vol_B_path).get_fdata()
        self.mask_A = nib.load(opt.mask_A_path).get_fdata()
        self.mask_B = nib.load(opt.mask_B_path).get_fdata()
        
        self.A_size = self.vol_A.shape[-1] # get the size of dataset A
        self.B_size = self.vol_B.shape[-1]  # get the size of dataset B     
        
        self.transform_A = get_transform_sync(self.opt, ct_domain=True)
        self.transform_B = get_transform_sync(self.opt, ct_domain=True)
        
        

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """
        
        index_A = index % self.A_size
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)

        A_data = {'image': Image.fromarray(self.vol_A[:, :, index_A]), 
                  'mask': Image.fromarray(((self.mask_A[:, :, index_A] >= 1)*255).astype(np.uint8), mode='L')}
        B_data = {'image': Image.fromarray(self.vol_B[:, :, index_B]), 
                  'mask': Image.fromarray(((self.mask_B[:, :, index_B] >= 1)*255).astype(np.uint8), mode='L')}
        A_data = self.transform_A(A_data)
        B_data = self.transform_B(B_data)
        A, A_mask = A_data['image'], A_data['mask']
        B, B_mask = B_data['image'], B_data['mask']
        
        return {'A': A, 'B': B, 'A_paths': f'{index_A:04d}.tiff', 'B_paths': f'{index_B:04d}.tiff', 'A_mask': A_mask, 'B_mask': B_mask}

    def __len__(self):
        """Return the total number of images."""
        return max(self.A_size, self.B_size)
