import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import nibabel as nib
import numpy as np
import torch
import torchvision.transforms.functional as F
import tqdm
from pathlib import Path
import sys

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')


if __name__ == '__main__':
    sys.stdout = open('/dev/null', 'w') # silences print output
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    # create a website
    out_dir = opt.results_dir #os.path.join(opt.results_dir, opt.name) #, '{}_{}'.format(opt.phase, opt.epoch)) #, Path(opt.vol_A_path).stem.split('.')[0])  # define the website directory
    os.makedirs(f'{out_dir}/images', exist_ok=True)
    os.makedirs(f'{out_dir}/diff', exist_ok=True)

    if opt.eval:
        model.eval()
     
    sys.stdout = sys.__stdout__ # reactivates print output
    if opt.save_nifti:
        fake_B, real_A = [], []
        # fake_A = []
    for i, data in tqdm.tqdm(enumerate(dataset), total=len(dataset), leave=False, position=1, desc=opt.vol_A_path):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        
        if opt.save_nifti:
            if opt.pseudo3d:
                middle_slice = opt.input_nc // 2 # model.fake_B.shape[1] // 2  # visuals['fake_B'].shape[1] // 2
                fake_B.append(model.fake_B[:, middle_slice, :, :].cpu().numpy().squeeze())
                real_A.append(model.real_A[:, middle_slice, :, :].cpu().numpy().squeeze())
                # fake_A.append(model.fake_A[:, middle_slice, :, :].cpu().numpy().squeeze())
            else: 
                fake_B.append(model.fake_B.cpu().numpy().squeeze())
                real_A.append(model.real_A.cpu().numpy().squeeze())
                # fake_A.append(model.fake_A.cpu().numpy().squeeze())
        
        if opt.save_slices:
            img_path = model.get_image_paths()     # get image paths
            visuals.pop('real_B')
            visuals.pop('fake_A')
            visuals.pop('rec_B')
            visuals.pop('real_B_mask')
            visuals.pop('diff_BA')
            #save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)
    #webpage.save()  # save the HTML
    
    if opt.save_nifti:
        fake_B = ((np.array(fake_B).transpose(1, 2, 0) / 2. + 0.5) * 1600 - 1000).astype(np.int16)
        real_A = ((np.array(real_A).transpose(1, 2, 0) / 2. + 0.5) * 1600 - 1000).astype(np.int16)
        diff_AB = fake_B - real_A
        
        orig_shape = nib.load(opt.vol_A_path).shape[-3:-1]
        inference_shape = fake_B.shape[-3:-1]
        
        pad_x, pad_y = inference_shape[0] - orig_shape[0], inference_shape[1] - orig_shape[1]
        if pad_x > 0 or pad_y > 0:
            x_min = pad_x // 2
            x_max = fake_B.shape[0] - (pad_x - (pad_x // 2))
            y_min = pad_y // 2
            y_max = fake_B.shape[1] - (pad_y - (pad_y // 2))
            
            fake_B = fake_B[x_min:x_max, y_min:y_max,:]
            diff_AB = diff_AB[x_min:x_max, y_min:y_max,:]
            #fake_B = F.crop(fake_B, pad_x // 2, pad_y // 2, orig_shape[0], orig_shape[1])
            #diff_AB = F.crop(diff_AB, pad_x // 2, pad_y // 2, orig_shape[0], orig_shape[1])
        

        # fake_A = ((np.array(fake_A).transpose(1, 2, 0) / 2. + 0.5) * 1600 - 1000).astype(np.int16)
        # nifti_fake_A = nib.Nifti1Image(fake_A, nib.load(opt.vol_A_path).affine)
        # nib.save(nifti_fake_A, os.path.join(web_dir, f'fake_A_{Path(opt.vol_A_path).stem.split(".")[0]}.nii.gz'))
        
        nifti_fake_B = nib.Nifti1Image(fake_B, nib.load(opt.vol_A_path).affine)
        nifti_diff_AB = nib.Nifti1Image(diff_AB, nib.load(opt.vol_A_path).affine)
        nib.save(nifti_fake_B, os.path.join(out_dir, 'images', f'{Path(opt.vol_A_path).stem.split(".")[0]}.nii.gz'))
        nib.save(nifti_diff_AB, os.path.join(out_dir, 'diff', f'{Path(opt.vol_A_path).stem.split(".")[0]}.nii.gz'))