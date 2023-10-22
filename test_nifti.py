import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import nibabel as nib
import numpy as np
import torch
import tqdm
from pathlib import Path

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')


if __name__ == '__main__':
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

    # initialize logger
    if opt.use_wandb:
        wandb_run = wandb.init(project=opt.wandb_project_name, name=opt.name, config=opt) if not wandb.run else wandb.run
        wandb_run._label(repo='CycleGAN-and-pix2pix')

    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch), Path(opt.vol_A_path).stem.split('.')[0])  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    if opt.eval:
        model.eval()
     
    fake_B, real_A = [], []
    for i, data in tqdm.tqdm(enumerate(dataset), total=len(dataset)):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        fake_B.append(visuals['fake_B'].cpu().numpy().squeeze())
        real_A.append(visuals['real_A'].cpu().numpy().squeeze())
        
        img_path = model.get_image_paths()     # get image paths
        visuals.pop('real_B')
        visuals.pop('fake_A')
        visuals.pop('rec_B')
        visuals.pop('real_B_mask')
        visuals.pop('diff_BA')
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)
    webpage.save()  # save the HTML
    
    fake_B = ((np.array(fake_B).transpose(1, 2, 0) / 2. + 0.5) * 1600 - 1000).astype(np.int16)
    real_A = ((np.array(real_A).transpose(1, 2, 0) / 2. + 0.5) * 1600 - 1000).astype(np.int16)
    diff_AB = np.abs(fake_B - real_A)
    
    #nifti_fake_A = nib.Nifti1Image(fake_A, nib.load(opt.vol_B_path).affine)
    nifti_fake_B = nib.Nifti1Image(fake_B, nib.load(opt.vol_A_path).affine)
    nifti_diff_AB = nib.Nifti1Image(diff_AB, nib.load(opt.vol_A_path).affine)
    #nib.save(nifti_fake_A, os.path.join(web_dir, f'fake_A_{Path(opt.vol_B_path).stem}.nii.gz'))
    nib.save(nifti_fake_B, os.path.join(web_dir, f'fake_B_{Path(opt.vol_A_path).stem.split(".")[0]}.nii.gz'))
    nib.save(nifti_diff_AB, os.path.join(web_dir, f'diff_AB_{Path(opt.vol_A_path).stem.split(".")[0]}.nii.gz'))