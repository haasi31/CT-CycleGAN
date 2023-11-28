import glob
import subprocess
import argparse
import tqdm
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='dataset_2023_11_15')
parser.add_argument('--experiment_name', type=str, default='dataset_2023_11_15_1slices_0.1idt')
parser.add_argument('--slices', type=str, default='3')
args = parser.parse_args()

dataset_name = args.dataset_name
experiment_name = args.experiment_name
slices = args.slices

image_A_paths = sorted(glob.glob(f'/home/ahaas/data/3_deformed_data/{dataset_name}/images/*.nii.gz'))
mask_A_paths = sorted(glob.glob(f'/home/ahaas/data/3_deformed_data/{dataset_name}/masks/*.nii.gz'))

results_dir = f'/home/ahaas/data/5_out_cyclegan'

for image_A_path, mask_A_path in tqdm.tqdm(zip(image_A_paths, mask_A_paths), total=len(image_A_paths), position=0):
    ATM_scan = image_A_path.split('/')[-1].split('_')[2]
    image_B_path = f'/home/ahaas/data/3_deformed_data/ATM22/train/images/ATM_{ATM_scan}_0_volume.nii.gz'
    mask_B_path = f'/home/ahaas/data/3_deformed_data/ATM22/train/masks/ATM_{ATM_scan}_0_mask.nii.gz'
    
    fake_out_path = os.path.join(results_dir, experiment_name, 'test_latest', image_A_path.split('/')[-1].split('.')[0], f'fake_B_{image_A_path.split("/")[-1]}')
    if os.path.exists(fake_out_path):
        print(f'{image_A_path} already processed, skipping')
        continue
    
    subprocess.run([
        'python', '/home/ahaas/pytorch-CycleGAN-and-pix2pix/test_nifti.py',
        '--dataroot', image_A_path, 
        '--name', experiment_name, 
        '--model', 'cycle_gan', 
        '--dataset_mode', 'nifti', 
        '--input_nc', slices,
        '--output_nc', slices,
        '--phase', 'test',
        '--no_dropout',
        '--num_test', '1000',
        '--preprocess', 'none', 
        '--no_flip',
        '--vol_A_path', image_A_path,
        '--mask_A_path', mask_A_path,
        '--vol_B_path', image_B_path,
        '--mask_B_path', mask_B_path,
        '--save_nifti',
        '--pseudo3d',
        '--results_dir', results_dir
    ])