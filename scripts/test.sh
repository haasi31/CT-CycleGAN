dataset_name=dataset_2023_11_15
experiment_name=dataset_2023_11_15_1slices_0.5idt
image=ATM_057_17_0
python test.py \
    --dataroot "/home/ahaas/data/3_deformed_data/${dataset_name}/images/syn_ATM_057_17_0_volume.nii.gz"  \
    --name $experiment_name \
    --model cycle_gan \
    --dataset_mode nifti \
    --input_nc 1 \
    --output_nc 1 \
    --phase test \
    --no_dropout \
    --num_test 1000 \
    --preprocess none \
    --no_flip \
    --vol_A_path "/home/ahaas/data/3_deformed_data/${dataset_name}/images/syn_ATM_057_17_0_volume.nii.gz" \
    --mask_A_path "/home/ahaas/data/3_deformed_data/${dataset_name}/masks/syn_ATM_057_17_0_mask.nii.gz" \
    --vol_B_path "/home/ahaas/data/3_deformed_data/ATM22/train/images/ATM_057_0_volume.nii.gz" \
    --mask_B_path "/home/ahaas/data/3_deformed_data/ATM22/train/masks/ATM_057_0_mask.nii.gz" \
    --save_nifti \
    --pseudo3d \
    --results_dir /home/ahaas/data/5_out_cyclegan