set -ex
name=idt0_size256
path=/home/ahaas/airway-seg/vessel_graph_generation/datasets/dataset_4/images/20230907_195116_1fe7dfb7-1f40-4b41-97bf-b1f7b6c0b4ad_volume.nii.gz
ATM=253
python test_nifti.py \
    --dataroot /home/ahaas/data/syn2CT_2 \
    --name "$name" \
    --model cycle_gan \
    --dataset_mode nifti \
    --input_nc 1 \
    --output_nc 1 \
    --phase test \
    --no_dropout \
    --num_test 1000 \
    --preprocess none \
    --no_flip \
    --vol_A_path "$path" \
    --vol_B_path "/home/shared/Data/ATM22/train/images/ATM_${ATM}_0000.nii.gz" \
    --mask_A_path "/home/ahaas/data/ATM22_masks/ATM_${ATM}_0000_mask_lobes.nii.gz" \
    --mask_B_path "/home/ahaas/data/ATM22_masks/ATM_${ATM}_0000_mask_lobes.nii.gz" \
    --save_nifti
     #--save_slices false