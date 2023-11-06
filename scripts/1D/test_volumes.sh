#!/bin/bash
experiment_name="$1" #no_noise_idt0.1_size512 # idt0_size256 #no_noise #syn2CT_2_size512_idt0.5_masked
dataset="$2" #dataset_5_no_noise
dir="/home/ahaas/airway-seg/vessel_graph_generation/datasets/$dataset/images" #dataset_4/images #
i=0
num_files=$(ls -1 "$dir"/* | wc -l)

for path in "$dir"/*
do 
    i=$((i+1))
    if [ $i -lt 11 ]
    then
        ATM=253
    elif [ $i -lt 21 ]
    then
        ATM=255
    else
        ATM=257
    fi
    echo "${i}/${num_files} | ATM: ${ATM} | ${path}"
    python test_nifti.py \
    --dataroot /home/ahaas/data/syn2CT_2 \
    --name "$experiment_name" \
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
    
done