#!/bin/bash
dir=/home/ahaas/airway-seg/vessel_graph_generation/datasets/dataset_4/images
i=0
num_files=$(ls -1 "$dir"/* | wc -l)

for path in "$dir"/*
do 
    i=$((i+1))
    echo "${i}/${num_files} - ${path}"
    if [ $i -lt 10 ]
    then
        ATM=253
    elif [ $i -lt 20 ]
    then
        ATM=255
    else
        ATM=257
    fi
    python test_nifti.py \
    --dataroot /home/ahaas/data/syn2CT_2 \
    --name syn2CT_2_masked_merge_idt0.1 \
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
    --mask_B_path "/home/ahaas/data/ATM22_masks/ATM_${ATM}_0000_mask_lobes.nii.gz"
    
done