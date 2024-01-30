#!/bin/bash
name="$1" #pseudo3d_slices5_idt0.1
dataset="$2"
dir="/home/ahaas/airway-seg/vessel_graph_generation/datasets/$dataset/images" #dataset_5_no_noise/images
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
    python test.py \
    --dataroot /home/ahaas/data/syn2CT_2 \
    --name "$name" \
    --model cycle_gan \
    --dataset_mode nifti \
    --input_nc 5 \
    --output_nc 5 \
    --phase test \
    --no_dropout \
    --num_test 1000 \
    --preprocess none \
    --no_flip \
    --vol_A_path "$path" \
    --mask_A_path "/home/ahaas/data/0_input_simulation/ATM_preparation/masks/ATM_${ATM}_0000_mask_lobes.nii.gz" \
    --vol_B_path  "/home/ahaas/data/3_deformed_data/ATM22/train/images/ATM_001_0_volume.nii.gz" \
    --mask_B_path "/home/ahaas/data/0_input_simulation/ATM_preparation/masks/ATM_${ATM}_0000_mask_lobes.nii.gz" \
    --save_nifti \
    --pseudo3d
    #--save_slices false
    # "/home/shared/Data/ATM22/train/images/ATM_${ATM}_0000.nii.gz" \
done