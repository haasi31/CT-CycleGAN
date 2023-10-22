set -ex
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
    --vol_A_path /home/ahaas/airway-seg/vessel_graph_generation/datasets/dataset_4/images/20230907_201008_0f134eb7-0178-48b5-8313-7474de12e727_volume.nii.gz \
    --vol_B_path /home/shared/Data/ATM22/train/images/ATM_255_0000.nii.gz \
    --mask_A_path /home/ahaas/data/ATM22_masks/ATM_255_0000_mask_lobes.nii.gz \
    --mask_B_path /home/ahaas/data/ATM22_masks/ATM_255_0000_mask_lobes.nii.gz