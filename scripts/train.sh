experiment_name=dataset_2023_11_15
slices=1
lambda=2

set -ex

python train.py \
    --dataroot_A /home/ahaas/data/4_input_cyclegan/"$experiment_name"/"$slices"_slices\
    --dataroot_B /home/ahaas/data/4_input_cyclegan/ATM22/"$slices"_slices\
    --name "$experiment_name"_"$slices"slices_"$lambda"idt \
    --model cycle_gan \
    --dataset_mode pseudo3d \
    --input_nc $slices \
    --output_nc $slices \
    --preprocess crop \
    --crop_size 256 \
    --lambda_idt_AB $lambda \
    --display_freq 200 \
    --display_ncols 6 \
    --batch_size 12 \
    --no_html \
    --n_epochs 20 \
    --n_epochs_decay 20