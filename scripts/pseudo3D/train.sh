set -ex
python train.py \
    --dataroot /home/ahaas/data/syn2CT/noise_5slices \
    --name pseudo3d_slices5_idt0.5 \
    --model cycle_gan \
    --dataset_mode pseudo3d \
    --input_nc 5 \
    --output_nc 5 \
    --preprocess none \
    --lambda_idt_AB 0.5 \
    --display_freq 200 \
    --display_ncols 7 \
    --batch_size 4 \
    --no_html