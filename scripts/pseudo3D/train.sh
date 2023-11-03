set -ex
python train.py \
    --dataroot /home/ahaas/data/syn2CT/noise_3slices \
    --name pseudo3d_slices3_idt0.1 \
    --model cycle_gan \
    --dataset_mode pseudo3d \
    --input_nc 3 \
    --output_nc 3 \
    --preprocess none \
    --lambda_idt_AB 0.1 \
    --display_freq 200 \
    --display_ncols 7 \
    --batch_size 4 \
    --no_html