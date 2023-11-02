set -ex
python train.py \
    --dataroot /home/ahaas/data/syn2CT/no_noise \
    --name no_noise_idt0.1_size512 \
    --model cycle_gan \
    --dataset_mode tiff \
    --input_nc 1 \
    --output_nc 1 \
    --preprocess none \
    --lambda_idt_AB 0.1 \
    --display_freq 200 \
    --display_ncols 7 \
    --batch_size 4