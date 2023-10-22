set -ex
python train.py \
    --dataroot /home/ahaas/data/syn2CT_2 \
    --name syn2CT_2_size512_idt0.1_masked \
    --model cycle_gan \
    --dataset_mode tiff \
    --input_nc 1 \
    --output_nc 1 \
    --preprocess none \
    --lambda_idt_AB 0.1 \
    --display_freq 200 \
    --display_ncols 6 \
    --batch_size 5
    # --preprocess crop \
    # --crop_size 256 \