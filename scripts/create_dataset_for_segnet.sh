#!/bin/bash
experiment_name="$1" #pseudo3d_slices3_idt0.5
in_dir=/home/ahaas/pytorch-CycleGAN-and-pix2pix/results/"$experiment_name"/test_latest
out_dir=/home/ahaas/data/cyclegan_out/"$experiment_name"
mkdir -p "$out_dir/images"

# images
find "$in_dir" -type f -name 'fake_B_*' -exec cp {} -t "$out_dir/images" \;
cd "$out_dir/images"; for file in fake_B_*; do mv "$file" "${file#fake_B_}"; done

# labels
cp -r /home/ahaas/airway-seg/vessel_graph_generation/datasets/dataset_4/labels "$out_dir"

# cycleGAN difference
mkdir -p "$out_dir/diff"
find "$in_dir" -type f -name 'diff_AB_*' -exec cp {} -t "$out_dir/diff" \;

# remove in_dir
rm -r /home/ahaas/pytorch-CycleGAN-and-pix2pix/results/"$experiment_name"
