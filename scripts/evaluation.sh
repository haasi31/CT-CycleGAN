echo "Testing idt0.1_size512"
sh scripts/1D/test_volumes.sh "idt0.1_size512" "dataset_4"
sh scripts/create_dataset_for_segnet.sh "idt0.1_size512"

echo "Testing idt0.5_size512"
sh scripts/1D/test_volumes.sh "idt0.5_size512" "dataset_4"
sh scripts/create_dataset_for_segnet.sh "idt0.5_size512"

echo "Testing no_noise_idt0.1_size512"
sh scripts/1D/test_volumes.sh "no_noise_idt0.1_size512" "dataset_5_no_noise"
sh scripts/create_dataset_for_segnet.sh "no_noise_idt0.1_size512"