#!/bin/bash

free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv -i 0 | grep -Eo [0-9]+)

while [ $free_mem -lt 30000 ] # free memory is less than 30 GB
do
    # echo "Waiting for memory to be freed up... free memory = ${free_mem} MB"
    echo "${free_mem} MB < 30000 MB | sleeping for 30 minutes... | $(date +%H:%M-%m.%d.%Y)"
    sleep 1800 # 30 minutes
    free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv -i 0 | grep -Eo [0-9]+)
done

sh scripts/train_syn2CT.sh