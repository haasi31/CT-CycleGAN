#!/bin/bash

free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv -i 0 | grep -Eo [0-9]+)

while [ $free_mem -lt 40000 ]
do
    echo "${free_mem} MB < 40000 MB | sleeping for 30 minutes... | $(date +%H:%M-%m.%d.%Y)"
    sleep 1800 # 30 minutes
    free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv -i 0 | grep -Eo [0-9]+)
done

sh scripts/pseudo3D/train.sh