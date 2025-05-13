#!/bin/bash

export CUDA_VISIBLE_DEVICES="0"
export DEEPCODEC_CORES="8"
export DEEPCODEC_DISABLED="TRUE"

for i in {1..10}; do
    echo "Run #$i"
    python -m lvu.lvu "qwen25_lvu" "0" "/scratch/b3schnei/movie1080p.BluRay.1hour_30min.mp4"
done
