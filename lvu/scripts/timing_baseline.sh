#!/bin/bash

export CUDA_VISIBLE_DEVICES="0"
export QUICKCODEC_CORES="16"
export DEEPCODEC_DISABLED="TRUE"

for i in {1..10}; do
    echo "Run #$i"
    python -m lvu.lvu --model_type "qwen25_lvu" --video_group_size 0 --video_path "/scratch/b3schnei/movie1080p.BluRay.1hour_30min.mp4"
done
