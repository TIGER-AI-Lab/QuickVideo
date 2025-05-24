# QuickVideo
<p align="center">
  <img src="https://github.com/TIGER-AI-Lab/QuickVideo/raw/main/assets/logo.png" alt="QuickVideo Logo" width="340"/>
</p>

<h3 align="center">
Efficient video loading and context prefill for hour-long video understanding
</h3>

<p align="center">
| 
<a href="https://github.com/TIGER-AI-Lab/QuickVideo?tab=readme-ov-file##installation"><b>Documentation</b></a> | 
<a href="https://arxiv.org/abs/2505.16175"><b>Paper</b></a> | 
<a href="https://github.com/TIGER-AI-Lab/QuickCodec"><b>QuickCodec</b></a> |
<a href="https://github.com/TIGER-AI-Lab/QuickVideo?tab=readme-ov-file##usage"><b>QuickPrefill</b></a> 
|
</p>

## Introduction
Long video understanding has emerged as a crucial capability in real-world applications such as meeting summarization, video surveillance, educational lecture analysis, and content moderation. However, it remains computationally prohibitive for VideoLLMs, primarily due to two bottlenecks:
1. **sequential video decoding**, the process of converting the raw bit stream to RGB frames can take up to a minute for hour-long video inputs
2. **costly prefilling of up to several million tokens for LLM inference**, resulting in high latency and memory use. 

![./assets/imgs/teaser.png](./assets/imgs/teaser.png)

To address these challenges, we propose **QuickVideo**, a system-algorithm co-design that substantially accelerates long video understanding to support real-time downstream applications. It comprises three key innovations: 

- **QuickDecoder**, a parallelized CPU-based video decoder that achieves 2–3 times
 speedup by splitting videos into keyframe-aligned intervals processed concurrently. 
- **QuickPrefill**, a memory-efficient prefilling method using KV-cache pruning to support more frames with less GPU memory; 
- **Overlapping scheme that overlaps CPU video decoding with GPU inference**. This brings the end-to-end latency down from 70 seconds to 20 seconds for a 1-hour video, achieving a 3.5x speedup (see following figure).


![./assets/imgs/interleaving_time.png](./assets/imgs/interleaving_time.png)

We evaluate both QuickCodec on video decoding efficiency (left figure) and QuickPrefill with 50% of the KV-cache tokens pruned (right figure and hidden table). Results show significant speedup and memory saving while preserving 97% of the original performance on 4 long video understanding benchmarks: VideoMME, LongVideoBench, LVBench, and MLVU.

<table>
  <tr>
    <td width="34%">
      <img src="./assets/imgs/video_processing_times.png" alt="Video Processing Times" width="100%">
    </td>
    <td width="66%">
      <img src="./assets/imgs/kv_pruning_avg_performance.png" alt="KV Pruning Average Performance" width="100%">
    </td>
  </tr>
</table>

<details>
<summary>Click to see the performance of different pruning methods</summary>

| Group  Size |      KV Pruning method     | \rho | VideoMME | LongVideoBench  (val) | LVBench | MLVU  (dev) |  Avg  | Performance |
|:-----------:|:--------------------------:|:----:|:--------:|:---------------------:|:-------:|:-----------:|:-----:|:-----------:|
|  64 Frames  |                            |      |          |                       |         |             |       |             |
|      -      |              -             |   1  |   62.41  |         59.69         |  40.09  |    63.86    | 56.51 |   100.00%   |
|      16     |         Value Norms        |  0.5 |   47.63  |         35.98         |  30.92  |    31.38    | 36.48 |    64.55%   |
|      16     |      Attention Scores      |  0.5 |   58.63  |         52.95         |  37.83  |    59.87    | 52.32 |    92.58%   |
|      16     | Key Norms  (Quick Prefill) |  0.5 |   60.56  |         56.17         |  37.70  |    62.34    | 54.19 |    95.90%   |
|  128 Frames |                            |      |          |                       |         |             |       |             |
|      -      |              -             |   1  |   66.41  |         60.96         |  42.87  |    66.86    | 59.27 |   100.00%   |
|      16     |         Value Norms        |  0.5 |   48.56  |         37.32         |  30.73  |    38.51    | 38.78 |    65.42%   |
|      16     |      Attention Scores      |  0.5 |   60.96  |         55.20         |  39.70  |    64.36    | 55.06 |    92.89%   |
|      16     |  Key Norms (Quick Prefill) |  0.5 |   63.41  |         58.19         |  39.57  |    64.99    | 56.54 |    95.39%   |
|  256 Frames |                            |      |          |                       |         |             |       |             |
|      -      |              -             |   1  |   65.78  |         61.56         |  43.90  |    68.65    | 59.97 |   100.00%   |
|      16     |         Value Norms        |  0.5 |   48.33  |         38.89         |  31.38  |    37.74    | 39.08 |    65.17%   |
|      16     |      Attention Scores      |  0.5 |   62.52  |         57.22         |  41.96  |    67.27    | 57.24 |    95.45%   |
|      16     |  Key Norms (Quick Prefill) |  0.5 |   64.04  |         60.21         |  41.90  |    66.73    | 58.22 |    97.08%   |
| 1024 Frames |                            |      |          |                       |         |             |       |             |
|      -      |              -             |   1  |   62.00  |         60.43         |  42.29  |    63.48    | 57.05 |   100.00%   |
|      16     |         Value Norms        |  0.5 |   47.37  |         33.66         |  29.18  |    32.65    | 35.71 |    62.60%   |
|      16     |      Attention Scores      |  0.5 |   62.22  |         58.49         |  42.03  |    64.45    | 56.80 |    99.56%   |
|      16     |          Key Norms         |  0.5 |   59.99  |         61.59         |  40.80  |    64.76    | 56.78 |    99.53%   |

</details>

## Installation
```bash
uv sync
source .venv/bin/activate
uv pip install -e .
uv pip install flash-attn --no-build-isolation
```

## Usage

1. Download the example video path
```bash
wget https://github.com/SCZwangxiao/video-FlexReduc/raw/refs/heads/main/misc/Q8AZ16uBhr8_resized_fps2_mute.mp4
video_path="Q8AZ16uBhr8_resized_fps2_mute.mp4"
#!/bin/bash
```

2. Run the QuickVideo with interleaved video decoding and group-based context prefilling with kv cache pruning.
```python
from lvu import LVU, LVUConfig
model_name_or_path = "Qwen/Qwen2.5-VL-7B-Instruct"
model_type = "qwen25_lvu_interleaved"
config = LVUConfig(
    model_name_or_path=model_name_or_path,
    model_type=model_type,
    top_k_predict_type="key_norms_small",
    video_group_size=16, # 16 frames per group
    top_k=64, # keep 64 tokens for each group
    top_p=None, # None or a float from 0 to 1, means pruning top_p * 100% of the tokens
    prefill_prune_starting_layer=None, # None or 0 means prune all layers
    num_frames=1024, # 1024 frames in total
    fps=None, 
    use_tqdm=True,
)
lvu = LVU(config)

question = "Describe this video."
video_path = "Q8AZ16uBhr8_resized_fps2_mute.mp4"
generation_kwargs = {
    "max_new_tokens": 128,
    "do_sample": False,
    "top_p": 1.0,
}
output = lvu.generate(question, video_path, **generation_kwargs)
print(output)
"""
# Example output
total time spent fetching frames was: 0.32964205741882324
total time spent on processor was: 10.441518306732178
total time spent on prefill was: 22.954123497009277
total time spent on e2e fetching and decoding was: 27.645442724227905
total time spent on decoding was: 4.490285396575928
Time saved by interleaved processing was: 10.5701265335083
['The video is a compilation of classic animated shorts featuring iconic characters from the 1940s and 1950s, showcasing slapstick humor and vibrant animation styles typical of that era. The clips include:\n\n1. **"A Bug\'s Life"**: A rabbit character is seen in a desert setting, engaging in a comedic chase sequence with a carrot. The rabbit exhibits exaggerated expressions and movements, typical of the cartoon\'s slapstick style.\n\n2. **"The Wabbit Who Could"**: Bugs Bunny appears in a whimsical scene where he is performing a magic trick involving a carrot. The animation is colorful and lively']
"""
```

3. Run the QuickVideo with non-interleaved video decoding and group-based context prefilling without kv cache pruning.
```python
from lvu import LVU, LVUConfig
model_name_or_path = "Qwen/Qwen2.5-VL-7B-Instruct"
model_type = "qwen25_lvu"
config = LVUConfig(
    model_name_or_path=model_name_or_path,
    model_type=model_type,
    top_k_predict_type="key_norms_small",
    video_group_size=16, # 16 frames per group
    top_k=64, # keep 64 tokens for each group
    top_p=None, # None or a float from 0 to 1, means pruning top_p * 100% of the tokens
    prefill_prune_starting_layer=None, # None or 0 means prune all layers
    num_frames=1024, # 1024 frames in total
    fps=None, 
    use_tqdm=True,
)
lvu = LVU(config)

question = "Describe this video."
video_path = "Q8AZ16uBhr8_resized_fps2_mute.mp4"
generation_kwargs = {
    "max_new_tokens": 128,
    "do_sample": False,
    "top_p": 1.0,
}
output = lvu.generate(question, video_path, **generation_kwargs)
print(output)
"""
# Example output
total time spent fetching frames was: 16.419464826583862
total time spent on processor was: 15.166603088378906
total time spent on prefill was: 21.452709436416626
total time spent on e2e fetching and decoding was: 57.85865354537964
total time spent on decoding was: 4.586025714874268
Time saved by interleaved processing was: -0.23385047912597656
['The video is a compilation of classic animated shorts featuring iconic characters from the 1940s and 1950s, showcasing slapstick humor and vibrant animation styles typical of that era. The clips include:\n\n1. **"A Bug\'s Life"**: A rabbit character is seen in a desert setting, engaging in a comedic chase sequence with a carrot. The rabbit exhibits exaggerated expressions and movements, typical of the cartoon\'s slapstick style.\n\n2. **"The Wabbit Who Could"**: Bugs Bunny appears in a whimsical scene where he is performing a magic trick involving a carrot. The animation is colorful and lively']
"""
```
**Clearly, the non-interleaved processing takes about 58 seconds, which is 2 times longer than the interleaved processing.**

## Video Understand Benchmark evaluation
We use lmms-eval to evaluate the performance of QuickVideo on the Video Understand Benchmark.
```bash
git submodule update --init --recursive
cd lmms-eval
uv pip install -e .
```

**Evaluation example**
```bash
export DEEPCODEC_CORES=8
export FORCE_QWENVL_VIDEO_READER='deepcodec'
adaptive_local_attention=True
num_processes=8
benchmark_name=lvbench,videomme,lvbench,mlvu_dev
# we recommend to select one combination of the following parameters to run the evaluation instead of all combinations if you are just testing.
for num_frame in 64 128 256 1024; do
    for local_attention_group_size in 16; do
        for top_k in 720 ; do
            for predict_type in key_norms_small vector_norms query_attention_weights; do 
                for top_k_starting_layer in 0; do
                    for prune_during_prefill_layer_idx in -1; do
                        echo "num_frame: $num_frame, local_attention_group_size: $local_attention_group_size, top_k: $top_k, predict_type: $predict_type, top_k_starting_layer: $top_k_starting_layer, prune_during_prefill_layer_idx: $prune_during_prefill_layer_idx"
                        accelerate launch --num_processes ${num_processes} --main_process_port 12351 -m lmms_eval \
                            --model qwen2_5_vl \
                            --model_args "pretrained="Qwen/Qwen2.5-VL-7B-Instruct",max_num_frames=$num_frame,use_flash_attention_2=True,adaptive_local_attention=$adaptive_local_attention,local_attention_group_size=${local_attention_group_size},top_k=$top_k,predict_type=$predict_type,top_k_starting_layer=$top_k_starting_layer,prune_during_prefill_layer_idx=$prune_during_prefill_layer_idx" \
                            --tasks $benchmark_name \
                            --batch_size 1 \
                            --log_samples \
                            --log_samples_suffix "Qwen2.5-VL-7B-Instruct-frames-$num_frame-local_attention_group_size-$local_attention_group_size-top_k-$top_k-predict_type-$predict_type-top_k_starting_layer-$top_k_starting_layer-prune_during_prefill_layer_idx-$prune_during_prefill_layer_idx" \
                            --output_path ./logs/qwen2_5_vl_$benchmark_name_$num_frame
                    done
                done
            done
        done
    done
done
```

(Note: for LVBench you might need to unzip the videos by your own first after running it for one time.)

## Citation
If you find this repository useful, please consider citing our paper:
```bibtex
@inproceedings{Schneider2025QuickVideoRL,
  title={QuickVideo: Real-Time Long Video Understanding with System Algorithm Co-Design},
  author={Benjamin Schneider and Dongfu Jiang and Chao Du and Tianyu Pang and Wenhu Chen},
  year={2025},
  url={https://api.semanticscholar.org/CorpusID:278789043}
}
```