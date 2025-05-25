# QuickVideo
<p align="center">
  <img src="https://github.com/TIGER-AI-Lab/QuickVideo/raw/main/assets/logo.png" alt="QuickVideo Logo" width="340"/>
</p>

<h3 align="center">
Efficient video loading and context prefill for hour-long video understanding
</h3>

<p align="center">
<em><strong>Benjamin Schneider</strong><sup>*</sup> • <strong>Dongfu Jiang</strong><sup>*</sup> • <strong>Chao Du</strong> • <strong>Tianyu Pang</strong> • <strong>Wenhu Chen</strong></em>
</p>

<p align="center">
<sub>University of Waterloo • SeaAI Lab</sub>
</p>

<p align="center">
<sub><sup>*</sup>Equal contribution</sub>
</p>

<p align="center">
| 
<a href="https://github.com/TIGER-AI-Lab/QuickVideo?tab=readme-ov-file#-quick-start"><b>Quick Start</b></a> | 
<a href="https://arxiv.org/abs/2505.16175"><b>Paper</b></a> | 
<a href="https://github.com/TIGER-AI-Lab/QuickCodec"><b>QuickCodec</b></a> |
<a href="https://github.com/TIGER-AI-Lab/QuickVideo?tab=readme-ov-file#2-run-quickvideo-recommended"><b>QuickPrefill</b></a> 
|
</p>

---

## 🎯 Overview

Long video understanding has emerged as a crucial capability for real-world applications such as meeting summarization, video surveillance, educational lecture analysis, and content moderation. However, it remains computationally prohibitive for VideoLLMs due to two critical bottlenecks:

1. **Sequential video decoding** - Converting raw bit streams to RGB frames can take up to a minute for hour-long videos
2. **Costly prefilling** - Processing millions of tokens for LLM inference results in high latency and memory usage

<p align="center">
  <img src="./assets/imgs/teaser.png" alt="QuickVideo System Overview" width="100%"/>
</p>

**QuickVideo** is a system-algorithm co-design that achieves **3.5× speedup** (from 70s to 20s for 1-hour videos) while maintaining **97% performance** with **50% less memory**.

## 🚀 Key Innovations

### 🔧 QuickDecoder
- **Parallelized CPU-based decoder** that splits videos into keyframe-aligned intervals
- **2-3× faster** than sequential processing through concurrent execution

### ⚡ QuickPrefill
- **Group-based prefilling** for memory-efficient activation handling
- **KV-cache pruning** using key norm selection (L2) to retain only essential tokens
- **50% memory reduction** while preserving 97% of original performance

### 🔄 Overlapping Pipeline
- **Concurrent CPU decoding and GPU inference** to minimize end-to-end latency
- Intelligent scheduling reduces total processing time significantly

<p align="center">
  <img src="./assets/imgs/interleaving_time.png" alt="Pipeline Optimization" width="100%"/>
</p>

## 📊 Performance Results

We evaluate both QuickCodec on video decoding efficiency (left figure) and QuickPrefill on avg QA accuracy results on 4 long video understanding benchmarks: VideoMME, LongVideoBench, LVBench, MLVU (right figure and hidden table). Results show significant speedup and memory saving while preserving 97% of the original performance.

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
<summary><b>Performance Table</b></summary>

<table style="width: 100%; border-collapse: collapse; font-family: Arial, sans-serif;">
  <colgroup>
    <col style="width: 10%;">
    <col style="width: 20%;">
    <col style="width: 8%;">
    <col style="width: 10%;">
    <col style="width: 15%;">
    <col style="width: 10%;">
    <col style="width: 10%;">
    <col style="width: 7%;">
    <col style="width: 10%;">
  </colgroup>
  <thead>
    <tr style="background-color: #f2f2f2;">
      <th style="border: 1px solid #ddd; padding: 8px; text-align: center;">Group Size</th>
      <th style="border: 1px solid #ddd; padding: 8px; text-align: center;">KV Pruning method</th>
      <th style="border: 1px solid #ddd; padding: 8px; text-align: center;">ρ</th>
      <th style="border: 1px solid #ddd; padding: 8px; text-align: center;">VideoMME</th>
      <th style="border: 1px solid #ddd; padding: 8px; text-align: center;">LongVideoBench (val)</th>
      <th style="border: 1px solid #ddd; padding: 8px; text-align: center;">LVBench</th>
      <th style="border: 1px solid #ddd; padding: 8px; text-align: center;">MLVU (dev)</th>
      <th style="border: 1px solid #ddd; padding: 8px; text-align: center;">Avg</th>
      <th style="border: 1px solid #ddd; padding: 8px; text-align: center;">Performance</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background-color: #e8f4f8;">
      <td colspan="9" style="border: 1px solid #ddd; padding: 8px; font-weight: bold; text-align: center;">64 Frames</td>
    </tr>
    <tr>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">-</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">-</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">1</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">62.41</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">59.69</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">40.09</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">63.86</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">56.51</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">100.00%</td>
    </tr>
    <tr>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">16</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">Value Norms</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">0.5</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">47.63</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">35.98</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">30.92</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">31.38</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">36.48</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">64.55%</td>
    </tr>
    <tr>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">16</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">Attention Scores</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">0.5</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">58.63</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">52.95</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">37.83</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">59.87</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">52.32</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">92.58%</td>
    </tr>
    <tr>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">16</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">Key Norms (↓)</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">0.5</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">60.56</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">56.17</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">37.70</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">62.34</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">54.19</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">95.90%</td>
    </tr>
    <tr style="background-color: #e8f4f8;">
      <td colspan="9" style="border: 1px solid #ddd; padding: 8px; font-weight: bold; text-align: center;">128 Frames</td>
    </tr>
    <tr>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">-</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">-</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">1</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">66.41</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">60.96</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">42.87</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">66.86</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">59.27</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">100.00%</td>
    </tr>
    <tr>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">16</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">Value Norms</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">0.5</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">48.56</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">37.32</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">30.73</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">38.51</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">38.78</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">65.42%</td>
    </tr>
    <tr>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">16</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">Attention Scores</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">0.5</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">60.96</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">55.20</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">39.70</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">64.36</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">55.06</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">92.89%</td>
    </tr>
    <tr>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">16</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">Key Norms (↓)</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">0.5</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">63.41</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">58.19</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">39.57</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">64.99</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">56.54</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">95.39%</td>
    </tr>
    <tr style="background-color: #e8f4f8;">
      <td colspan="9" style="border: 1px solid #ddd; padding: 8px; font-weight: bold; text-align: center;">256 Frames</td>
    </tr>
    <tr>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">-</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">-</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">1</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">65.78</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">61.56</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">43.90</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">68.65</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">59.97</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">100.00%</td>
    </tr>
    <tr>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">16</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">Value Norms</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">0.5</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">48.33</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">38.89</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">31.38</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">37.74</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">39.08</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">65.17%</td>
    </tr>
    <tr>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">16</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">Attention Scores</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">0.5</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">62.52</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">57.22</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">41.96</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">67.27</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">57.24</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">95.45%</td>
    </tr>
    <tr>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">16</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">Key Norms (↓)</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">0.5</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">64.04</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">60.21</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">41.90</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">66.73</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">58.22</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">97.08%</td>
    </tr>
    <tr style="background-color: #e8f4f8;">
      <td colspan="9" style="border: 1px solid #ddd; padding: 8px; font-weight: bold; text-align: center;">1024 Frames</td>
    </tr>
    <tr>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">-</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">-</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">1</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">62.00</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">60.43</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">42.29</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">63.48</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">57.05</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">100.00%</td>
    </tr>
    <tr>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">16</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">Value Norms</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">0.5</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">47.37</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">33.66</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">29.18</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">32.65</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">35.71</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">62.60%</td>
    </tr>
    <tr>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">16</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">Attention Scores</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">0.5</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">62.22</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">58.49</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">42.03</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">64.45</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">56.80</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">99.56%</td>
    </tr>
    <tr>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">16</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">Key Norms</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">0.5</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">59.99</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">61.59</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">40.80</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">64.76</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">56.78</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">99.53%</td>
    </tr>
  </tbody>
</table>


</details>

## 🛠️ Installation

```bash
# Clone and setup environment
uv sync
source .venv/bin/activate
uv pip install -e .
uv pip install flash-attn --no-build-isolation
```

## 🎮 Quick Start

### 1. Download Example Video
```bash
wget https://github.com/TIGER-AI-Lab/QuickVideo/raw/refs/heads/dev/video/Q8AZ16uBhr8_resized_fps2_mute.mp4
video_path="Q8AZ16uBhr8_resized_fps2_mute.mp4"
```

### 2. Run QuickVideo (Recommended)
**With interleaved processing + KV pruning** - ⚡ **Fastest configuration**

```python
from lvu import LVU, LVUConfig

# Configure QuickVideo with all optimizations
config = LVUConfig(
    model_name_or_path="Qwen/Qwen2.5-VL-7B-Instruct",
    model_type="qwen25_lvu_interleaved",  # Enable interleaved processing
    top_k_predict_type="key_norms_small",  # Use key norm pruning
    video_group_size=16,     # Process 16 frames per group
    top_k=64,               # Keep 64 most important tokens per group
    num_frames=1024,        # Process up to 1024 frames
    use_tqdm=True,
)

lvu = LVU(config)
question = "Describe this video."
video_path = "Q8AZ16uBhr8_resized_fps2_mute.mp4"

# Generate response
output = lvu.generate(question, video_path, max_new_tokens=128, do_sample=False)
print(output)
```

**Expected Output:**
```
⏱️  Performance Metrics:
• Frame fetching: 0.33s
• Processing: 10.44s  
• Prefill: 22.95s
• End-to-end: 27.65s (vs 57.86s baseline)
• Time saved: 10.57s ⚡

🎬 Generated Response:
['The video is a compilation of classic animated shorts featuring iconic characters from the 1940s and 1950s, showcasing slapstick humor and vibrant animation styles typical of that era. The clips include:\n\n1. **"A Bug\'s Life"**: A rabbit character is seen in a desert setting, engaging in a comedic chase sequence with a carrot. The rabbit exhibits exaggerated expressions and movements, typical of the cartoon\'s slapstick style.\n\n2. **"The Wabbit Who Could"**: Bugs Bunny appears in a whimsical scene where he is performing a magic trick involving a carrot. The animation is colorful and lively']
"The video is a compilation of classic animated shorts featuring iconic 
characters from the 1940s and 1950s, showcasing slapstick humor and 
vibrant animation styles typical of that era..."
```

### 3. Baseline Comparison
**Without interleaved processing** - 🐌 **Slower but still optimized**

```python
config = LVUConfig(
    model_name_or_path="Qwen/Qwen2.5-VL-7B-Instruct",
    model_type="qwen25_lvu",  # Standard processing
    video_group_size=16,
    top_k=64,
    num_frames=1024,
    use_tqdm=True,
)
# Same usage as above - notice the 2x slower processing time
```

## 🔬 Benchmark Evaluation

Evaluate QuickVideo performance on standard video understanding benchmarks:

```bash
# Setup evaluation environment
git submodule update --init --recursive
cd lmms-eval
uv pip install -e .

# Configure environment
export DEEPCODEC_CORES=8
export FORCE_QWENVL_VIDEO_READER='deepcodec'
```

**Run comprehensive evaluation:**

```bash
# Example evaluation script
num_frame=1024
benchmark_name="videomme,longvideobench_val_v,lvbench,mlvu_dev"

accelerate launch --num_processes 8 --main_process_port 12351 -m lmms_eval \
    --model qwen2_5_vl \
    --model_args "pretrained=Qwen/Qwen2.5-VL-7B-Instruct,max_num_frames=$num_frame,use_flash_attention_2=True,adaptive_local_attention=True,local_attention_group_size=16,top_k=64,predict_type=key_norms_small" \
    --tasks $benchmark_name \
    --batch_size 1 \
    --log_samples \
    --output_path ./logs/quickvideo_evaluation
```

## 🧪 Advanced Configuration

<details>
<summary><b>Configuration Parameters</b></summary>

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `model_type` | Processing mode | `qwen25_lvu` | `qwen25_lvu`, `qwen25_lvu_interleaved` |
| `video_group_size` | Frames per processing group | `16` | `8`, `16`, `32`, ... |
| `top_k` | Tokens to keep per group | `64` | Any positive integer |
| `top_k_predict_type` | Pruning strategy | `key_norms_small` | `key_norms_small`, `attention_scores`, `value_norms` |
| `num_frames` | Maximum frames to process | `1024` | `64`, `128`, `256`, `1024`, ... |
| `top_p` | Percentage-based pruning | `None` | `0.0` to `1.0` |

</details>

## 🤝 Contributing

We welcome contributions! To add new models or KV pruning methods:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/new-model`
3. **Implement your changes** following our coding standards
4. **Add tests** and documentation
5. **Submit a pull request**

See our [contribution guidelines](CONTRIBUTING.md) for detailed instructions. (under construction)

## 📜 Citation

If you find QuickVideo useful in your research, please cite our paper:

```bibtex
@inproceedings{Schneider2025QuickVideoRL,
  title={QuickVideo: Real-Time Long Video Understanding with System Algorithm Co-Design},
  author={Benjamin Schneider and Dongfu Jiang and Chao Du and Tianyu Pang and Wenhu Chen},
  year={2025},
  url={https://api.semanticscholar.org/CorpusID:278789043}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=TIGER-AI-Lab/QuickVideo&type=Date)](https://www.star-history.com/#TIGER-AI-Lab/QuickVideo&Date)

---

<p align="center">
Made with ❤️ by the <a href="https://github.com/TIGER-AI-Lab">TIGER AI Lab</a> team
</p>
