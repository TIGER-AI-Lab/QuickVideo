# LVU

## Installation

```bash
uv sync
uv pip install -e .
uv pip install flash-attn --no-build-isolation
```

## Usage

```bash
wget https://github.com/SCZwangxiao/video-FlexReduc/raw/refs/heads/main/misc/Q8AZ16uBhr8_resized_fps2_mute.mp4
video_path="Q8AZ16uBhr8_resized_fps2_mute.mp4"
python -m lvu.lvu --model_type "qwen25_lvu" --video_group_size 32 --video_path $video_path
python -m lvu.lvu --model_type "qwen25_lvu_interleaved" --video_group_size 32 --video_path $video_path
```

## Example

```python
from lvu import LVU, LVUConfig
config = LVUConfig(
    model_name_or_path="Qwen/Qwen2.5-VL-7B-Instruct", 
    model_type="qwen25_vl",
    top_k_predict_type="key_norms_small",
    video_group_size=16,
    top_k=64,
    prefill_prune_starting_layer=None,
    adaptive_local_attention=True,
    num_frames=1024,
    use_tqdm=True,
)
lvu = LVU(config)

question = "Describe this video."
video_path = ""Q8AZ16uBhr8_resized_fps2_mute.mp4""
generation_kwargs = {
    "max_new_tokens": 128,
    "do_sample": False,
    "top_p": 1.0,
}
output = lvu.generate(question, video_path, **generation_kwargs)
print(output)
```

## Evaluation
```bash
git submodule update --init --recursive
cd lmms-eval
uv pip install -e .
```
