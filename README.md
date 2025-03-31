# LVU

## Installation

```bash
uv sync
uv pip install flash-attn --no-build-isolation
```

## Usage

```bash
python -m lvu.lvu
```

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
video_path = "/home/dongfu/data/.cache/huggingface/videomme/data/ZXoaMa6jlO4.mp4"
generation_kwargs = {
    "max_new_tokens": 128,
    "do_sample": False,
    "top_p": 1.0,
}
output = lvu.generate(question, video_path, **generation_kwargs)
print(output)
```
