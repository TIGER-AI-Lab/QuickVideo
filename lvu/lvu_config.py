from dataclasses import dataclass

@dataclass
class LVUConfig:
    model_name_or_path: str
    model_type: str = "qwen25_vl"
    top_k_predict_type: str = "key_norms_small"
    top_k: int = None
    top_k_starting_layer: int = None
    adaptive_local_attention: bool = True
    video_group_size: int = None # per frame
    prefill_prune_starting_layer: int = None
    layer_idx: int = None
    is_last_layer: bool = False
    fps: int = None
    num_frames: int = 32
    use_tqdm: bool = False