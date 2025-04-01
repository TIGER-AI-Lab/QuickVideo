from dataclasses import dataclass

@dataclass
class LVUConfig:
    model_name_or_path: str
    model_type: str = "qwen25_vl"
    top_k_predict_type: str = "key_norms_small"
    top_k: int = None
    top_k_starting_layer: int = None
    do_top_k_for_query: bool = False
    adaptive_local_attention: bool = True
    video_group_size: int = None # per frame
    prefill_prune_starting_layer: int = None
    fps: int = None
    num_frames: int = 32
    use_tqdm: bool = False
    extra_kwargs: dict = None
    enable: bool = True
    
@dataclass
class LVULayerConfig:
    layer_idx: int
    is_last_layer: bool
    lvu_config: LVUConfig
    prune_for_next_layer: bool = False
    def __init__(self, layer_idx=None, is_last_layer=None, lvu_config=None):
        self.layer_idx = layer_idx
        self.is_last_layer = is_last_layer
        self.lvu_config = lvu_config
        if self.lvu_config is None:
            self.lvu_config = LVUConfig()
        if self.layer_idx is None:
            raise ValueError("layer_idx cannot be None")
        if self.is_last_layer is None:
            raise ValueError("is_last_layer cannot be None")
        if isinstance(self.lvu_config.prefill_prune_starting_layer, int) and \
            self.lvu_config.prefill_prune_starting_layer >= 0 and \
            self.lvu_config.layer_idx >= self.lvu_config.prefill_prune_starting_layer:
            self.prune_for_next_layer = True
        else:
            self.prune_for_next_layer = False
        
            
        