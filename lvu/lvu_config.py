from dataclasses import dataclass

@dataclass
class LVUConfig:
    model_name_or_path: str
    model_type: str = "qwen25_vl"
    top_k_predict_type: str = "key_norms_small"
    top_k: int = None
    top_p: float = None
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
    cache_dir: str = None
    save_video_cache: bool = False
    top_k_decay_factor: float = None
    top_k_decay_type: str = None
    
    def __post_init__(self):
        # check and auto set default values
        if self.top_k_decay_type == "linear" and self.top_k_decay_factor is None:
            print(f"Warning: top_k_decay_type is set to {self.top_k_decay_type} but top_k_decay_factor is None. Setting it to 0.5.")
            self.top_k_decay_factor = 0.5
            
    
@dataclass
class LVULayerConfig:
    layer_idx: int
    total_layers: int
    lvu_config: LVUConfig
    is_last_layer: bool = False
    prune_for_next_layer: bool = False
    
    def __post_init__(self):
        self.is_last_layer = (self.layer_idx == self.total_layers - 1)
        if self.lvu_config is None:
            self.lvu_config = LVUConfig()
        if self.layer_idx is None:
            raise ValueError("layer_idx cannot be None")
        if self.is_last_layer is None:
            raise ValueError("is_last_layer cannot be None")
        if isinstance(self.lvu_config.prefill_prune_starting_layer, int) and \
            self.lvu_config.prefill_prune_starting_layer >= 0 and \
            self.layer_idx >= self.lvu_config.prefill_prune_starting_layer:
            self.prune_for_next_layer = True
        else:
            self.prune_for_next_layer = False
        
            
        