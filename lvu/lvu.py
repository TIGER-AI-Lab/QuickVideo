import torch
from dataclasses import dataclass
from transformers import AutoProcessor, AutoModelForImageTextToText
from .models import lvu_init_model_map, lvu_run_model_map

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

class LVU:
    def __init__(self, config, model=None, processor=None, model_init_kwargs={}):
        self.config = config
        if model is None:
            model_init_kwargs = {
                "torch_dtype": torch.bfloat16,
                "device_map": "auto",
                "attn_implementation": "flash_attention_2",
            }
            model = AutoModelForImageTextToText.from_pretrained(config.model_name_or_path, **model_init_kwargs)
        if processor is None:
            processor = AutoProcessor.from_pretrained(config.model_name_or_path)
        
        self.model = model
        self.processor = processor
        self.model = self.init_lvu()
        
    def init_lvu(self):
        if self.config.model_type not in lvu_init_model_map:
            raise ValueError(f"Model type {self.config.model_type} not supported.")
        
        init_model_func = lvu_init_model_map[self.config.model_type]
        run_model_func = lvu_run_model_map[self.config.model_type]
        model = init_model_func(self.model, self.config)
        self.run_model_func = run_model_func.__get__(self)
        
        return model
    
    def generate(self, question, video_path, **generation_kwargs):
        if self.config.model_type not in lvu_run_model_map:
            raise ValueError(f"Model type {self.config.model_type} not supported.")

        output = self.run_model_func(question, video_path, **generation_kwargs)
        
        return output
    
    
if __name__ == "__main__":
    config = LVUConfig(
        model_name_or_path="Qwen/Qwen2.5-VL-7B-Instruct", 
        model_type="qwen25_vl",
        top_k_predict_type="key_norms_small",
        video_group_size=4,
        top_k=10,
        prefill_prune_starting_layer=1,
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