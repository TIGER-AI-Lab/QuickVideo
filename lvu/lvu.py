import torch
from dataclasses import dataclass
from transformers import AutoProcessor, AutoModelForImageTextToText
from .models import lvu_init_model_map, lvu_run_model_map, lvu_chat_model_map
from .lvu_config import LVUConfig

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
    
    def run_model_func(self, question, video_path, **generation_kwargs):
        raise NotImplementedError("run_model_func not implemented.")

    def chat_model_func(self, messages, **generation_kwargs):
        raise NotImplementedError("chat_model_func not implemented.")
        
    def init_lvu(self):
        if self.config.model_type not in lvu_init_model_map:
            raise ValueError(f"Model type {self.config.model_type} not supported.")
        
        init_model_func = lvu_init_model_map[self.config.model_type]
        run_model_func = lvu_run_model_map[self.config.model_type]
        model = init_model_func(self.model, self.config)
        self.run_model_func = run_model_func.__get__(self)
        if self.config.model_type in lvu_chat_model_map:
            self.chat_model_func = lvu_chat_model_map[self.config.model_type].__get__(self)
        
        return model
    
    def generate(self, question, video_path, **generation_kwargs):
        if self.config.model_type not in lvu_run_model_map:
            raise ValueError(f"Model type {self.config.model_type} not supported.")

        output = self.run_model_func(question, video_path, **generation_kwargs)
        
        return output
    
    def chat(self, messages:dict, **generation_kwargs):
        if self.config.model_type not in lvu_run_model_map:
            raise ValueError(f"Model type {self.config.model_type} not supported.")
        output = self.chat_model_func(messages, **generation_kwargs)
        return output
    
if __name__ == "__main__":
    config = LVUConfig(
        model_name_or_path="Qwen/Qwen2.5-VL-7B-Instruct", 
        model_type="qwen25_vl",
        top_k_predict_type="key_norms_small",
        video_group_size=16,
        top_k=None,
        prefill_prune_starting_layer=None,
        adaptive_local_attention=True,
        num_frames=512,
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