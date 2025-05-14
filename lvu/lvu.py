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
        
        # time processing
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

def main(
    model_name_or_path: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    model_type: str = "qwen25_lvu",
    video_group_size: int = 1,
    video_path: str = "Q8AZ16uBhr8_resized_fps2_mute.mp4",
    top_k_predict_type: str = "key_norms_small",
):
    config = LVUConfig(
        model_name_or_path=model_name_or_path,
        model_type=model_type,
        # top_k_predict_type="query_attention_weights",
        # top_k_predict_type="query_attention_weights_by_value_norm",
        top_k_predict_type=top_k_predict_type,
        video_group_size=video_group_size,
        top_k=None,
        top_p=0.2,
        prefill_prune_starting_layer=None,
        adaptive_local_attention=True,
        # num_frames=128,
        fps=1,
        use_tqdm=True,
        # top_k_decay_type="linear",
        # top_k_decay_factor=0.33,
    )
    lvu = LVU(config)
    
    # question = "Describe this video."
    # video_path = "Q8AZ16uBhr8_resized_fps2_mute.mp4"
    # generation_kwargs = {
    #     "max_new_tokens": 512,
    #     "do_sample": False,
    #     "top_p": 1.0,
    # }
    # output = lvu.generate(question, video_path, **generation_kwargs)
    # print(output)
    
    DEMO_QUESTIONS = [
        "As depicted in the video, how is the relationship between the rabbit and human?\nOptions:\nA. Hostile.\nB. Friend.\nC. Cooperator.\nD. No one is correct above.\nAnswer with the option's letter from the given choices directly.",
#        "What is the impression of the video?\nOptions:\nA. Sad.\nB. Funny.\nC. Horrible.\nD. Silent.\nAnswer with the option's letter from the given choices directly.",
#        "What is the subject of the video?\nOptions:\nA. Rabbit likes to eat carrots.\nB. How to raise a rabbit.\nC. A rabbit gives people trouble.\nD. A rabbit performs for food.\nAnswer with the option's letter from the given choices directly.",
    ]
    EXPECTED_ANSWERS = ['A', 'B', 'C']
    
    for question, expected_answer in zip(DEMO_QUESTIONS, EXPECTED_ANSWERS):
        generation_kwargs = {
            "max_new_tokens": 512,
            "do_sample": False,
            "top_p": 1.0,
        }
        output = lvu.generate(question, video_path, **generation_kwargs)
        print(f"Question: {question}")
        print(f"Expected Answer: {expected_answer}")
        print(f"Model Output: {output}")
             
if __name__ == "__main__":
    import fire
    fire.Fire(main)