import torch
import numpy as np
import dataclasses
from typing import Optional, Tuple
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLModel
from transformers.cache_utils import Cache
from qwen_vl_utils import process_vision_info
from ..utils import post_process_kv_cache, preprocess_hidden_states
from ..lvu import LVUConfig    

def lvu_qwen25_vl_decoder_layer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
    **kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    """
    Args:
        hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
        attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
            `(batch, sequence_length)` where padding elements are indicated by 0.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more detail.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
            (see `past_key_values`).
        past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence.
        position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
            Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
            with `head_dim` being the embedding dimension of each attention head.
        kwargs (`dict`, *optional*):
            Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
            into the model
    """
    lvu_config = getattr(self, "lvu_config", None)
    if lvu_config is None:
        raise ValueError("LVUConfig is not set in the model. Please initialize the LVU model first.")
    hidden_states, attention_mask, position_ids = preprocess_hidden_states(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
    )
    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
    )
    hidden_states, attention_mask, position_ids, present_key_value = post_process_kv_cache(
        hidden_states,
        attention_mask,
        position_ids,
        attn_weights=self_attn_weights,
        present_key_value=present_key_value,
        lvu_config=lvu_config,
    )
    hidden_states = residual + hidden_states
    
    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    if lvu_config.prefill_prune_starting_layer is not None and lvu_config.layer_idx >= lvu_config.prefill_prune_starting_layer:
        # prune for next layer
        hidden_states = (hidden_states, attention_mask, position_ids)
        
    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    return outputs

def _get_initial_cache_position(self, input_ids, model_kwargs):
    if "cache_position" in model_kwargs:
        return model_kwargs
    """Calculates `cache_position` for the pre-fill stage based on `input_ids` and optionally past length"""
    # `torch.compile`-friendly `torch.arange` from a shape -- the lines below are equivalent to `torch.arange`
    if "inputs_embeds" in model_kwargs and not self.config.is_encoder_decoder:
        cache_position = torch.ones_like(model_kwargs["inputs_embeds"][0, :, 0], dtype=torch.int64).cumsum(0) - 1
    elif "decoder_inputs_embeds" in model_kwargs and self.config.is_encoder_decoder:
        cache_position = (
            torch.ones_like(model_kwargs["decoder_inputs_embeds"][0, :, 0], dtype=torch.int64).cumsum(0) - 1
        )
    else:
        cache_position = torch.ones_like(input_ids[0, :], dtype=torch.int64).cumsum(0) - 1

    past_length = 0
    if model_kwargs.get("past_key_values") is not None:
        cache = model_kwargs["past_key_values"]
        past_length = 0
        if not isinstance(cache, Cache):
            past_length = cache[0][0].shape[2]
        elif hasattr(cache, "get_seq_length") and cache.get_seq_length() is not None:
            past_length = cache.get_seq_length()

        cache_position = cache_position[past_length:]

    model_kwargs["cache_position"] = cache_position
    return model_kwargs

    
def init_lvu_model(model, config: LVUConfig):
    """
    Initialize the LVU model for Qwen 2.5 VL. 
    - replace the decoder layer forward function with the LVU version
    Args:
        model: The model to be initialized.
        config: The configuration for the LVU model.
    """
    if isinstance(model, Qwen2_5_VLForConditionalGeneration):
        decoder_layers = model.model.layers
    elif isinstance(model, Qwen2_5_VLModel):
        decoder_layers = model.layers
    else:
        raise ValueError("Model must be either Qwen2_5_VLForConditionalGeneration or Qwen2_5_VLModel")
    
    for i, layer in enumerate(decoder_layers):
        # Set the forward function for each decoder layer and filling the parameters in the config
        # layer.forward = lvu_qwen25_vl_decoder_layer_forward.__get__(layer)
        layer.lvu_config = dataclasses.replace(config)
        layer.lvu_config.layer_idx = layer.self_attn.layer_idx # should be same as i
    model._get_initial_cache_position = _get_initial_cache_position.__get__(model)
    
    return model
        
def run_lvu_model(self, question, video_path, **generation_kwargs):
    model = self.model
    processor = self.processor
    lvu_config = self.config
    fps = 0.2
    # Messages containing a local video path and a text query
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": 360 * 420,
                    "fps": fps,
                },
            ],
        }
    ]
    # Process the messages
    #In Qwen 2.5 VL, frame rate information is also input into the model to align with absolute time.
    # Preparation for inference
    video_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    messages[0]['content'].append({"type": "text", "text": question})
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    
    assert len(video_inputs) <= 1, "Only one video is supported for now."
    video_group_size = lvu_config.video_group_size
    if video_group_size is not None and video_group_size > 0:
        video_groups = [
            video_inputs[0][i : i + video_group_size] for i in range(0, len(video_inputs[0]), video_group_size)
        ]
        assert all(len(group) % 2 == 0 for group in video_groups), "The video group size should be even."
    else:
        video_groups = [video_inputs[0]]
    # start to process the video groups
    past_video_groups = None
    past_key_values = None
    past_len = 0
    past_pixel_values_video_len = 0
    for i, video_group_i in enumerate(video_groups):
        if past_video_groups is None:
            video_group_0_i = video_group_i
        else:
            video_group_0_i = np.concatenate((past_video_groups, video_group_i), axis=0)
        past_video_groups = video_group_0_i
        
        group_i_inputs = processor(
            text=[video_text],
            images=image_inputs,
            videos=[video_group_0_i],
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        input_ids = group_i_inputs['input_ids'][:, past_len:-3]
        attention_mask = group_i_inputs['attention_mask'][:, past_len:-3]
        cache_position = torch.arange(input_ids.shape[1], dtype=torch.int64, device=model.device) + past_len
        pixel_values_videos = group_i_inputs['pixel_values_videos'][past_pixel_values_video_len:]
        past_len += input_ids.shape[1]
        past_pixel_values_video_len += pixel_values_videos.shape[0]
        # second time parsing, the video grid information correct
        group_i_inputs = processor(
            text=[video_text],
            images=image_inputs,
            videos=[video_group_i],
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        group_i_inputs['input_ids'] = input_ids
        group_i_inputs['attention_mask'] = attention_mask
        group_i_inputs['cache_position'] = cache_position
        group_i_inputs['pixel_values_videos'] = pixel_values_videos
        
        group_i_inputs = group_i_inputs.to(model.device)
        group_i_inputs['past_key_values'] = past_key_values
        outputs = model(**group_i_inputs, use_cache=True)
        past_key_values = outputs.past_key_values

    assert len(past_video_groups) == video_inputs[0].shape[0], "The length of the past video groups should be equal to the input video length."
    final_inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )
    assert past_len < final_inputs['input_ids'].shape[1], "The past length should be less than the final input length."
    final_inputs['input_ids'] = final_inputs['input_ids'][:, past_len:]
    final_inputs['attention_mask'] = final_inputs['attention_mask'][:, past_len:]
    final_inputs['cache_position'] = torch.arange(final_inputs.input_ids.shape[1], dtype=torch.int64, device=model.device) + past_len
    final_inputs = final_inputs.to(model.device)
    final_inputs['past_key_values'] = past_key_values
    generated_ids = model.generate(**final_inputs, **generation_kwargs)
        
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(final_inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text
    
