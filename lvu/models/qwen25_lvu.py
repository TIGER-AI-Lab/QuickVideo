import torch
import numpy as np
import dataclasses
import time
import hashlib
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Tuple
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLModel
from transformers.feature_extraction_utils import BatchFeature
from transformers.cache_utils import Cache
from qwen_vl_utils import process_vision_info, extract_vision_info
from ..utils import post_process_kv_cache
from ..lvu_config import LVUConfig, LVULayerConfig
from ..lvu_cache import LVUCache
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    apply_multimodal_rotary_pos_emb,
    repeat_kv,
    _flash_attention_forward,
)

def lvu_qwen25_vl_flash_attention_2_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
    ):
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

    # Because the input can be padded, the absolute sequence length depends on the max position id.
    cos, sin = position_embeddings
    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
    )

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position, "query_states": query_states}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    dropout_rate = 0.0 if not self.training else self.attention_dropout

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in float16 just to be sure everything works as expected.
    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        logger.warning_once(
            f"The input hidden states seems to be silently casted in float32, this might be related to"
            f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
            f" {target_dtype}."
        )

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    # Reashape to the expected shape for Flash Attention
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    if (
        self.config.use_sliding_window
        and getattr(self.config, "sliding_window", None) is not None
        and self.layer_idx >= self.config.max_window_layers
    ):
        sliding_window = self.config.sliding_window
    else:
        sliding_window = None

    attn_output = _flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        q_len,
        dropout=dropout_rate,
        sliding_window=sliding_window,
        is_causal=self.is_causal,
        use_top_left_mask=self._flash_attn_uses_top_left_mask,
    )

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

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
        hidden_states (`torch.FloatTensor`): 
            - input to the layer of shape `(batch, seq_len, embed_dim)`
            - or a tuple of `(hidden_states, attention_mask, position_ids, cache_position, position_embeddings)` 
                meaning that the previous layer has prune the hidden states to topk
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
    lvu_layer_config = getattr(self, "lvu_layer_config", None)
    lvu_config = getattr(lvu_layer_config, "lvu_config", None)
    if lvu_config is None:
        raise ValueError("LVUConfig is not set in the model. Please initialize the LVU model first.")
    
    if isinstance(hidden_states, tuple):
        # this means that previous layer has prune the hidden states to topk
        hidden_states, attention_mask, position_ids, cache_position, position_embeddings = hidden_states

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
    hidden_states = residual.to(hidden_states.device) + hidden_states
    hidden_states, attention_mask, position_ids, cache_position, position_embeddings, present_key_value = post_process_kv_cache(
        hidden_states,
        attention_mask,
        position_ids=position_ids,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        attn_weights=self_attn_weights,
        present_key_value=present_key_value,
        lvu_layer_config=lvu_layer_config,
    )
    
    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    if lvu_config.enable and lvu_layer_config.prune_for_next_layer and not lvu_layer_config.is_last_layer:
        # pass all the pruned information to next layer. If the last layer, we don't need to save other information except hidden_states
        hidden_states = (hidden_states, attention_mask, position_ids, cache_position, position_embeddings)
        
    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    return outputs


import qwen_vl_utils.vision_process
from qwen_vl_utils.vision_process import *
import sys

def _read_video_decord_cpu(
    ele: dict,
) -> (torch.Tensor, float):
    """read video using decord.VideoReader

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    import decord
    num_cores = int(os.environ.get("DEEPCODEC_CORES", "4"))
    video_path = ele["video"]
    st = time.time()
    vr = decord.VideoReader(video_path, num_threads=num_cores)
    # TODO: support start_pts and end_pts
    if 'video_start' in ele or 'video_end' in ele:
        raise NotImplementedError("not support start_pts and end_pts in decord for now.")
    total_frames, video_fps = len(vr), vr.get_avg_fps()
    logger.info(f"decord:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()
    video = vr.get_batch(idx).asnumpy()
    video = torch.tensor(video).permute(0, 3, 1, 2)  # Convert to TCHW format
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps
    return video, sample_fps


def is_deepcodec_available() -> bool:
    import importlib.util
    if "DEEPCODEC_DISABLED" in os.environ:
        return False
    else:
        return importlib.util.find_spec("deepcodec") is not None

@lru_cache(maxsize=1)
def get_video_reader_backend() -> str:
    if FORCE_QWENVL_VIDEO_READER is not None:
        video_reader_backend = FORCE_QWENVL_VIDEO_READER
    elif is_deepcodec_available():
        video_reader_backend = "deepcodec"
    elif is_decord_available():
        video_reader_backend = "decord"
    else:
        video_reader_backend = "torchvision"
    print(f"qwen-vl-utils using {video_reader_backend} to read video.", file=sys.stderr)
    return video_reader_backend

def _read_video_deepcodec(
    ele: dict,
) -> (torch.Tensor, float):
    """read video using decord.VideoReader

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    from deepcodec import VideoReader as DCVideoReader
    video_path = ele["video"]
    resize = ele.pop("resize")

    st = time.time()
    num_cores = int(os.environ.get("DEEPCODEC_CORES", "4"))
    vr = DCVideoReader(video_path, num_threads=num_cores)

    total_frames, video_fps = len(vr), vr.get_fps()
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)

    total_pixels = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
    min_pixels = ele.get("min_pixels", VIDEO_MIN_PIXELS)
    max_pixels = max(min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR), int(min_pixels * 1.05))
    max_pixels_supposed = ele.get("max_pixels", max_pixels)
    if max_pixels_supposed > max_pixels:
        logger.warning(f"The given max_pixels[{max_pixels_supposed}] exceeds limit[{max_pixels}].")
    max_pixels = min(max_pixels_supposed, max_pixels)
    
    height, width = vr.height, vr.width
    resized_height, resized_width = resize["fxn"](
        height,
        width,
        factor=resize["image_factor"],
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    vr.height = resized_height
    vr.width = resized_width
    vr.interpolation = "LANCZOS"

    # TODO: support start_pts and end_pts
    if 'video_start' in ele or 'video_end' in ele:
        raise NotImplementedError("not support start_pts and end_pts in deepcodec for now.")

    idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()
    video = torch.from_numpy(vr.get_batch(idx))
    batch_end_time = time.time()
    print(f"deepcodec:  {video_path=}, {total_frames=}, {video_fps=}, time={batch_end_time-st:.3f}s")
    print(video.shape)
    # deepcodec already returns in TCHW format
    #video = torch.tensor(video).permute(0, 3, 1, 2)
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps

    return video, sample_fps

VIDEO_READER_BACKENDS = {
    "deepcodec": _read_video_deepcodec,
    "decord": _read_video_decord_cpu,
    "torchvision": qwen_vl_utils.vision_process._read_video_torchvision,
}

def fetch_video(ele: dict, image_factor: int = IMAGE_FACTOR, return_video_sample_fps: bool = False) -> torch.Tensor | list[Image.Image]:
    if isinstance(ele["video"], str):
        
        video_reader_backend = get_video_reader_backend()

        if video_reader_backend == "deepcodec":
            ele["resize"] = {
                "fxn": smart_resize,
                "image_factor": image_factor,
                }

        try:
            video, sample_fps = VIDEO_READER_BACKENDS[video_reader_backend](ele)
        except Exception as e:
            logger.warning(f"video_reader_backend {video_reader_backend} error, use torchvision as default, msg: {e}")
            video, sample_fps = VIDEO_READER_BACKENDS["torchvision"](ele)

        nframes, _, height, width = video.shape
        total_pixels = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
        min_pixels = ele.get("min_pixels", VIDEO_MIN_PIXELS)
        max_pixels = max(min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR), int(min_pixels * 1.05))
        max_pixels_supposed = ele.get("max_pixels", max_pixels)
        if max_pixels_supposed > max_pixels:
            logger.warning(f"The given max_pixels[{max_pixels_supposed}] exceeds limit[{max_pixels}].")
        max_pixels = min(max_pixels_supposed, max_pixels)
        if "resized_height" in ele and "resized_width" in ele:
            resized_height, resized_width = smart_resize(
                ele["resized_height"],
                ele["resized_width"],
                factor=image_factor,
            )
        elif video_reader_backend != "deepcodec":
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=image_factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )

        # deepcodec already handles resizing
        if video_reader_backend == "deepcodec":
            video = video.float()
        else:
            video = transforms.functional.resize(
                video,
                [resized_height, resized_width],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ).float()
        if return_video_sample_fps:
            return video, sample_fps
        return video
    else:
        assert isinstance(ele["video"], (list, tuple))
        process_info = ele.copy()
        process_info.pop("type", None)
        process_info.pop("video", None)
        images = [
            fetch_image({"image": video_element, **process_info}, size_factor=image_factor)
            for video_element in ele["video"]
        ]
        nframes = ceil_by_factor(len(images), FRAME_FACTOR)
        if len(images) < nframes:
            images.extend([images[-1]] * (nframes - len(images)))
        if return_video_sample_fps:
            return images, process_info.pop("fps", 2.0)
        return images


sys.modules["qwen_vl_utils.vision_process"].get_video_reader_backend = get_video_reader_backend
sys.modules["qwen_vl_utils.vision_process"].VIDEO_READER_BACKENDS = VIDEO_READER_BACKENDS
sys.modules["qwen_vl_utils.vision_process"].fetch_video = fetch_video
FPS_MAX_FRAMES = 100_000 # 768 = 256 * 3

def smart_nframes(
    ele: dict,
    total_frames: int,
    video_fps: int | float,
) -> int:
    """calculate the number of frames for video used for model inputs.

    Args:
        ele (dict): a dict contains the configuration of video.
            support either `fps` or `nframes`:
                - nframes: the number of frames to extract for model inputs.
                - fps: the fps to extract frames for model inputs.
                    - min_frames: the minimum number of frames of the video, only used when fps is provided.
                    - max_frames: the maximum number of frames of the video, only used when fps is provided.
        total_frames (int): the original total number of frames of the video.
        video_fps (int | float): the original fps of the video.

    Raises:
        ValueError: nframes should in interval [FRAME_FACTOR, total_frames].

    Returns:
        int: the number of frames for video used for model inputs.
    """
    assert not ("fps" in ele and "nframes" in ele), "Only accept either `fps` or `nframes`"
    if "nframes" in ele:
        nframes = round_by_factor(ele["nframes"], FRAME_FACTOR)
        nframes = min(nframes, total_frames)
        nframes -= (nframes % FRAME_FACTOR)
    else:
        fps = ele.get("fps", FPS)
        min_frames = ceil_by_factor(ele.get("min_frames", FPS_MIN_FRAMES), FRAME_FACTOR)
        max_frames = floor_by_factor(ele.get("max_frames", min(FPS_MAX_FRAMES, total_frames)), FRAME_FACTOR)
        nframes = total_frames / video_fps * fps
        if nframes > total_frames:
            logger.warning(f"smart_nframes: nframes[{nframes}] > total_frames[{total_frames}]")
        nframes = min(min(max(nframes, min_frames), max_frames), total_frames)
        nframes = floor_by_factor(nframes, FRAME_FACTOR)
    if not (FRAME_FACTOR <= nframes and nframes <= total_frames):
        raise ValueError(f"nframes should in interval [{FRAME_FACTOR}, {total_frames}], but got {nframes}.")
    return nframes

sys.modules["qwen_vl_utils.vision_process"].smart_nframes = smart_nframes

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
    
    total_layers= len(decoder_layers)
    for i, layer in enumerate(decoder_layers):
        # Set the forward function for each decoder layer and filling the parameters in the config
        layer.forward = lvu_qwen25_vl_decoder_layer_forward.__get__(layer)
        layer.self_attn.forward = lvu_qwen25_vl_flash_attention_2_forward.__get__(layer.self_attn)
        layer.lvu_layer_config = LVULayerConfig(layer_idx=layer.self_attn.layer_idx, total_layers=total_layers, lvu_config=config)
    model._get_initial_cache_position = _get_initial_cache_position.__get__(model)
    
    return model

def run_lvu_model(self, question, video_path, **generation_kwargs):
    lvu_config = self.config
    fps = lvu_config.fps
    num_frames = lvu_config.num_frames
    extra_kwargs = lvu_config.extra_kwargs or {}
    max_pixels = extra_kwargs.get("max_pixels", 360 * 420)
    min_pixels = extra_kwargs.get("min_pixels", None)

    video_content = {
        "type": "video",
        "video": video_path,
    }
    if max_pixels is not None:
        video_content["max_pixels"] = max_pixels
    if min_pixels is not None:
        video_content["min_pixels"] = min_pixels
    if fps is not None:
        video_content["fps"] = fps
    elif num_frames is not None:
        video_content["nframes"] = num_frames
    else:
        raise ValueError("Either fps or num_frames should be set.")
    # Messages containing a local video path and a text query
    messages = [
        {
            "role": "user",
            "content": [
                video_content,
                {"type": "text", "text": question}
            ],
        }
    ]
    return chat_lvu_model(self, messages, **generation_kwargs)
    
def chat_lvu_model(self, messages, **generation_kwargs):
    model = self.model
    processor = self.processor
    lvu_config = self.config
    
    # Process the messages
    #In Qwen 2.5 VL, frame rate information is also input into the model to align with absolute time.
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    start = time.time()
    cache_dir = lvu_config.cache_dir or "~/.cache/video_cache/qwen25_vl"
    vision_info = extract_vision_info(messages)
    assert len(vision_info) == 1, "Only one video is supported for now."
    video_path = Path(vision_info[0]["video"])
    cache_key = video_path.stem
    for k, v in vision_info[0].items():
        if k not in ['type', 'video']:
            cache_key += f"_{k}={v}"
    # cache_key = video_path.stem + "_" + hashlib.md5(str(vision_info).encode()).hexdigest()
    cache_dir = Path(cache_dir).expanduser()
    cache_file = cache_dir / f"{cache_key}.pt"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not cache_file.exists():
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        # save to cache
        if lvu_config.save_video_cache:
            torch.save({
                "image_inputs": image_inputs,
                "video_inputs": video_inputs,
                "video_kwargs": video_kwargs,
            }, cache_file)
            cache_file_size_gb = cache_file.stat().st_size / (1024 ** 3)
            print(f"Saved video cache to {cache_file} ({cache_file_size_gb:.2f} GB)")
    else:
        print(f"Cache file {cache_file} found. Loading video frames...")
        results = torch.load(cache_file)
        image_inputs = results["image_inputs"]
        video_inputs = results["video_inputs"]
        video_kwargs = results["video_kwargs"]
    end = time.time()
    video_processing_time = end - start
    
    whole_inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )

    whole_inputs = whole_inputs.to(model.device)
    n_video_tokens = (whole_inputs['input_ids'] == model.config.video_token_id).sum().item()
    video_token_idxs = (whole_inputs['input_ids'] == model.config.video_token_id).nonzero(as_tuple=True)[1]
    first_video_token_id_idx = video_token_idxs[0].item()
    last_video_token_id_idx = video_token_idxs[-1].item()
    position_ids, rope_deltas = model.get_rope_index(
        whole_inputs['input_ids'],
        whole_inputs.get('image_grid_thw', None),
        whole_inputs.get('video_grid_thw', None),
        whole_inputs.get('second_per_grid_ts', None),
        whole_inputs['attention_mask'],
    )
    model.rope_deltas = rope_deltas
    
    assert len(video_inputs) <= 1, "Only one video is supported for now."
    video_group_size = lvu_config.video_group_size
    if video_group_size is not None and video_group_size > 0:
        video_groups = video_inputs[0].split(video_group_size)
        assert all(len(group) % 2 == 0 for group in video_groups), "The video group size should be even."
        video_groups_tokens = [int(n_video_tokens * (len(group) / len(video_inputs[0]))) for group in video_groups]
        video_grid_thw = whole_inputs['video_grid_thw'][0]
        video_groups_grid_thw = []
        for group in video_groups:
            video_groups_grid_thw.append(
                torch.tensor(
                    [int(video_grid_thw[0] * (len(group) / len(video_inputs[0]))),
                    video_grid_thw[1],
                    video_grid_thw[2]]
                ).unsqueeze(0)
            )
        pixel_values_videos_group_size = int((video_group_size / len(video_inputs[0])) * whole_inputs['pixel_values_videos'].shape[0])
        pixel_values_videos_groups = whole_inputs['pixel_values_videos'].split(pixel_values_videos_group_size)
    else:
        video_groups = [video_inputs[0]]
        video_groups_tokens = [n_video_tokens]
        video_groups_grid_thw = [whole_inputs['video_grid_thw']]
        pixel_values_videos_groups = [whole_inputs['pixel_values_videos']]

    # print("Sampled video frames: ", len(video_inputs[0]))
    # print("Video groups: ", [len(group) for group in video_groups])
    # print("Video groups tokens: ", video_groups_tokens)
    # print("Video groups grid thw: ", video_groups_grid_thw)
    # print("Pixel values videos groups: ", [group.shape for group in pixel_values_videos_groups])
    
    # preprepare the chunk processing
    past_key_values = LVUCache()
    past_len = 0
    video_token_idxs = (whole_inputs['input_ids'] == model.config.video_token_id).nonzero(as_tuple=True)[1]
    first_video_token_id_idx = video_token_idxs[0].item()
    last_video_token_id_idx = video_token_idxs[-1].item()
    prompt_input_ids = whole_inputs['input_ids'][:, last_video_token_id_idx + 1:]
    prompt_attention_mask = whole_inputs['attention_mask'][:, last_video_token_id_idx + 1:]
    if lvu_config.query_based:
        past_key_values.set_prompt_length(prompt_input_ids.shape[1])
    video_groups_tokens[0] += first_video_token_id_idx # add the tokens before the first video group as well
    
    total_prefill = 0

    # start processing the video groups
    for i, pixel_values_videos_groups_i in tqdm(enumerate(pixel_values_videos_groups), 
        desc="Processing video groups", total=len(pixel_values_videos_groups), disable=not lvu_config.use_tqdm):
        
        prefill_start = time.time()
        
        group_i_inputs = {
            "video_grid_thw": video_groups_grid_thw[i],
            "second_per_grid_ts": whole_inputs['second_per_grid_ts'],
            "pixel_values_videos": pixel_values_videos_groups_i,
        }
        group_i_inputs = BatchFeature(data=group_i_inputs)
        group_i_inputs['input_ids'] = whole_inputs['input_ids'][:, past_len:past_len + video_groups_tokens[i]]
        group_i_inputs['attention_mask'] = whole_inputs['attention_mask'][:, past_len:past_len + video_groups_tokens[i]]
        if lvu_config.query_based:
            group_i_inputs['input_ids'] = torch.cat((group_i_inputs['input_ids'], prompt_input_ids), dim=1)
            group_i_inputs['attention_mask'] = torch.cat((group_i_inputs['attention_mask'], prompt_attention_mask), dim=1)
        
        group_i_inputs['cache_position'] = torch.arange(group_i_inputs['input_ids'].shape[1], dtype=torch.int64, device=model.device) + past_len
        group_i_inputs['position_ids'] = position_ids[:, :, past_len:past_len + group_i_inputs['input_ids'].shape[1]]
        past_len += video_groups_tokens[i] # only the video group tokens are counted, prompt tokens are not counted
        group_i_inputs = group_i_inputs.to(model.device)
        group_i_inputs['use_cache'] = True
        if lvu_config.adaptive_local_attention:
            group_i_inputs['past_key_values'] = past_key_values
            with torch.no_grad():
                outputs = model(**group_i_inputs)
            # later video groups will use the past key values
            past_key_values = outputs.past_key_values
        else:
            with torch.no_grad():
                outputs = model(**group_i_inputs)
            if not past_key_values:
                # first time parsing, the video grid information is not correct
                past_key_values = outputs.past_key_values
            else:
                # update the past key values
                if isinstance(outputs.past_key_values, Cache):
                    for i in range(len(outputs.past_key_values)):
                        past_key_values.update(outputs.past_key_values[i][0], outputs.past_key_values[i][1], i)
                else:
                    for i in range(len(outputs.past_key_values)):
                        for j in range(len(outputs.past_key_values[i])):
                            past_key_values[i][j] = torch.cat((past_key_values[i][j], outputs.past_key_values[i][j]), dim=2)
        # print(f"past_key_values shape: {past_key_values[0][0].shape}")
        prefill_end = time.time()
        total_prefill += prefill_end-prefill_start
    assert past_len < whole_inputs['input_ids'].shape[1], "The past length should be less than the final input length."
    if lvu_config.query_based:
        # reset prompt length as all video groups are processed
        past_key_values.set_prompt_length(0)
    # end of processing the video groups
    
    final_inputs = {
        "input_ids": whole_inputs['input_ids'][:, past_len:],
        "attention_mask": whole_inputs['attention_mask'][:, past_len:],
    }
    final_inputs = BatchFeature(data=final_inputs)
    final_inputs['cache_position'] = torch.arange(final_inputs.input_ids.shape[1], dtype=torch.int64, device=model.device) + past_len
    final_inputs['position_ids'] = position_ids[:, :, past_len:]
    assert final_inputs['input_ids'].shape[1] == final_inputs['position_ids'].shape[2], "The input ids and position ids should have the same length, but got {} and {}".format(
        final_inputs['input_ids'].shape[1], final_inputs['position_ids'].shape[2])
    final_inputs = final_inputs.to(model.device)
    final_inputs['past_key_values'] = past_key_values
    final_inputs['use_cache'] = True
    
    cache_enable = lvu_config.enable
    lvu_config.enable = lvu_config.do_top_k_for_query # determine whether to do topk or not
    decoding_start = time.time()
    generated_ids = model.generate(**final_inputs, **generation_kwargs)
    decoding_end = time.time()
    lvu_config.enable = cache_enable
    
    print(f"total time spent fetching frames was: {video_processing_time}")
    print(f"total time spent on prefill was: {total_prefill}")
    print(f"Time spent decoding was {decoding_end-decoding_start}")

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(final_inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text
    
