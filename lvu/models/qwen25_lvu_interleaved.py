import torch
import numpy as np
import time
import time
import sys
import types
import threading
import numpy as np
import os
from PIL import Image
from queue import Queue
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Tuple, Union
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
from transformers.models.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessorKwargs, BatchFeature
import warnings
import qwen_vl_utils.vision_process
from qwen_vl_utils.vision_process import *
from deepcodec import InterleavedVideoReader
FPS_MAX_FRAMES = 100_000 # originally: 768 = 256 * 3

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


def save_image_to_home(img_array: torch.Tensor, filename: str = "img/output.png"):
    # Ensure shape is (H, W, 3)
    img = np.transpose(img_array.cpu().numpy(), (1, 2, 0))

    if img.dtype != np.uint8:
        img = np.clip(img, 0, 1) if img.max() <= 1.0 else np.clip(img, 0, 255)
        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)

    pil_img = Image.fromarray(img)
    # Get home directory
    home_path = os.path.expanduser("~")
    # Save image
    save_path = os.path.join(home_path, filename)
    pil_img.save(save_path)
    print(f"Image saved to {save_path}")


class PixelIterator:

    def __init__(self,qwen_vr, vr, frames_per_block,video_kwargs, processor):

        self.qwen_vr = qwen_vr
        self.iterations = 0
        self.frames_per_block = frames_per_block
        self.vr = vr
        self.processor = processor
        self.video_kwargs = video_kwargs
        self.processor_timing = 0

    def __iter__(self):
        return self
    
    def __next__(self):

        s = time.time()
        frames = torch.from_numpy(next(self.vr)).float()
        e = time.time()
        self.qwen_vr.total_timing += e-s
        s = time.time()
        #save_image_to_home(frames[8], f"img/{self.iterations}.png")
        pixels = self.processor(
            text="a",
            images=[],
            videos=[frames],
            padding=True,
            return_tensors="pt",
            **self.video_kwargs,
        )['pixel_values_videos']
        self.iterations += 1
        e = time.time()
        self.processor_timing += e - s
        return pixels

class AsyncPixelIterator(PixelIterator):
    def __init__(self, qwen_vr, vr, frames_per_block, video_kwargs, processor, buffer_size=3):
        super().__init__(qwen_vr, vr, frames_per_block, video_kwargs, processor)
        # Threading components
        self.buffer = Queue(maxsize=buffer_size)
        self.is_finished = False
        self.worker_thread = None
        self.exception = None
        
    def __iter__(self):
        # Start the background worker thread
        self.worker_thread = threading.Thread(target=self._background_worker, daemon=True)
        self.worker_thread.start()
        return self
    
    def __next__(self):
        # Get the next processed frame from the buffer
        while True:
            if self.exception:
                raise self.exception
            
            if not self.buffer.empty():
                return self.buffer.get()
            
            if self.is_finished and self.buffer.empty():
                raise StopIteration
            
            # Wait a bit before checking again
            time.sleep(0.01)
    
    def _background_worker(self):
        """Background thread that continuously processes frames"""
        try:
            while True:
                # Process the next frame
                pixels = self._process_frame()
                if pixels is None:
                    break
                self.buffer.put(pixels)
        except StopIteration:
            self.is_finished = True
        except Exception as e:
            self.exception = e
            self.is_finished = True
    
    def _process_frame(self):
        """Process a single frame"""
        try:
            s = time.time()
            
            # Get the next frame
            frames = torch.from_numpy(next(self.vr)).float()
            e = time.time()
            self.qwen_vr.total_timing += e - s
            #save_image_to_home(frames[8], f"img/{self.iterations}.png")
            s = time.time()
            pixels = self.processor(
                text="a",
                images=[],
                videos=[frames],
                padding=True,
                return_tensors="pt",
                **self.video_kwargs,
            )['pixel_values_videos']
            self.iterations += 1
            e = time.time()
            self.processor_timing += e - s
            return pixels
        except StopIteration:
            raise

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

def _read_video_interleaved(
    ele: dict,
):
    
    video_path = ele["video"]

    num_cores = int(os.environ.get("QUICKCODEC_CORES", "8"))
    num_intervals = int(os.environ.get("QUICKCODEC_INTERVALS", "64"))

    if os.cpu_count()-1 > num_cores:
        num_cores = os.cpu_count()-1 if os.cpu_count()-1 > 0 else 1
        warnings.warn(f"QuickCodec requested more cores than the system supports, num_cores was set to {num_cores}.")
    
    vr = InterleavedVideoReader(video_path, num_threads=num_cores, num_intervals=num_intervals)
    # TODO: support start_pts and end_pts
    if 'video_start' in ele or 'video_end' in ele:
        raise NotImplementedError("not support start_pts and end_pts in deepcodec for now.")
    total_frames, video_fps = len(vr), vr.get_fps()

    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()
    #video = torch.from_numpy(vr.get_batch(idx))

    sample_fps = nframes / max(total_frames, 1e-6) * video_fps

    return vr,idx, sample_fps

def fetch_video(ele: dict, image_factor: int = IMAGE_FACTOR, return_video_sample_fps: bool = False) -> torch.Tensor | list[Image.Image]:
    if isinstance(ele["video"], str):
        vr, idx, sample_fps = _read_video_interleaved(ele)
        nframes, height, width = len(idx), vr.height, vr.width
        min_pixels = ele.get("min_pixels", VIDEO_MIN_PIXELS)
        total_pixels = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
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
        else:
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=image_factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
        
        vr.height = resized_height
        vr.width = resized_width
        vr.interpolation = "LANCZOS"
        #vr.interpolation = "BICUBIC"
        vr.process(idx)

        if return_video_sample_fps:
            return vr, sample_fps, nframes
        return vr
    else:
        raise NotImplementedError


def process_vision_info(
    conversations: list[dict] | list[list[dict]],
    return_video_kwargs: bool = False,
) -> tuple[list[Image.Image] | None, list[torch.Tensor | list[Image.Image]] | None, Optional[dict]]:

    vision_infos = extract_vision_info(conversations)
    ## Read images or videos
    image_inputs = []
    video_inputs = []
    video_sample_fps_list = []
    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            raise Exception("NotImplementedError")
        elif "video" in vision_info:
            video_input, video_sample_fps, nframes = fetch_video(vision_info, return_video_sample_fps=True)
            video_sample_fps_list.append(video_sample_fps)
            video_inputs.append(video_input)
        else:
            raise ValueError("image, image_url or video should in content.")
    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None
    if return_video_kwargs:
        return image_inputs, video_inputs[0], {'fps': video_sample_fps_list}, nframes
    return image_inputs, video_inputs[0]

class QwenVideoReaderInterleaved:
    def __init__(self, path, processor):
        self.total_timing = 0
        self.path = path
        self.frames_per_block = None
        self.batch = None
        self.processor = processor
        
    def process(self, conv):

        # conv = [{
        #     "role": "user",
        #     "content": [{"type": "video", "video": str(self.path), "max_pixels": math.inf, "fps": 2}]
        # }]
        s = time.time()        
        self.image_inputs, self.vr,  self.video_kwargs, self.nframes = process_vision_info(conv, return_video_kwargs=True)
        e = time.time()
        self.total_timing += e-s

    def dummy_input(self):
                
        return {
            "video_grid_thw" : torch.tensor((
                self.nframes / self.processor.image_processor.temporal_patch_size,
                self.vr.height / self.processor.image_processor.patch_size,
                self.vr.width / self.processor.image_processor.patch_size
            ), dtype=torch.int64).unsqueeze(dim=0), 
            "second_per_grid_ts": -1,
            "pixel_values_videos": None,
            "fps" : self.video_kwargs.get("fps", 2.0),
        }
        
    def dummy_video_inputs(self):
        return [torch.empty((self.nframes, 3, self.vr.height, self.vr.width), dtype=torch.float32)]

    def set_frames_per_block(self, num_frames):
        self.vr.frame_iter = num_frames
        self.frames_per_block = num_frames


    def get_pixel_iterator(self):
        # return PixelIterator(self, self.vr, self.frames_per_block,self.video_kwargs, self.processor)
        return AsyncPixelIterator(self, self.vr, self.frames_per_block,self.video_kwargs, self.processor)

def dummy_call(
    self,
    images = None,
    text = None,
    videos = None,
    **kwargs,
):
    """
    Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
    and `kwargs` arguments to Qwen2TokenizerFast's [`~Qwen2TokenizerFast.__call__`] if `text` is not `None` to encode
    the text. To prepare the vision inputs, this method forwards the `vision_infos` and `kwrags` arguments to
    Qwen2VLImageProcessor's [`~Qwen2VLImageProcessor.__call__`] if `vision_infos` is not `None`.

    Args:
        images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
            The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
            tensor. Both channels-first and channels-last formats are supported.
        text (`str`, `List[str]`, `List[List[str]]`):
            The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
            (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
            `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
        videos (`np.ndarray`, `torch.Tensor`, `List[np.ndarray]`, `List[torch.Tensor]`):
            The image or batch of videos to be prepared. Each video can be a 4D NumPy array or PyTorch
            tensor, or a nested list of 3D frames. Both channels-first and channels-last formats are supported.
        return_tensors (`str` or [`~utils.TensorType`], *optional*):
            If set, will return tensors of a particular framework. Acceptable values are:
            - `'tf'`: Return TensorFlow `tf.constant` objects.
            - `'pt'`: Return PyTorch `torch.Tensor` objects.
            - `'np'`: Return NumPy `np.ndarray` objects.
            - `'jax'`: Return JAX `jnp.ndarray` objects.

    Returns:
        [`BatchFeature`]: A [`BatchFeature`] with the following fields:

        - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
        - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
            `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
            `None`).
        - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
        - **pixel_values_videos** -- Pixel values of videos to be fed to a model. Returned when `videos` is not `None`.
        - **image_grid_thw** -- List of image 3D grid in LLM. Returned when `images` is not `None`.
        - **video_grid_thw** -- List of video 3D grid in LLM. Returned when `videos` is not `None`.
        - **second_per_grid_ts** -- List of video seconds per time grid. Returned when `videos` is not `None`.
    """
    video_in = kwargs.pop("video_inputs")


    output_kwargs = self._merge_kwargs(
        Qwen2_5_VLProcessorKwargs,
        tokenizer_init_kwargs=self.tokenizer.init_kwargs,
        **kwargs,
    )
    if images is not None:
        image_inputs = self.image_processor(images=images, videos=None, **output_kwargs["images_kwargs"])
        image_grid_thw = image_inputs["image_grid_thw"]
    else:
        image_inputs = {}
        image_grid_thw = None

    if videos is not None:
        
        videos_inputs = {
            "video_grid_thw" : video_in["video_grid_thw"],
            "second_per_grid_ts": -1,
            "pixel_values_videos": None
        }        

        video_grid_thw = video_in["video_grid_thw"]

        #fps = output_kwargs["videos_kwargs"].pop("fps", 2.0)
        fps = video_in["fps"]
        if isinstance(fps, (int, float)):
            second_per_grid_ts = [self.image_processor.temporal_patch_size / fps] * len(video_grid_thw)
        elif hasattr(fps, "__len__") and len(fps) == len(video_grid_thw):
            second_per_grid_ts = [self.image_processor.temporal_patch_size / tmp for tmp in fps]
        else:
            raise ValueError(
                f"The length of fps ({len(fps) if hasattr(fps, '__len__') else fps}) must be equal to the length of video_grid_thw ({len(video_grid_thw)}) or fps should be a single number."
            )
        videos_inputs.update({"second_per_grid_ts": second_per_grid_ts})

    else:
        videos_inputs = {}
        video_grid_thw = None

    if not isinstance(text, list):
        text = [text]

    if image_grid_thw is not None:
        merge_length = self.image_processor.merge_size**2
        index = 0
        for i in range(len(text)):
            while self.image_token in text[i]:
                text[i] = text[i].replace(
                    self.image_token,
                    "<|placeholder|>" * (image_grid_thw[index].prod() // merge_length),
                    1,
                )
                index += 1
            text[i] = text[i].replace("<|placeholder|>", self.image_token)

    if video_grid_thw is not None:
        merge_length = self.image_processor.merge_size**2
        index = 0
        for i in range(len(text)):
            while self.video_token in text[i]:
                text[i] = text[i].replace(
                    self.video_token,
                    "<|placeholder|>" * (video_grid_thw[index].prod() // merge_length),
                    1,
                )
                index += 1
            text[i] = text[i].replace("<|placeholder|>", self.video_token)

    text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])

    return BatchFeature(data={**text_inputs, **image_inputs, **videos_inputs})

import inspect
def _get_initial_cache_position(self, *args, **kwargs):
    # Get the function signature
    sig = inspect.signature(self.old_get_initial_cache_position)
    
    # Get parameter names (excluding 'self')
    param_names = [param.name for param in sig.parameters.values() 
                   if param.name not in ('self', 'args', 'kwargs')]
    
    # Transform *args to **kwargs using parameter names
    args_as_kwargs = dict(zip(param_names, args))
    
    # Combine with existing kwargs
    all_kwargs = {**args_as_kwargs, **kwargs}
    # Find model_kwargs in the mapped arguments
    model_kwargs = all_kwargs.get('model_kwargs')
    
    # Early return if cache_position already exists in model_kwargs
    if model_kwargs is not None and "cache_position" in model_kwargs:
        return model_kwargs
    return self.old_get_initial_cache_position(*args, **kwargs)
    
def init_lvu_model(model, config: LVUConfig):
    """
    Initialize the LVU model for Qwen 2.5 VL. 
    - replace the decoder layer forward function with the LVU version
    Args:
        model: The model to be initialized.
        config: The configuration for the LVU model.
    """
    _model = model
    if isinstance(_model, Qwen2_5_VLForConditionalGeneration):
        _model = _model.model
        if not hasattr(model, "get_rope_index"):
            # for transformers > 4.50.0
            model.get_rope_index = _model.get_rope_index
    if isinstance(_model, Qwen2_5_VLModel):
        if hasattr(_model, "layers"):
            _model = _model
        elif hasattr(_model, "language_model"):
            _model = _model.language_model
        else:
            raise ValueError("Qwen2_5_VLModel must have either `model` or `language_model` attribute.")
    try:
        decoder_layers = _model.layers
    except AttributeError:
        raise ValueError("Did not find `layers` attribute in the model. Please check your qwen2.5_vl source code and transformers version.") 
    
    total_layers= len(decoder_layers)
    for i, layer in enumerate(decoder_layers):
        # Set the forward function for each decoder layer and filling the parameters in the config
        layer.forward = lvu_qwen25_vl_decoder_layer_forward.__get__(layer)
        layer.self_attn.forward = lvu_qwen25_vl_flash_attention_2_forward.__get__(layer.self_attn)
        layer.lvu_layer_config = LVULayerConfig(layer_idx=layer.self_attn.layer_idx, total_layers=total_layers, lvu_config=config)
    model.old_get_initial_cache_position = model._get_initial_cache_position
    model._get_initial_cache_position = _get_initial_cache_position.__get__(model)
    
    return model

def run_lvu_model(self, question, video_path, **generation_kwargs):
    lvu_config = self.config
    fps = lvu_config.fps
    num_frames = lvu_config.num_frames
    extra_kwargs = lvu_config.extra_kwargs or {}
    max_pixels = extra_kwargs.get("max_pixels", None)
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
    if not cache_file.exists() or False:        
        # Interleaved processing
        vr = QwenVideoReaderInterleaved(video_path, processor)
        vr.process(messages)

        # used for finding correct shapes of blocks
        # I think only time dim is needed ~ can probably remove other dims
        video_inputs = vr.dummy_video_inputs()
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
    print(f"Preprocessing time for video: {end - start:.2f}s")
    
    s = time.time()
    
    processor.dummy_call = types.MethodType(dummy_call, processor)
    whole_inputs = processor.dummy_call(
        text=text,
        images=None,
        videos=[],
        padding=True,
        return_tensors="pt",
        **vr.video_kwargs,
        video_inputs = vr.dummy_input()
    )

    e = time.time()
    print(f"Tokenizer time was: {e - s:.2f}s")
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
    temporal_patch_size = processor.image_processor.temporal_patch_size
    if not video_group_size % temporal_patch_size == 0:
        video_group_size += temporal_patch_size - (video_group_size % temporal_patch_size)
    if video_group_size is not None and video_group_size > 0:
        video_groups = video_inputs[0].split(video_group_size)
        assert all(len(group) % 2 == 0 for group in video_groups), "The video group size should be even."
        video_groups_tokens = [int(n_video_tokens * (len(group) / len(video_inputs[0]))) for group in video_groups]
        video_grid_thw = whole_inputs['video_grid_thw'][0]
        video_groups_grid_thw = []
        for group in video_groups:
            video_groups_grid_thw.append(
                torch.tensor(
                    [(len(group) -1 ) // temporal_patch_size + 1,
                    video_grid_thw[1],
                    video_grid_thw[2]]
                ).unsqueeze(0)
            )
    
    vr.set_frames_per_block(video_group_size)
    pixel_iter = vr.get_pixel_iterator()
    
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
    
    total_prefill_time = 0

    # start processing the video groups
    print(f"Processing total of {vr.nframes} frames of {video_group_size} frames each.")
    e2e_start = time.time()
    for i, (pixel_values_videos_groups_i) in tqdm(enumerate(pixel_iter), 
        desc="Processing video groups", disable=not lvu_config.use_tqdm,
        total=vr.nframes // video_group_size,
        ):
        start_of_block_prefill = time.time()
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
        end_of_block_prefill_time = time.time()
        total_prefill_time += end_of_block_prefill_time - start_of_block_prefill
        # print(f"past_key_values shape: {past_key_values[0][0].shape}")
    assert past_len < whole_inputs['input_ids'].shape[1], "The past length should be less than the final input length."
    if lvu_config.query_based:
        # reset prompt length as all video groups are processed
        past_key_values.set_prompt_length(0)
    # end of processing the video groups
    start_of_decoding = time.time()

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
    generated_ids = model.generate(**final_inputs, **generation_kwargs)
    lvu_config.enable = cache_enable
    end_of_decoding = time.time()
    decoding_time = end_of_decoding - start_of_decoding
    
    e2e_end = time.time()
    e2e_time = e2e_end - e2e_start

    print(f"total time spent fetching frames was: {vr.total_timing}")
    print(f"total time spent on processor was: {pixel_iter.processor_timing}")
    print(f"total time spent on prefill was: {total_prefill_time}")
    print(f"total time spent on decoding was: {decoding_time}")
    print(f"total time spent on e2e fetching and decoding was: {e2e_time}")
    print(f"Time saved by interleaved processing was: {vr.total_timing + pixel_iter.processor_timing + total_prefill_time + decoding_time - e2e_time}")

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(final_inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text
    
sys.modules["qwen_vl_utils.vision_process"].smart_nframes = smart_nframes
