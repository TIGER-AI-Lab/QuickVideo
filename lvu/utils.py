import torch
import random
import torch.nn.functional as F
from typing import Tuple, Union, Optional
from .lvu_config import LVUConfig, LVULayerConfig
from .lvu_cache import DynamicCache, LVUCache
import math
import time

import numpy as np
from PIL import Image
import os

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

    def __iter__(self):
        return self
    
    def __next__(self):

        s = time.time()
        frames = torch.from_numpy(next(self.vr)).float()
        e = time.time()
        self.qwen_vr.total_timing += e-s
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
        return pixels

from qwen_vl_utils.vision_process import smart_nframes, extract_vision_info,IMAGE_FACTOR,VIDEO_MIN_PIXELS, get_video_reader_backend, VIDEO_READER_BACKENDS,FRAME_FACTOR,smart_resize,transforms,InterpolationMode,VIDEO_TOTAL_PIXELS,VIDEO_MAX_PIXELS,logger
from deepcodec import InterleavedVideoReader

def _read_video_interleaved(
    ele: dict,
):
    
    video_path = ele["video"]
    num_cores = int(os.environ.get("DEEPCODEC_CORES", "4"))
    vr = InterleavedVideoReader(video_path, num_threads=num_cores)
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

    def __init__(self, path, threads, intervals, processor):
        self.total_timing = 0
        self.path = path
        self.frames_per_block = None
        self.batch = None
        self.processor = processor
        self.threads = threads
        self.intervals = intervals
        
    def process(self):

        conv = [{
            "role": "user",
            "content": [{"type": "video", "video": str(self.path), "max_pixels": math.inf, "fps": 2}]
        }]
        s = time.time()        
        self.image_inputs, self.vr,  self.video_kwargs, self.nframes = process_vision_info(conv, return_video_kwargs=True)
        e = time.time()
        self.total_timing += e-s

    def dummy_input(self):
                
        return {
            "video_grid_thw" : torch.tensor((self.nframes/2, self.vr.height / 14, self.vr.width / 14), dtype=torch.int64).unsqueeze(dim=0), 
            "second_per_grid_ts": -1,
            "pixel_values_videos": None
        }
        
    def dummy_video_inputs(self):
        return [torch.empty((self.nframes, 3, self.vr.height, self.vr.width), dtype=torch.float32)]

    def set_frames_per_block(self, num_frames):
        self.vr.frame_iter = num_frames
        self.frames_per_block = num_frames


    def get_pixel_iterator(self):
        return PixelIterator(self, self.vr, self.frames_per_block,self.video_kwargs, self.processor)


def get_top_k_mask_to_predict(attn_weights, keys, values, outputs, top_k=100, predict_type="attention_weights"):
    """
    Args:
        attn_weights: (bz, 1, Q_len, K_len) or (bz, K_len)
        keys: (bz, num_heads, Q_len, C)
        values: (bz, num_heads, K_len, C)
        outputs: (bz, Q_len, C)
    Returns:
        top_k_mask: (bz, K_len)
    """
    if top_k <= 0:
        return None
    # random.seed(0)
    bz, _, k_len, _ = values.shape
    bz_top_k_idxs = []
    for bz_i in range(bz):
        if attn_weights is not None:
            if attn_weights.dim() == 4:
                attn_weights_i = attn_weights[bz_i].mean(0)[:, -k_len:] # (K_len, K_len)
            elif attn_weights.dim() == 2:
                attn_weights_i = attn_weights[bz_i] # (k_len)
            else:
                raise ValueError(f"Unknown attn_weights shape: {attn_weights.shape}")
        else:
            attn_weights_i = None
        keys_i = keys[bz_i]
        values_i = values[bz_i]
        outputs_i = outputs[bz_i]
        if predict_type == "salient_tokens":
            slident_value = []
            for i in range(len(attn_weights_i)):
                weights = attn_weights_i[i:, i]
                slident_value.append(weights.std().item() + weights.mean().item())
            top_k_idxs = sorted(range(len(slident_value)), key=lambda x: slident_value[x], reverse=True)[:top_k]
        elif predict_type == "attention_weights":
            mean_weights = []
            for i in range(len(attn_weights_i)):
                weights = attn_weights_i[i:, i]
                mean_weights.append(weights.mean().item())
            top_k_idxs = sorted(range(len(mean_weights)), key=lambda x: mean_weights[x], reverse=True)[:top_k]
        elif predict_type == "query_attention_weights":
            assert attn_weights_i is not None and attn_weights_i.dim() == 1, f"attn_weights_i should be 1D, but got {attn_weights_i.shape}"
            top_k_idxs = attn_weights_i.argsort(descending=True)[:top_k].tolist()
        elif predict_type == "query_attention_weights_by_value_norm":
            assert attn_weights_i is not None and attn_weights_i.dim() == 1, f"attn_weights should be 1D, but got {attn_weights_i.shape}"
            cur_layer_value_vectors = values_i.transpose(0, 1).flatten(1, 2)
            vector_norms = cur_layer_value_vectors.norm(2, dim=-1)
            weighted_vector_norms = attn_weights_i * vector_norms
            top_k_idxs = weighted_vector_norms.argsort(descending=True)[:top_k].tolist()
        elif predict_type == "attention_weights_sum":
            sum_weights = []
            for i in range(len(attn_weights_i)):
                weights = attn_weights_i[i:, i]
                sum_weights.append(weights.sum().item())
            top_k_idxs = sorted(range(len(sum_weights)), key=lambda x: sum_weights[x], reverse=True)[:top_k]
        elif predict_type == "attention_weights_sum_head_tail":
            sum_weights = []
            for i in range(len(attn_weights_i)):
                weights = attn_weights_i[i:, i]
                sum_weights.append(weights.sum().item())
            top_k_idxs = sorted(range(len(sum_weights)), key=lambda x: sum_weights[x], reverse=True)
            top_k_idxs = top_k_idxs[:top_k//2] + top_k_idxs[-top_k//2:]
        elif predict_type == "attention_weights_sum_per_image":
            sum_weights = []
            for i in range(len(attn_weights_i)):
                weights = attn_weights_i[i:i+258, i] # 258 is the number of tokens in an image
                sum_weights.append(weights.sum().item())
            top_k_idxs = sorted(range(len(sum_weights)), key=lambda x: sum_weights[x], reverse=True)[:top_k]
        elif predict_type == "attention_weights_sum_with_random":
            sum_weights = []
            for i in range(len(attn_weights_i)):
                weights = attn_weights_i[i:, i]
                sum_weights.append(weights.sum().item())
            top_k_idxs = sorted(range(len(sum_weights)), key=lambda x: sum_weights[x], reverse=True)
            top_k_idxs = top_k_idxs[:top_k//2]
            random_top_k_idxs = list(set(list(range(len(sum_weights)))) - set(top_k_idxs))
            random_top_k_idxs = random.sample(random_top_k_idxs, min(top_k//2, len(random_top_k_idxs)))
            top_k_idxs.extend(random_top_k_idxs)
        elif predict_type == "attention_weights_deduplication":
            # pivot:retained tokens = 1:32
            num_pivot_tokens = (top_k - 1) // 2 + 1
            sum_weights = []
            for i in range(len(attn_weights_i)):
                weights = attn_weights_i[i:, i]
                sum_weights.append(weights.sum().item())
            top_k_idxs = sorted(range(len(sum_weights)), key=lambda x: sum_weights[x], reverse=True)
            top_k_idxs, other_top_k_idxs = top_k_idxs[:num_pivot_tokens], top_k_idxs[num_pivot_tokens:]
            # select num_other_tokens from other_top_k_idxs by the lowest cosine similarity
            cur_layer_value_vectors = values_i.transpose(0, 1).flatten(1, 2)
            local_self_attn_value_vectors = cur_layer_value_vectors[:attn_weights_i.shape[0]]
            pivot_tokens_values = local_self_attn_value_vectors[top_k_idxs] # (P, C)
            other_tokens_values = local_self_attn_value_vectors[other_top_k_idxs] # (O, C)
            # Step 1: Normalize both sets of vectors
            pivot_tokens_normalized = F.normalize(pivot_tokens_values, p=2, dim=1)  # Normalize along embedding dimension
            other_tokens_normalized = F.normalize(other_tokens_values, p=2, dim=1)  # Normalize along embedding dimension

            # Step 2: Compute the cosine similarity matrix
            # This performs a matrix multiplication: (P, C) × (C, O) = (P, O)
            cosine_similarity_matrix = torch.matmul(pivot_tokens_normalized, other_tokens_normalized.transpose(0, 1))
            top_k_idxs.extend([other_top_k_idxs[j] for j in cosine_similarity_matrix.mean(dim=0).argsort()[:top_k - num_pivot_tokens]])

            # # select the num_pick_tokens from other_top_k_idxs for each pivot token
            # for i in range(len(top_k_idxs)):
            #     pivot_cosine_similarity = cosine_similarity_matrix[i]
            #     top_k_idxs.extend([other_top_k_idxs[j] for j in pivot_cosine_similarity.argsort()[:num_pivot_tokens]])
            top_k_idxs = list(set(top_k_idxs))
        elif predict_type == "vector_norms":
            cur_layer_value_vectors = values_i.transpose(0, 1).flatten(1, 2)
            vector_norms = cur_layer_value_vectors.norm(2, dim=-1)
            top_k_idxs = vector_norms.argsort(descending=True)[:top_k].tolist()
        elif predict_type == "vector_norms_small":
            cur_layer_value_vectors = values_i.transpose(0, 1).flatten(1, 2)
            vector_norms = cur_layer_value_vectors.norm(2, dim=-1)
            top_k_idxs = vector_norms.argsort(descending=False)[:top_k].tolist()
        elif predict_type == "key_norms":
            cur_layer_key_vectors = keys_i.transpose(0, 1).flatten(1, 2)
            key_norms = cur_layer_key_vectors.norm(2, dim=-1)
            top_k_idxs = key_norms.argsort(descending=True)[:top_k].tolist()
        elif predict_type == "key_norms_small":
            cur_layer_key_vectors = keys_i.transpose(0, 1).flatten(1, 2)
            key_norms = cur_layer_key_vectors.norm(2, dim=-1)
            top_k_idxs = key_norms.argsort(descending=False)[:top_k].tolist()
        elif predict_type == "key_norms_small_random":
            # half of the top_k tokens are selected by the highest key norms, and the other half are randomly selected
            cur_layer_key_vectors = keys_i.transpose(0, 1).flatten(1, 2)
            key_norms = cur_layer_key_vectors.norm(2, dim=-1)
            sorted_idxs = key_norms.argsort(descending=False)
            top_k_idxs = sorted_idxs[:top_k//2].tolist()
            random_top_k_idxs = sorted_idxs[top_k//2:].tolist()
            random_top_k_idxs = random.sample(random_top_k_idxs, min(top_k//2, len(random_top_k_idxs)))
            top_k_idxs.extend(random_top_k_idxs)
        elif predict_type == "random":
            top_k_idxs = random.sample(range(k_len), top_k)
            if 0 not in top_k_idxs:
                top_k_idxs.append(0)
        elif predict_type == "key_norms_small_deduplication":
            num_pivot_tokens = (top_k - 1) // 16 + 1
            key_vectors = keys_i.transpose(0, 1).flatten(1, 2)
            key_norms = key_vectors.norm(2, dim=-1)
            sorted_idxs = key_norms.argsort(descending=False)
            top_k_idxs = sorted_idxs[:num_pivot_tokens].tolist()
            other_top_k_idxs = sorted_idxs[num_pivot_tokens:].tolist()
            # select num_other_tokens from other_top_k_idxs by the lowest cosine similarity
            # keys_i: (num_heads, Q_len, C)
            normalized_key_vectors = F.normalize(key_vectors, p=2, dim=-1)
            pivot_key_vectors = normalized_key_vectors[top_k_idxs] # (P, C)
            other_key_vectors = normalized_key_vectors[other_top_k_idxs]
            cosine_similarity_matrix = torch.matmul(pivot_key_vectors, other_key_vectors.transpose(0, 1))
            top_k_idxs.extend([other_top_k_idxs[j] for j in cosine_similarity_matrix.mean(dim=0).argsort()[:top_k - num_pivot_tokens]])
            top_k_idxs = list(set(top_k_idxs))
        elif predict_type == "key_weighted_vector_norms":
            cur_layer_key_vectors = keys_i.transpose(0, 1).flatten(1, 2)
            key_norms = cur_layer_key_vectors.norm(2, dim=-1)
            # softmax the key norms
            key_norms = F.softmax(key_norms, dim=-1)
            cur_layer_value_vectors = values_i.transpose(0, 1).flatten(1, 2)
            value_norms = cur_layer_value_vectors.norm(2, dim=-1)
            weighted_norms = key_norms * value_norms
            top_k_idxs = weighted_norms.argsort(descending=True)[:top_k].tolist()
        elif predict_type == "output_norms":
            outputs_norms = outputs_i.norm(2, dim=-1)
            top_k_idxs = outputs_norms.argsort(descending=True)[:top_k].tolist()
        elif predict_type == "weighted_norms":
            weights = attn_weights_i # (Q_len, K_len)
            cur_layer_value_vectors = values_i.transpose(0, 1).flatten(1, 2) # (K_len, C)
            all_weighted_norms = []
            for q_i in range(len(weights)):
                cur_weights = weights[q_i]
                weighted_vectors = cur_weights.unsqueeze(-1) * cur_layer_value_vectors
                weighted_norms = weighted_vectors.norm(2, dim=-1)
                all_weighted_norms.append(weighted_norms)
            all_weighted_norms = torch.stack(all_weighted_norms, dim=0).mean(dim=0)
            top_k_idxs = all_weighted_norms.argsort(descending=True)[:top_k].tolist()
        else:
            raise ValueError(f"Unknown predict type: {predict_type}")
        bz_top_k_idxs.append(top_k_idxs)
    bz_top_k_idxs = torch.tensor(bz_top_k_idxs, device=values.device)
    top_k_select_mask = torch.zeros(bz, k_len, dtype=torch.bool, device=values.device)
    top_k_select_mask.scatter_(1, bz_top_k_idxs, 1)    
    return top_k_select_mask


def post_process_kv_cache(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.LongTensor,
    cache_position: torch.Tensor,
    position_embeddings: torch.Tensor,
    attn_weights: torch.Tensor,
    present_key_value: Union[Tuple[torch.Tensor, torch.Tensor], DynamicCache, LVUCache],
    lvu_layer_config: LVULayerConfig,
):
    """
    Args:
        hidden_states: (bz, Q_len, C)
        attn_weights: (bz, 1, Q_len, K_len)
        position_ids: (bz, Q_len)
        cache_position: (bz, Q_len)
        position_embeddings: (bz, Q_len, C)
        present_key_value: keys and values: ((bz, num_heads, K_len, C), (bz, num_heads, K_len, C))
        values: 
        top_k: int
        predict_type: str
    Returns:
        hidden_states: (bz, top_k, C)
        attention_mask: (bz, top_k) or None
        position_ids: (bz, top_k) or None
        cache_position: (bz, top_k) or None
        position_embeddings: (bz, top_k, C) or None
        present_key_value: ((bz, num_heads, top_k, C), (bz, num_heads, top_k, C))
        
    """
    if lvu_layer_config is None:
        return hidden_states, attention_mask, position_ids, cache_position, position_embeddings, present_key_value
    lvu_config = lvu_layer_config.lvu_config
    
    top_k = lvu_config.top_k
    top_p = lvu_config.top_p
    predict_type = lvu_config.top_k_predict_type
    layer_idx = lvu_layer_config.layer_idx
    prune_for_next_layer = lvu_layer_config.prune_for_next_layer
    q_len = hidden_states.shape[1]
    if isinstance(present_key_value, LVUCache) and present_key_value.prompt_length > 0:
        q_len -= present_key_value.prompt_length
        attn_weights = present_key_value.accum_attn_scores[layer_idx][-1]
        
    if top_p is not None and top_p >= 0:
        top_k = min((top_k or q_len), int(q_len * top_p))
        
    if not lvu_config.top_k_decay_type:
        top_k = top_k
    elif lvu_config.top_k_decay_type == "linear":
        top_k = top_k - int(top_k * (layer_idx / lvu_layer_config.total_layers))
    elif lvu_config.top_k_decay_type == "exponential":
        top_k = int(top_k * (lvu_config.top_k_decay_factor ** layer_idx))
    else:
        raise ValueError(f"Unknown top_k_decay_type: {lvu_config.top_k_decay_type}")
    if not lvu_config.enable or not top_k or top_k <= 0 or q_len <= top_k or \
        (isinstance(lvu_config.top_k_starting_layer, int) and lvu_config.top_k_starting_layer > 0 and lvu_config.layer_idx < lvu_config.top_k_starting_layer):
        # no need to prune
        return hidden_states, attention_mask, position_ids, cache_position, position_embeddings, present_key_value
    
    if isinstance(present_key_value, DynamicCache):
        keys, values = present_key_value[layer_idx]
    elif isinstance(present_key_value, tuple):
        keys, values = present_key_value
    else:
        raise ValueError(f"Unknown present_key_value type: {type(present_key_value)}")
    bz = keys.shape[0]
    assert bz == 1, f"Only support batch size 1 for now, but got {bz}"
    
    # only process the current new k
    past_keys = keys[:, :, :-q_len]
    past_values = values[:, :, :-q_len]
    keys = keys[:, :, -q_len:]
    values = values[:, :, -q_len:]
    
    top_k_select_mask = get_top_k_mask_to_predict(attn_weights, keys, values, hidden_states[:, :q_len], top_k=top_k, predict_type=predict_type)
    
    top_k_keys_list = []
    top_k_values_list = []
    top_k_hidden_states_list = [] if prune_for_next_layer else None
    top_k_attention_mask_list = [] if prune_for_next_layer else None
    top_k_position_ids_list = [] if prune_for_next_layer else None
    top_k_cache_position_list = [] if prune_for_next_layer else None
    top_k_position_embeddings_list = [] if prune_for_next_layer else None
    for bz_i in range(bz):
        top_k_select_mask_i = top_k_select_mask[bz_i]
        indices = torch.nonzero(top_k_select_mask_i, as_tuple=True)[0].cpu()
        assert len(indices) == top_k, f"top_k_select_mask_i: {top_k_select_mask_i}, indices: {indices}"
        
        bz_top_k_keys = keys[bz_i][:, indices]
        bz_top_k_values = values[bz_i][:, indices]
        top_k_keys_list.append(bz_top_k_keys)
        top_k_values_list.append(bz_top_k_values)
        
        if prune_for_next_layer:
            bz_top_k_hidden_states = hidden_states[bz_i][indices]
            bz_top_k_cache_position = cache_position[indices]
            if position_ids.dim() == 3:
                # (constant, bz, q_len)
                bz_top_k_position_ids = position_ids[:, bz_i][:, indices]
            elif position_ids.dim() == 2:
                # (bz, q_len)
                bz_top_k_position_ids = position_ids[bz_i][indices]
            if isinstance(position_embeddings, tuple):
                bz_top_k_position_embeddings = []
                for x in position_embeddings:
                    if x.dim() == 4:
                        # (constant, bz, q_len, c)
                        bz_top_k_position_embeddings.append(x[:, bz_i][:, indices])
                    elif x.dim() == 3:
                        # (bz, q_len, c)
                        bz_top_k_position_embeddings.append(x[bz_i][indices])
                    else:
                        raise ValueError(f"Unknown position_embeddings shape: {x.shape}")
                bz_top_k_position_embeddings = tuple(bz_top_k_position_embeddings)
            elif position_embeddings.dim() == 3:
                # (bz, q_len, c)
                bz_top_k_position_embeddings = position_embeddings[bz_i][indices]
            else:
                raise ValueError(f"Unknown position_embeddings type: {type(position_embeddings)}")
            if attention_mask is not None:
                if attention_mask.dim() == 2:
                    bz_top_k_attention_mask = attention_mask[bz_i][indices]
                elif attention_mask.dim() == 4:
                    bz_top_k_attention_mask = attention_mask[bz_i][:, indices, indices]
                else:
                    raise ValueError(f"Unknown attention_mask shape: {attention_mask.shape}")
            else:
                bz_top_k_attention_mask = None
            top_k_hidden_states_list.append(bz_top_k_hidden_states)
            top_k_attention_mask_list.append(bz_top_k_attention_mask)
            top_k_position_ids_list.append(bz_top_k_position_ids)
            top_k_cache_position_list.append(bz_top_k_cache_position)
            top_k_position_embeddings_list.append(bz_top_k_position_embeddings)
            
    top_k_keys = torch.stack(top_k_keys_list, dim=0)
    top_k_values = torch.stack(top_k_values_list, dim=0)
    keys = torch.cat([past_keys, top_k_keys], dim=2)
    values = torch.cat([past_values, top_k_values], dim=2)
    if isinstance(present_key_value, DynamicCache):
        # present_key_value.update(keys, values, layer_idx)
        present_key_value.key_cache[layer_idx] = keys
        present_key_value.value_cache[layer_idx] = values
    else:
        present_key_value = (keys, values)
    
    if prune_for_next_layer:
        hidden_states = torch.stack(top_k_hidden_states_list, dim=0)
        if not top_k_attention_mask_list or None in top_k_attention_mask_list:
            attention_mask = None
        else:
            attention_mask = torch.stack(top_k_attention_mask_list, dim=0)
        if position_ids.dim() == 3:
            position_ids = torch.stack(top_k_position_ids_list, dim=1)
        elif position_ids.dim() == 2:
            position_ids = torch.stack(top_k_position_ids_list, dim=0)
        cache_position = top_k_cache_position_list[0]
        
        if isinstance(position_embeddings, tuple):
            new_position_embeddings = []
            for i in range(len(position_embeddings)):
                if position_embeddings[i].dim() == 4:
                    # (constant, bz, q_len, c), stack in the batch dim
                    new_position_embeddings.append(torch.stack([x[i] for x in top_k_position_embeddings_list], dim=1))
                elif position_embeddings[i].dim() == 3:
                    # (bz, q_len, c), stack in the batch dim
                    new_position_embeddings.append(torch.stack([x[i] for x in top_k_position_embeddings_list], dim=0))
                else:
                    raise ValueError(f"Unknown position_embeddings shape: {position_embeddings[i].shape}")
            position_embeddings = tuple(new_position_embeddings)
        elif position_embeddings.dim() == 3:
            # (bz, q_len, c), stack in the batch dim
            position_embeddings = torch.stack(top_k_position_embeddings_list, dim=0)
        else:
            raise ValueError(f"Unknown position_embeddings type: {type(position_embeddings)}")
        
    # print(f"Reduced keys and values from {old_k_shape} to {top_k_keys.shape} for layer {layer_idx}")
    
    return hidden_states, attention_mask, position_ids, cache_position, position_embeddings, present_key_value

from transformers.models.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessorKwargs, BatchFeature

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

        fps = output_kwargs["videos_kwargs"].pop("fps", 2.0)
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