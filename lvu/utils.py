import torch
import random
import torch.nn.functional as F
from typing import Tuple, Union
from .lvu import LVUConfig
from transformers.cache_utils import DynamicCache
def get_top_k_mask_to_predict(attn_weights, keys, values, outputs, top_k=100, predict_type="attention_weights"):
    """
    Args:
        attn_weights: (bz, 1, Q_len, K_len)
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
        attn_weights_i = attn_weights[bz_i].mean(0) if attn_weights is not None else None
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


def preprocess_hidden_states(
    hidden_states: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    attention_mask: torch.Tensor,
    position_ids: torch.LongTensor,
):
    if isinstance(hidden_states, tuple):
        hidden_states, attention_mask, position_ids = hidden_states
    else:
        hidden_states = hidden_states
        attention_mask = attention_mask
        position_ids = position_ids
    return hidden_states, attention_mask, position_ids

def post_process_kv_cache(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.LongTensor,
    attn_weights: torch.Tensor,
    present_key_value: Union[Tuple[torch.Tensor, torch.Tensor], DynamicCache],
    lvu_config: LVUConfig,
):
    """
    Args:
        hidden_states: (bz, Q_len, C)
        position_ids: (bz, Q_len)
        attn_weights: (bz, 1, Q_len, K_len)
        present_key_value: keys and values: ((bz, num_heads, K_len, C), (bz, num_heads, K_len, C))
        values: 
        top_k: int
        predict_type: str
    Returns:
        hidden_states: (bz, top_k, C)
        attention_mask: (bz, top_k) or None
        position_ids: (bz, top_k) or None
        present_key_value: ((bz, num_heads, top_k, C), (bz, num_heads, top_k, C))
        
    """
    top_k = lvu_config.top_k
    predict_type = lvu_config.top_k_predict_type
    layer_idx = lvu_config.layer_idx
    if lvu_config.prefill_prune_starting_layer is not None and lvu_config.layer_idx >= lvu_config.prefill_prune_starting_layer:
        prune_for_next_layer = True
    else:
        prune_for_next_layer = False
    
    if isinstance(present_key_value, DynamicCache):
        keys, values = present_key_value[layer_idx]
    elif isinstance(present_key_value, tuple):
        keys, values = present_key_value
    else:
        raise ValueError(f"Unknown present_key_value type: {type(present_key_value)}")
    bz, num_heads, k_len, _ = keys.shape
    q_len = hidden_states.shape[1]
    
    if not top_k or top_k <= 0 or q_len <= top_k or \
        (isinstance(lvu_config.top_k_starting_layer, int) and lvu_config.top_k_starting_layer > 0 and lvu_config.layer_idx < lvu_config.top_k_starting_layer):
        # no need to prune
        return hidden_states, attention_mask, position_ids, present_key_value
    
    # only process the current new k
    attn_weights = attn_weights[:, :, -q_len:] if attn_weights is not None else None
    past_keys = keys[:, :, :-q_len]
    past_values = values[:, :, :-q_len]
    keys = keys[:, :, -q_len:]
    values = values[:, :, -q_len:]
    
    top_k_select_mask = get_top_k_mask_to_predict(attn_weights, keys, values, hidden_states, top_k=top_k, predict_type=predict_type)
    
    top_k_keys_list = []
    top_k_values_list = []
    top_k_hidden_states_list = [] if prune_for_next_layer else None
    top_k_attention_mask_list = [] if prune_for_next_layer else None
    top_k_position_ids_list = [] if prune_for_next_layer else None
    for bz_i in range(bz):
        top_k_select_mask_i = top_k_select_mask[bz_i]
        indices = torch.nonzero(top_k_select_mask_i, as_tuple=True)[0]
        assert len(indices) == top_k, f"top_k_select_mask_i: {top_k_select_mask_i}, indices: {indices}"
        
        bz_top_k_keys = keys[bz_i][:, indices]
        bz_top_k_values = values[bz_i][:, indices]
        top_k_keys_list.append(bz_top_k_keys)
        top_k_values_list.append(bz_top_k_values)
        
        if prune_for_next_layer:
            bz_top_k_hidden_states = hidden_states[bz_i][indices]
            bz_top_k_position_ids = position_ids[bz_i][indices]
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
        attention_mask = torch.stack(top_k_attention_mask_list, dim=0)
        position_ids = torch.stack(top_k_position_ids_list, dim=0)
    
    return hidden_states, attention_mask, position_ids, present_key_value