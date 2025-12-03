import gc
import re
from enum import Enum

import jax
import safetensors
from etils import epath
from flax import nnx

import modeling as model_lib

def _get_key_and_transform_mapping(cfg: model_lib.ModelConfig):
    class Transform(Enum):
        """Transformations for model parameters"""

        BIAS = None
        LINEAR = ((1, 0), None, False)
        EMBED = None
        ATTN_Q = ((2, 0, 1), (cfg.num_heads, cfg.head_dim, cfg.emb_dim), True)
        ATTN_KV = ((2, 0, 1), (cfg.num_kv_heads, cfg.head_dim, cfg.emb_dim), True)
        ATTN_OUT = ((1, 0), (cfg.num_heads, cfg.head_dim, cfg.emb_dim), False)
        SCALE = None

    # Mapping of torch_keys -> (nnx_keys, (permute_rule, reshape_rule)).
    return {
        r"model\.embed_tokens\.weight": ("embedder.embedding", Transform.EMBED),
        r"model\.layers\.([0-9]+)\.self_attn\.q_proj\.weight": (r"layers.\1.attn.q_proj.w", Transform.ATTN_Q),
        r"model\.layers\.([0-9]+)\.self_attn\.k_proj\.weight": (r"layers.\1.attn.k_proj.w", Transform.ATTN_KV),
        r"model\.layers\.([0-9]+)\.self_attn\.v_proj\.weight": (r"layers.\1.attn.v_proj.w", Transform.ATTN_KV),
        r"model\.layers\.([0-9]+)\.self_attn\.o_proj\.weight": (r"layers.\1.attn.o_proj.w", Transform.ATTN_OUT),
        # mlp
        r"model\.layers\.([0-9]+)\.mlp\.gate_proj\.weight": (r"layers.\1.mlp.gate_proj.kernel", Transform.LINEAR),
        r"model\.layers\.([0-9]+)\.mlp\.up_proj\.weight": (r"layers.\1.mlp.up_proj.kernel", Transform.LINEAR),
        r"model\.layers\.([0-9]+)\.mlp\.down_proj\.weight": (r"layers.\1.mlp.down_proj.kernel", Transform.LINEAR),
        r"model\.norm\.weight": ("final_norm.scale", Transform.SCALE),
        # norms
        r"model\.layers\.([0-9]+)\.self_attn\.q_norm\.weight": (r"layers.\1.attn.q_norm.scale", Transform.SCALE),
        r"model\.layers\.([0-9]+)\.self_attn\.k_norm\.weight": (r"layers.\1.attn.k_norm.scale", Transform.SCALE),
        # layer norms (pre/post attention)
        r"model\.layers\.([0-9]+)\.input_layernorm\.weight": (r"layers.\1.input_layernorm.scale", Transform.SCALE),
        r"model\.layers\.([0-9]+)\.post_attention_layernorm\.weight": (
            r"layers.\1.post_attention_layernorm.scale",
            Transform.SCALE,
        ),
        r"lm_head\.weight": ("lm_head.w", Transform.LINEAR),
    }
