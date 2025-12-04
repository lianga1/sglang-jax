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
        # MoE transforms
        MOE_GATE = None  # [emb_dim, num_experts] -> no transform needed
        MOE_EXPERT_GATE_UP = None  # [num_experts, emb_dim, intermediate_dim] -> no transform needed
        MOE_EXPERT_DOWN = None  # [num_experts, intermediate_dim, emb_dim] -> no transform needed

    # Mapping of torch_keys -> (nnx_keys, (permute_rule, reshape_rule)).
    mapping = {
        r"model\.embed_tokens\.weight": ("embedder.embedding", Transform.EMBED),
        r"model\.layers\.([0-9]+)\.self_attn\.q_proj\.weight": (r"layers.\1.attn.q_proj.w", Transform.ATTN_Q),
        r"model\.layers\.([0-9]+)\.self_attn\.k_proj\.weight": (r"layers.\1.attn.k_proj.w", Transform.ATTN_KV),
        r"model\.layers\.([0-9]+)\.self_attn\.v_proj\.weight": (r"layers.\1.attn.v_proj.w", Transform.ATTN_KV),
        r"model\.layers\.([0-9]+)\.self_attn\.o_proj\.weight": (r"layers.\1.attn.o_proj.w", Transform.ATTN_OUT),
        # mlp (dense)
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

    # Add MoE mappings if MoE is enabled
    if cfg.num_experts > 1:
        mapping.update({
            # MoE gate (router) - shape: [emb_dim, num_experts]
            r"model\.layers\.([0-9]+)\.mlp\.gate\.weight": (
                r"layers.\1.mlp.gate.gate",
                Transform.MOE_GATE,
            ),
            # MoE expert gate_proj - shape: [num_experts, emb_dim, intermediate_dim]
            r"model\.layers\.([0-9]+)\.mlp\.gate_proj\.weight": (
                r"layers.\1.mlp.gate_proj.value",
                Transform.MOE_EXPERT_GATE_UP,
            ),
            # MoE expert up_proj - shape: [num_experts, emb_dim, intermediate_dim]
            r"model\.layers\.([0-9]+)\.mlp\.up_proj\.weight": (
                r"layers.\1.mlp.up_proj.value",
                Transform.MOE_EXPERT_GATE_UP,
            ),
            # MoE expert down_proj - shape: [num_experts, intermediate_dim, emb_dim]
            r"model\.layers\.([0-9]+)\.mlp\.down_proj\.weight": (
                r"layers.\1.mlp.down_proj.value",
                Transform.MOE_EXPERT_DOWN,
            ),
        })

    return mapping


def _torch_key_to_jax_key(mapping, source_key):
    subs = [
        (re.sub(pat, repl, source_key), reshape)
        for pat, (repl, reshape) in mapping.items()
        if re.match(pat, source_key)
    ]
    if len(subs) != 1:
        raise ValueError(f"Only one key should be found: {subs[0]}")
    return subs[0]


def _assign_weights(keys, tensor, state_dict, st_key, transform, sharding_dict):
    """Recursively descend into state_dict and assign the (possibly permuted/reshaped) tensor."""
    key, *rest = keys
    if not rest:
        if transform is not None:
            # Handle MoE-specific transformations
            if transform.name == "MOE_GATE":
                # MoE gate weight: [emb_dim, num_experts] -> no transform needed
                pass
            elif transform.name == "MOE_EXPERT_GATE_UP":
                # MoE expert gate_proj/up_proj: need to reshape from [num_experts, intermediate_dim, emb_dim] to [num_experts, emb_dim, intermediate_dim]
                # This is a transpose of last two dimensions
                tensor = tensor.transpose(0, 2, 1)
            elif transform.name == "MOE_EXPERT_DOWN":
                # MoE expert down_proj: [num_experts, intermediate_dim, emb_dim] -> reshape to [num_experts, intermediate_dim, emb_dim]
                # Already in correct shape
                pass
            else:
                # Standard transformations for dense models
                permute, reshape, reshape_first = transform
                if reshape_first and reshape is not None:
                    tensor = tensor.reshape(reshape)
                if permute:
                    tensor = tensor.transpose(permute)
                if not reshape_first and reshape is not None:
                    tensor = tensor.reshape(reshape)
        if tensor.shape != state_dict[key].shape:
            raise ValueError(f"Shape mismatch for {st_key}: {tensor.shape} vs {state_dict[key].shape}")
        # Only apply sharding if sharding_dict is provided
        if sharding_dict is not None:
            state_dict[key] = jax.device_put(tensor, sharding_dict[key])
        else:
            state_dict[key] = jax.device_put(tensor)
    else:
        next_sharding = sharding_dict[key] if sharding_dict is not None else None
        _assign_weights(rest, tensor, state_dict[key], st_key, transform, next_sharding)


def _stoi(s):
    try:
        return int(s)
    except ValueError:
        return s


def create_model_from_safe_tensors(
    file_dir: str, cfg: model_lib.ModelConfig, mesh: jax.sharding.Mesh | None = None
) -> model_lib.Qwen3:
    """Load tensors from the safetensors file and create a Qwen3 model (memory-optimized)."""
    files = list(epath.Path(file_dir).expanduser().glob("*.safetensors"))
    if not files:
        raise ValueError(f"No safetensors found in {file_dir}")

    qwen3 = nnx.eval_shape(lambda: model_lib.Qwen3(cfg, rngs=nnx.Rngs(params=0)))
    graph_def, abs_state = nnx.split(qwen3)
    state_dict = abs_state.to_pure_dict()
    # Only use sharding if mesh is provided
    sharding = nnx.get_named_sharding(abs_state, mesh).to_pure_dict() if mesh is not None else None

    key_mapping = _get_key_and_transform_mapping(cfg)
    conversion_errors = []
    for f in files:
        with safetensors.safe_open(f, framework="numpy") as sf:
            for torch_key in sf.keys():
                tensor = sf.get_tensor(torch_key)

                jax_key, transform = _torch_key_to_jax_key(key_mapping, torch_key)
                if jax_key is None:
                    continue
                keys = [_stoi(k) for k in jax_key.split(".")]
                try:
                    _assign_weights(keys, tensor, state_dict, torch_key, transform.value, sharding)
                except Exception as e:
                    full_jax_key = ".".join([str(k) for k in keys])
                    conversion_errors.append(
                        f"Failed to assign '{torch_key}' to '{full_jax_key}': {type(e).__name__}: {e}"
                    )
        gc.collect()

    if conversion_errors:
        full_error_log = "\n".join(conversion_errors)
        raise RuntimeError(f"Encountered {len(conversion_errors)} weight conversion errors. Log:\n{full_error_log}")

    if cfg.tie_word_embeddings:
        state_dict["lm_head"]["w"] = state_dict["embedder"]["embedding"].T
    gc.collect()
    return nnx.merge(graph_def, state_dict)
