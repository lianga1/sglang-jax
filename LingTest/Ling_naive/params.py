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


if __name__ == "__main__":
    import os
    cfg = model_lib.ModelConfig.ling_minimal()
    # 使用相对于项目根目录的路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    model_path = os.path.join(project_root, "inclusionAI/Ling-mini-2.0")
    print(f"Loading model from: {model_path}")
    model = create_model_from_safe_tensors(model_path, cfg)
    print("Model created successfully.")

'''
import gc
import re
from enum import Enum

import jax
import safetensors
from etils import epath
from flax import nnx

import modeling as model_lib

def _get_key_and_transform_mapping(cfg: model_lib.ModelConfig):
    """
    Ling模型权重映射函数。
    
    Ling模型的safetensors权重格式:
    - model.word_embeddings.weight  (词嵌入)
    - model.norm.weight  (最终层归一化)
    - lm_head.weight
    - model.layers.{N}.attention.query_key_value.weight  (融合的QKV投影)
    - model.layers.{N}.attention.dense.weight  (输出投影)
    - model.layers.{N}.attention.query_layernorm.weight
    - model.layers.{N}.attention.key_layernorm.weight
    - model.layers.{N}.input_layernorm.weight
    - model.layers.{N}.post_attention_layernorm.weight
    
    Dense层(第0层):
    - model.layers.0.mlp.gate_proj.weight
    - model.layers.0.mlp.up_proj.weight
    - model.layers.0.mlp.down_proj.weight
    
    MoE层(其他层):
    - model.layers.{N}.mlp.gate.weight  (路由门控)
    - model.layers.{N}.mlp.gate.expert_bias  (专家偏置)
    - model.layers.{N}.mlp.shared_experts.gate_proj.weight
    - model.layers.{N}.mlp.shared_experts.up_proj.weight
    - model.layers.{N}.mlp.shared_experts.down_proj.weight
    - model.layers.{N}.mlp.experts.{E}.gate_proj.weight  (MoE专家)
    - model.layers.{N}.mlp.experts.{E}.up_proj.weight
    - model.layers.{N}.mlp.experts.{E}.down_proj.weight
    """
    
    class Transform(Enum):
        """Transformations for model parameters"""
        
        BIAS = None
        LINEAR = ((1, 0), None, False)
        EMBED = None
        SCALE = None
        
        # Ling特有的融合QKV变换
        # 输入: [(num_heads + 2 * num_kv_heads) * head_dim, hidden_size]
        # 需要拆分为Q, K, V并分别reshape
        ATTN_QKV = "QKV_FUSED"
        
        # 输出投影变换
        ATTN_OUT = ((1, 0), (cfg.num_heads, cfg.head_dim, cfg.emb_dim), False)
        
        # 单独的Q/K投影 (如果有)
        ATTN_Q = ((2, 0, 1), (cfg.num_heads, cfg.head_dim, cfg.emb_dim), True)
        ATTN_KV = ((2, 0, 1), (cfg.num_kv_heads, cfg.head_dim, cfg.emb_dim), True)
        
        # MoE transforms
        MOE_GATE = None  # [emb_dim, num_experts] -> no transform needed
        MOE_GATE_TRANSPOSE = ((1, 0), None, False)  # [num_experts, emb_dim] -> [emb_dim, num_experts]
        MOE_EXPERT_BIAS = None  # [num_experts] -> no transform needed
        MOE_SHARED_EXPERT_GATE_UP = ((1, 0), None, False)  # 需要transpose
        MOE_SHARED_EXPERT_DOWN = ((1, 0), None, False)  # 需要transpose
        
        # MoE Expert 权重变换 (单个专家)
        MOE_EXPERT_GATE_UP = ((1, 0), None, False)  # [intermediate, hidden] -> [hidden, intermediate]
        MOE_EXPERT_DOWN = ((1, 0), None, False)  # [hidden, intermediate] -> [intermediate, hidden]

    # Ling模型的映射 (torch_keys -> (nnx_keys, transform))
    mapping = {
        # 词嵌入
        r"model\.word_embeddings\.weight": ("embedder.embedding", Transform.EMBED),
        
        # 最终层归一化
        r"model\.norm\.weight": ("final_norm.scale", Transform.SCALE),
        
        # lm_head
        r"lm_head\.weight": ("lm_head.w", Transform.LINEAR),
        
        # 注意力层 - 融合的QKV
        r"model\.layers\.([0-9]+)\.attention\.query_key_value\.weight": (
            r"layers.\1.attn.qkv_proj",  # 需要特殊处理拆分
            Transform.ATTN_QKV,
        ),
        
        # 注意力层 - 输出投影
        r"model\.layers\.([0-9]+)\.attention\.dense\.weight": (
            r"layers.\1.attn.o_proj.w", 
            Transform.ATTN_OUT,
        ),
        
        # 注意力层 - LayerNorm
        r"model\.layers\.([0-9]+)\.attention\.query_layernorm\.weight": (
            r"layers.\1.attn.q_norm.scale", 
            Transform.SCALE,
        ),
        r"model\.layers\.([0-9]+)\.attention\.key_layernorm\.weight": (
            r"layers.\1.attn.k_norm.scale", 
            Transform.SCALE,
        ),
        
        # 层级LayerNorm
        r"model\.layers\.([0-9]+)\.input_layernorm\.weight": (
            r"layers.\1.input_layernorm.scale", 
            Transform.SCALE,
        ),
        r"model\.layers\.([0-9]+)\.post_attention_layernorm\.weight": (
            r"layers.\1.post_attention_layernorm.scale",
            Transform.SCALE,
        ),
        
        # Dense MLP (第0层) - MoEMLP 的 Param 直接存储，没有 .w 后缀
        r"model\.layers\.0\.mlp\.gate_proj\.weight": (
            r"layers.0.mlp.gate_proj", 
            Transform.MOE_SHARED_EXPERT_GATE_UP,
        ),
        r"model\.layers\.0\.mlp\.up_proj\.weight": (
            r"layers.0.mlp.up_proj", 
            Transform.MOE_SHARED_EXPERT_GATE_UP,
        ),
        r"model\.layers\.0\.mlp\.down_proj\.weight": (
            r"layers.0.mlp.down_proj", 
            Transform.MOE_SHARED_EXPERT_DOWN,
        ),
        
        # MoE门控 (非第0层)
        r"model\.layers\.([1-9][0-9]*)\.mlp\.gate\.weight": (
            r"layers.\1.mlp.gate.gate",
            Transform.MOE_GATE_TRANSPOSE,
        ),
        # expert_bias 暂时跳过，模型中没有此属性
        # r"model\.layers\.([1-9][0-9]*)\.mlp\.gate\.expert_bias": (
        #     None,  # 跳过
        #     Transform.MOE_EXPERT_BIAS,
        # ),
        
        # # MoE shared experts (非第0层) - 暂时跳过，模型中没有此属性
        # r"model\.layers\.([1-9][0-9]*)\.mlp\.shared_experts\.gate_proj\.weight": (
        #     None,  # 跳过
        #     Transform.MOE_SHARED_EXPERT_GATE_UP,
        # ),
        # r"model\.layers\.([1-9][0-9]*)\.mlp\.shared_experts\.up_proj\.weight": (
        #     None,  # 跳过
        #     Transform.MOE_SHARED_EXPERT_GATE_UP,
        # ),
        # r"model\.layers\.([1-9][0-9]*)\.mlp\.shared_experts\.down_proj\.weight": (
        #     None,  # 跳过
        #     Transform.MOE_SHARED_EXPERT_DOWN,
        # ),
        
        # MoE experts (非第0层) - 每个专家单独的权重
        # 需要特殊处理：聚合到 layers.{N}.mlp.gate_proj[E]
        r"model\.layers\.([1-9][0-9]*)\.mlp\.experts\.([0-9]+)\.gate_proj\.weight": (
            r"layers.\1.mlp.gate_proj.\2",  # 特殊处理
            Transform.MOE_EXPERT_GATE_UP,
        ),
        r"model\.layers\.([1-9][0-9]*)\.mlp\.experts\.([0-9]+)\.up_proj\.weight": (
            r"layers.\1.mlp.up_proj.\2",  # 特殊处理
            Transform.MOE_EXPERT_GATE_UP,
        ),
        r"model\.layers\.([1-9][0-9]*)\.mlp\.experts\.([0-9]+)\.down_proj\.weight": (
            r"layers.\1.mlp.down_proj.\2",  # 特殊处理
            Transform.MOE_EXPERT_DOWN,
        ),
    }

    return mapping


def _torch_key_to_jax_key(mapping, source_key):
    subs = [
        (re.sub(pat, repl, source_key), reshape)
        for pat, (repl, reshape) in mapping.items()
        if re.match(pat, source_key)
    ]
    if len(subs) == 0:
        raise ValueError(f"No mapping found for key: {source_key}")
    if len(subs) != 1:
        raise ValueError(f"Multiple mappings found for key {source_key}: {subs}")
    return subs[0]


def _assign_weights(keys, tensor, state_dict, st_key, transform, sharding_dict, cfg=None):
    """Recursively descend into state_dict and assign the (possibly permuted/reshaped) tensor."""
    key, *rest = keys
    if not rest:
        if transform is not None:
            # 检查是否是字符串类型的特殊变换
            if isinstance(transform, str):
                if transform == "QKV_FUSED":
                    # 融合的QKV权重需要特殊处理
                    # 输入形状: [hidden_size, (num_heads + 2 * num_kv_heads) * head_dim]
                    # 需要拆分为Q, K, V
                    raise ValueError(f"QKV_FUSED transform requires special handling in create_model_from_safe_tensors")
            elif hasattr(transform, 'name'):
                # Handle MoE-specific transformations by name
                if transform.name == "MOE_GATE":
                    # MoE gate weight: no transform needed
                    pass
                elif transform.name == "MOE_GATE_TRANSPOSE":
                    # MoE gate weight: [num_experts, emb_dim] -> [emb_dim, num_experts]
                    tensor = tensor.transpose(1, 0)
                elif transform.name == "MOE_EXPERT_BIAS":
                    # MoE expert bias: no transform needed
                    pass
                elif transform.name == "MOE_SHARED_EXPERT_GATE_UP":
                    # Shared expert gate_proj/up_proj: transpose
                    tensor = tensor.transpose(1, 0)
                elif transform.name == "MOE_SHARED_EXPERT_DOWN":
                    # Shared expert down_proj: transpose
                    tensor = tensor.transpose(1, 0)
                elif transform.name in ("BIAS", "EMBED", "SCALE"):
                    # No transform needed
                    pass
                else:
                    # 标准变换 for LINEAR, ATTN_OUT, ATTN_Q, ATTN_KV
                    if transform.value is not None and not isinstance(transform.value, str):
                        permute, reshape, reshape_first = transform.value
                        if reshape_first and reshape is not None:
                            tensor = tensor.reshape(reshape)
                        if permute:
                            tensor = tensor.transpose(permute)
                        if not reshape_first and reshape is not None:
                            tensor = tensor.reshape(reshape)
            else:
                # 元组格式的标准变换
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
        _assign_weights(rest, tensor, state_dict[key], st_key, transform, next_sharding, cfg)


def _stoi(s):
    try:
        return int(s)
    except ValueError:
        return s


def _split_qkv_weight(tensor, cfg: model_lib.ModelConfig):
    """
    拆分融合的QKV权重。
    
    输入形状: [(num_heads + 2 * num_kv_heads) * head_dim, hidden_size]
    
    模型期望的形状 (Einsum w 属性):
    - Q: (emb_dim, num_heads, head_dim) = (2048, 16, 128)
    - K: (emb_dim, num_kv_heads, head_dim) = (2048, 4, 128)
    - V: (emb_dim, num_kv_heads, head_dim) = (2048, 4, 128)
    """
    num_heads = cfg.num_heads
    num_kv_heads = cfg.num_kv_heads
    head_dim = cfg.head_dim
    hidden_size = cfg.emb_dim
    
    # 输入: [(num_heads + 2 * num_kv_heads) * head_dim, hidden_size]
    # 例如: [(16 + 2*4) * 128, 2048] = [3072, 2048]
    
    q_dim = num_heads * head_dim
    kv_dim = num_kv_heads * head_dim
    
    # 拆分 Q, K, V (沿第一个维度)
    q_weight = tensor[:q_dim, :]  # [num_heads * head_dim, hidden_size]
    k_weight = tensor[q_dim:q_dim + kv_dim, :]  # [num_kv_heads * head_dim, hidden_size]
    v_weight = tensor[q_dim + kv_dim:, :]  # [num_kv_heads * head_dim, hidden_size]
    
    # Reshape到目标格式
    # Q: [num_heads * head_dim, hidden_size] -> [num_heads, head_dim, hidden_size] -> [hidden_size, num_heads, head_dim]
    q_weight = q_weight.reshape(num_heads, head_dim, hidden_size).transpose(2, 0, 1)
    # K: [num_kv_heads * head_dim, hidden_size] -> [num_kv_heads, head_dim, hidden_size] -> [hidden_size, num_kv_heads, head_dim]
    k_weight = k_weight.reshape(num_kv_heads, head_dim, hidden_size).transpose(2, 0, 1)
    # V: [num_kv_heads * head_dim, hidden_size] -> [num_kv_heads, head_dim, hidden_size] -> [hidden_size, num_kv_heads, head_dim]
    v_weight = v_weight.reshape(num_kv_heads, head_dim, hidden_size).transpose(2, 0, 1)
    
    return q_weight, k_weight, v_weight


def create_model_from_safe_tensors(
    file_dir: str, cfg: model_lib.ModelConfig, mesh: jax.sharding.Mesh | None = None
) -> model_lib.Qwen3:
    """Load tensors from the safetensors file and create a Ling model (memory-optimized)."""
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
                
                try:
                    jax_key, transform = _torch_key_to_jax_key(key_mapping, torch_key)
                except ValueError:
                    # 没有匹配的映射，跳过
                    conversion_errors.append(f"No mapping found for '{torch_key}'")
                    continue
                    
                if jax_key is None:
                    continue
                
                # 特殊处理融合的QKV权重
                if transform.value == "QKV_FUSED":
                    # 提取层号
                    match = re.match(r"model\.layers\.([0-9]+)\.attention\.query_key_value\.weight", torch_key)
                    if match:
                        layer_idx = int(match.group(1))
                        try:
                            q_weight, k_weight, v_weight = _split_qkv_weight(tensor, cfg)
                            
                            # 分配Q权重
                            q_keys = ["layers", layer_idx, "attn", "q_proj", "w"]
                            _assign_weights(q_keys, q_weight, state_dict, torch_key + ".q", None, sharding, cfg)
                            
                            # 分配K权重
                            k_keys = ["layers", layer_idx, "attn", "k_proj", "w"]
                            _assign_weights(k_keys, k_weight, state_dict, torch_key + ".k", None, sharding, cfg)
                            
                            # 分配V权重
                            v_keys = ["layers", layer_idx, "attn", "v_proj", "w"]
                            _assign_weights(v_keys, v_weight, state_dict, torch_key + ".v", None, sharding, cfg)
                        except Exception as e:
                            conversion_errors.append(
                                f"Failed to split QKV for '{torch_key}': {type(e).__name__}: {e}"
                            )
                    continue
                
                keys = [_stoi(k) for k in jax_key.split(".")]
                try:
                    _assign_weights(keys, tensor, state_dict, torch_key, transform.value, sharding, cfg)
                except Exception as e:
                    full_jax_key = ".".join([str(k) for k in keys])
                    conversion_errors.append(
                        f"Failed to assign '{torch_key}' to '{full_jax_key}': {type(e).__name__}: {e}"
                    )
        gc.collect()

    if conversion_errors:
        full_error_log = "\n".join(conversion_errors)
        print(f"Warning: Encountered {len(conversion_errors)} weight conversion issues. Log:\n{full_error_log}")

    if cfg.tie_word_embeddings:
        state_dict["lm_head"]["w"] = state_dict["embedder"]["embedding"].T
    gc.collect()
    return nnx.merge(graph_def, state_dict)



'''