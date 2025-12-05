import gc
import re
import os
from enum import Enum
import jax
import jax.numpy as jnp
import numpy as np
import safetensors.numpy as safetensors
from etils import epath
from flax import nnx
import modeling as model_lib




def _get_key_and_transform_mapping(cfg: model_lib.ModelConfig):
    class Transform(Enum):
        BIAS = None
        # PyTorch Linear 通常是 [out, in]，JAX 是 [in, out]，需要转置 (1, 0)
        LINEAR = ((1, 0), None, False) 
        EMBED = None
        # Attention: QKV 在这里定义基本规则，具体拆分在主循环逻辑中
        ATTN_QKV = ((2, 0, 1), (cfg.num_heads, cfg.head_dim, cfg.emb_dim), True)
        ATTN_OUT = ((1, 0), (cfg.num_heads, cfg.head_dim, cfg.emb_dim), False)
        SCALE = None
        
        # MoE Router: PyTorch [hidden, experts] -> JAX [hidden, experts] (通常不需要转置，视具体模型而定)
        # 你的结构显示 router: (2048, 64)，如果 PyTorch 也是 (2048, 64) 则为 None；如果是 (64, 2048) 则为 (1,0)
        # Qwen/DeepSeek 通常 router 不需要转置，或者视 safetensors 实际形状而定。
        # 这里暂定 MOE_ROUTER = None，如果报错形状不匹配，改为 ((1, 0), None, False)
        # MOE_ROUTER = None 
        MOE_ROUTER_BIAS = None
        MOE_ROUTER = ((1, 0), None, False)
        # MoE Experts: 单个专家的权重处理
        # PyTorch 单个专家 Gate/Up: [intermediate, hidden] -> JAX [hidden, intermediate] => 转置
        MOE_EXPERT_UP = ((1, 0), (cfg.moe_intermediate_dim, cfg.emb_dim), False)
        # PyTorch 单个专家 Down: [hidden, intermediate] -> JAX [intermediate, hidden] => 转置
        MOE_EXPERT_DOWN = ((1, 0), (cfg.emb_dim, cfg.moe_intermediate_dim), False)

    # === 基于你的 structure_dump 严格校对的 Mapping ===
    mapping = {
        # === 全局部分 ===
        r"model\.word_embeddings\.weight": ("embedder.embedding", Transform.EMBED),
        r"lm_head\.weight": ("lm_head.w", Transform.LINEAR), # [修正] 加回 .w
        r"model\.norm\.weight": ("final_norm.scale", Transform.SCALE),
        
        # === Attention & Norms (所有层通用) ===
        # 注意：Attention 的内部投影全都有 .w
        r"model\.layers\.([0-9]+)\.attention\.dense\.weight": (r"layers.\1.attn.o_proj.w", Transform.ATTN_OUT),
        # QKV 会被特殊逻辑拦截，这里的 target 仅作参考前缀
        r"model\.layers\.([0-9]+)\.attention\.query_key_value\.weight": (r"layers.\1.attn.qkv_proj", Transform.ATTN_QKV),
        
        # Norms
        r"model\.layers\.([0-9]+)\.attention\.query_layernorm\.weight": (r"layers.\1.attn.q_norm.scale", Transform.SCALE),
        r"model\.layers\.([0-9]+)\.attention\.key_layernorm\.weight": (r"layers.\1.attn.k_norm.scale", Transform.SCALE),
        r"model\.layers\.([0-9]+)\.input_layernorm\.weight": (r"layers.\1.input_layernorm.scale", Transform.SCALE),
        r"model\.layers\.([0-9]+)\.post_attention_layernorm\.weight": (r"layers.\1.post_attention_layernorm.scale", Transform.SCALE),
    }

    # === Layer 0: Dense MLP (Einsum 结构，有 .w) ===
    mapping.update({
        r"model\.layers\.0\.mlp\.gate_proj\.weight": (r"layers.0.mlp.gate_proj.w", Transform.LINEAR), # [修正] 加回 .w
        r"model\.layers\.0\.mlp\.up_proj\.weight": (r"layers.0.mlp.up_proj.w", Transform.LINEAR),     # [修正] 加回 .w
        r"model\.layers\.0\.mlp\.down_proj\.weight": (r"layers.0.mlp.down_proj.w", Transform.LINEAR), # [修正] 加回 .w
    })

    # === Layer 1-19: MoE layers ===
    
    # 1. Shared Expert (Einsum 结构，有 .w)
    mapping.update({
        r"model\.layers\.([1-9]|1[0-9])\.mlp\.shared_experts\.gate_proj\.weight": (r"layers.\1.mlp.shared_expert.gate_proj.w", Transform.LINEAR), # [修正] 路径补全+加 .w
        r"model\.layers\.([1-9]|1[0-9])\.mlp\.shared_experts\.up_proj\.weight": (r"layers.\1.mlp.shared_expert.up_proj.w", Transform.LINEAR),     # [修正] 路径补全+加 .w
        r"model\.layers\.([1-9]|1[0-9])\.mlp\.shared_experts\.down_proj\.weight": (r"layers.\1.mlp.shared_expert.down_proj.w", Transform.LINEAR), # [修正] 路径补全+加 .w
    })

    # 2. Router (nnx.Param 结构，无 .w，直接是叶子)
    mapping.update({
        r"model\.layers\.([1-9]|1[0-9])\.mlp\.gate\.weight": (r"layers.\1.mlp.router", Transform.MOE_ROUTER), # [修正] 改名 gate->router，无 .w
        # 你的结构里 router 没有 bias，如果源文件有 bias，这个映射会报错 KeyNotFound。
        # 如果确定不需要 bias，可以在循环里把 expert_bias 的处理注释掉，或者保留这里但它会因为匹配不到 JAX key 而被忽略/报错
        # r"model\.layers\.([1-9]|1[0-9])\.mlp\.gate\.expert_bias": (r"layers.\1.mlp.router.bias", Transform.MOE_ROUTER_BIAS), 
    })

    # 3. Routed Experts (nnx.Param 结构，直接是堆叠的大张量，无 .w)
    # 目标键名直接指向 layers.X.mlp.experts_gate_proj
    mapping.update({
        r"model\.layers\.([1-9]|1[0-9])\.mlp\.experts\.([0-9]+)\.gate_proj\.weight": (r"layers.\1.mlp.experts_gate_proj", Transform.MOE_EXPERT_UP),
        r"model\.layers\.([1-9]|1[0-9])\.mlp\.experts\.([0-9]+)\.up_proj\.weight": (r"layers.\1.mlp.experts_up_proj", Transform.MOE_EXPERT_UP),
        r"model\.layers\.([1-9]|1[0-9])\.mlp\.experts\.([0-9]+)\.down_proj\.weight": (r"layers.\1.mlp.experts_down_proj", Transform.MOE_EXPERT_DOWN),
    })

    return mapping, Transform


def _find_matching_rule(mapping, source_key):
    for pat, (repl, transform) in mapping.items():
        match = re.fullmatch(pat, source_key)
        if match:
            return repl, transform, match
    return None, None, None


def _assign_weights(keys, tensor, state_dict, st_key, transform, sharding_dict):
    key, *rest = keys
    
    if key not in state_dict:
        raise KeyError(f"Key '{key}' not found. Current path segment: {st_key}. Available: {list(state_dict.keys())}")

    if not rest:
        # 这里的 state_dict[key] 应该是 ShapeDtypeStruct (叶子)
        
        # Transform Logic
        if transform is not None and transform.value is not None:
            permute, reshape, reshape_first = transform.value
            if reshape_first and reshape is not None:
                tensor = tensor.reshape(reshape)
            if permute:
                tensor = tensor.transpose(permute)
            if not reshape_first and reshape is not None:
                tensor = tensor.reshape(reshape)
        
        target_shape = state_dict[key].shape
        if tensor.shape != target_shape:
            # 自动处理 Transpose (补救措施)
            if tensor.shape[::-1] == target_shape:
                print(f"  [Auto-Transpose] {st_key}: {tensor.shape} -> {target_shape}")
                tensor = tensor.T
            else:
                raise ValueError(f"Shape mismatch for {st_key}: Loaded {tensor.shape} vs Expected {target_shape}")
        
        if sharding_dict is not None:
            state_dict[key] = jax.device_put(tensor, sharding_dict[key])
        else:
            state_dict[key] = jax.device_put(tensor)
    else:
        current_node = state_dict[key]
        if hasattr(current_node, 'shape') and not isinstance(current_node, dict):
             raise TypeError(f"Path too deep! Attempted to access '{rest[0]}' inside '{key}', but '{key}' is already a leaf node. Check mapping for .w suffix issues.")
             
        next_sharding = sharding_dict[key] if sharding_dict is not None else None
        _assign_weights(rest, tensor, current_node, st_key, transform, next_sharding)


def _assign_partial_expert(keys, sub_tensor, expert_idx, state_dict, transform, sharding_dict=None):
    node = state_dict
    sharding_node = sharding_dict
    
    # 这里的 keys 应该直接指向 experts_gate_proj (叶子)
    # 例如: ['layers', '1', 'mlp', 'experts_gate_proj']
    
    path_debug = []
    for k in keys[:-1]:
        if k not in node:
             raise KeyError(f"MoE Path Error: '{k}' not found. Path: {path_debug}. Available: {list(node.keys())}")
        node = node[k]
        path_debug.append(k)
        if sharding_node: sharding_node = sharding_node[k]
    
    last_key = keys[-1]
    if last_key not in node:
         raise KeyError(f"MoE Leaf Error: '{last_key}' not found. Path: {path_debug}. Available: {list(node.keys())}")

    target_tensor = node[last_key] # 这应该是 ShapeDtypeStruct

    # Transform (Individual Expert)
    if transform is not None and transform.value is not None:
        permute, reshape, reshape_first = transform.value
        if permute:
            sub_tensor = sub_tensor.transpose(permute)

    # Lazy Initialization
    if hasattr(target_tensor, 'shape') and not isinstance(target_tensor, (np.ndarray, jax.Array)):
        full_shape = target_tensor.shape
        dtype = target_tensor.dtype
        # print(f"  [Init MoE Buffer] {'.'.join(str(k) for k in keys)} shape={full_shape}")
        node[last_key] = np.zeros(full_shape, dtype=dtype)
        target_tensor = node[last_key]

    # Assign slice
    # JAX MoE tensor shape: [num_experts, ..., ...]
    # 所以第一个维度是 expert_idx
    try:
        target_tensor[expert_idx] = sub_tensor
    except IndexError:
        raise IndexError(f"Expert Index {expert_idx} out of bounds for tensor with shape {target_tensor.shape}")
    except ValueError as e:
         raise ValueError(f"MoE Assignment Error for Expert {expert_idx}: Target slice shape {target_tensor[expert_idx].shape} vs Source {sub_tensor.shape}. Error: {e}")


def _stoi(s):
    try:
        return int(s)
    except ValueError:
        return s


def create_model_from_safe_tensors(file_dir: str, cfg: model_lib.ModelConfig, mesh: jax.sharding.Mesh | None = None):
    files = list(epath.Path(file_dir).expanduser().glob("*.safetensors"))
    if not files:
        raise ValueError(f"No safetensors found in {file_dir}")

    print("Initializing model skeleton...")
    ling2_mini = nnx.eval_shape(lambda: model_lib.Ling2_mini(cfg, rngs=nnx.Rngs(params=0)))
    graph_def, abs_state = nnx.split(ling2_mini)
    state_dict = abs_state.to_pure_dict()
    
    sharding = nnx.get_named_sharding(abs_state, mesh).to_pure_dict() if mesh is not None else None
    mapping, TransformEnum = _get_key_and_transform_mapping(cfg)
    conversion_errors = []

    print(f"Start loading weights from {len(files)} files...")
    
    for f in files:
        with safetensors.safe_open(f, framework="numpy") as sf:
            for torch_key in sf.keys():
                tensor = sf.get_tensor(torch_key)
                repl_pattern, transform, match = _find_matching_rule(mapping, torch_key)
                
                if not match: 
                    # 忽略 bias 报错
                    if "bias" in torch_key and "expert_bias" in torch_key:
                        continue
                    continue

                # Case 1: QKV Splitting
                if "query_key_value" in torch_key:
                    layer_idx = int(match.group(1))
                    head_dim = cfg.head_dim
                    num_heads = cfg.num_heads
                    num_kv_heads = cfg.num_kv_heads
                    q_dim = num_heads * head_dim
                    kv_dim = num_kv_heads * head_dim
                    
                    q, k, v = np.split(tensor, [q_dim, q_dim + kv_dim], axis=0)
                    
                    q_path = ["layers", layer_idx, "attn", "q_proj", "w"]
                    k_path = ["layers", layer_idx, "attn", "k_proj", "w"]
                    v_path = ["layers", layer_idx, "attn", "v_proj", "w"]
                    
                    q_trans = (TransformEnum.ATTN_QKV.value[0], (cfg.num_heads, cfg.head_dim, cfg.emb_dim), True)
                    kv_trans = (TransformEnum.ATTN_QKV.value[0], (cfg.num_kv_heads, cfg.head_dim, cfg.emb_dim), True)

                    class TempTransform: pass
                    t_q = TempTransform(); t_q.value = q_trans
                    t_kv = TempTransform(); t_kv.value = kv_trans

                    try:
                        _assign_weights(q_path, q, state_dict, torch_key + "_Q", t_q, sharding)
                        _assign_weights(k_path, k, state_dict, torch_key + "_K", t_kv, sharding)
                        _assign_weights(v_path, v, state_dict, torch_key + "_V", t_kv, sharding)
                    except Exception as e:
                        conversion_errors.append(f"QKV Split Error {torch_key}: {e}")
                    continue

                # Case 2: MoE Experts
                if len(match.groups()) == 2 and "experts" in torch_key and "shared" not in torch_key:
                    layer_idx = int(match.group(1))
                    expert_idx = int(match.group(2))
                    jax_key_str = match.expand(repl_pattern)
                    keys = [_stoi(k) for k in jax_key_str.split(".")]
                    
                    try:
                        _assign_partial_expert(keys, tensor, expert_idx, state_dict, transform, sharding)
                    except Exception as e:
                        conversion_errors.append(f"MoE Expert Assign Error {torch_key}: {e}")
                    continue

                # Case 3: Standard
                jax_key_str = match.expand(repl_pattern)
                keys = [_stoi(k) for k in jax_key_str.split(".")]
                try:
                    _assign_weights(keys, tensor, state_dict, torch_key, transform, sharding)
                except Exception as e:
                    conversion_errors.append(f"Standard Assign Error '{torch_key}' -> '{jax_key_str}': {e}")
        
        gc.collect()

    if conversion_errors:
        full_error_log = "\n".join(conversion_errors[:30])
        raise RuntimeError(f"Encountered {len(conversion_errors)} errors. First 30:\n{full_error_log}")

    if cfg.tie_word_embeddings:
        # 修正: lm_head.w 和 embedder.embedding
        state_dict["lm_head"]["w"] = state_dict["embedder"]["embedding"].T
    
    gc.collect()
    print("Merging state dict into model graph...")
    return nnx.merge(graph_def, state_dict)

if __name__ == "__main__":
    import os
    # 模拟 Config，实际请使用你的 config 加载逻辑
    try:
        cfg = model_lib.ModelConfig.ling_minimal()
    except AttributeError:
        # 如果 model_lib 没有 ling_minimal，创建一个 dummy config 用于测试语法
        print("Warning: Using dummy config for syntax check.")
        class DummyConfig:
            num_heads = 16
            num_kv_heads = 4
            head_dim = 64
            emb_dim = 1024
            moe_intermediate_dim = 14336 // 2
            tie_word_embeddings = False
        cfg = DummyConfig()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 假设路径结构
    project_root = os.path.dirname(os.path.dirname(script_dir))
    model_path = os.path.join(project_root, "inclusionAI/Ling-mini-2.0")
    
    print(f"Loading model from: {model_path}")
    if os.path.exists(model_path):
        try:
            model = create_model_from_safe_tensors(model_path, cfg)
            print("Model created successfully.")
        except Exception as e:
            print(f"Loading failed: {e}")
    else:
        print(f"Path not found: {model_path}")
# import gc
# import re
# import os
# from enum import Enum
# from typing import Any, Tuple, List, Optional

# import jax
# import jax.numpy as jnp
# import numpy as np
# import safetensors.numpy as safetensors  # 使用 numpy 后端读取
# from etils import epath
# from flax import nnx

# # 假设 modeling 就在当前目录下，或者在 python path 中
# import modeling as model_lib

# def dump_structure_to_file(state_dict, filename="model_structure_debug.txt"):
#     """将 JAX 模型的实际结构树打印到文件，方便核对 mapping"""
#     with open(filename, 'w', encoding='utf-8') as f:
#         def _print(d, indent=0):
#             prefix = "  " * indent
#             if isinstance(d, dict):
#                 for k in sorted(d.keys()):
#                     val = d[k]
#                     if hasattr(val, 'shape'): # 是叶子节点 (ShapeDtypeStruct)
#                         f.write(f"{prefix}{k}: {val.shape}\n")
#                     else:
#                         f.write(f"{prefix}{k}/\n")
#                         _print(val, indent + 1)
#             else:
#                 f.write(f"{prefix}Leaf: {type(d)}\n")
        
#         f.write("=== JAX Model State Dict Structure ===\n")
#         _print(state_dict)
#     print(f"Model structure dumped to {filename}. Please check paths if errors persist.")
# def _get_key_and_transform_mapping(cfg: model_lib.ModelConfig):
#     class Transform(Enum):
#         BIAS = None
#         LINEAR = ((1, 0), None, False)
#         EMBED = None
#         ATTN_QKV = ((2, 0, 1), (cfg.num_heads, cfg.head_dim, cfg.emb_dim), True)
#         ATTN_OUT = ((1, 0), (cfg.num_heads, cfg.head_dim, cfg.emb_dim), False)
#         SCALE = None
        
#         # MoE transforms
#         MOE_ROUTER = ((1, 0), None, False)
#         MOE_ROUTER_BIAS = None
#         # MoE Experts
#         MOE_EXPERT_UP = ((1, 0), (cfg.moe_intermediate_dim, cfg.emb_dim), False)
#         MOE_EXPERT_DOWN = ((1, 0), (cfg.emb_dim, cfg.moe_intermediate_dim), False)

#     mapping = {
#         # === 公共部分 ===
#         r"model\.word_embeddings\.weight": ("embedder.embedding", Transform.EMBED),
#         r"lm_head\.weight": ("lm_head.w", Transform.LINEAR),
#         r"model\.norm\.weight": ("final_norm.scale", Transform.SCALE),
        
#         # Layer Norms & Attention
#         r"model\.layers\.([0-9]+)\.attention\.dense\.weight": (r"layers.\1.attn.o_proj.w", Transform.ATTN_OUT),
#         r"model\.layers\.([0-9]+)\.attention\.query_key_value\.weight": (r"layers.\1.attn.qkv_proj", Transform.ATTN_QKV),
#         r"model\.layers\.([0-9]+)\.attention\.query_layernorm\.weight": (r"layers.\1.attn.q_norm.scale", Transform.SCALE),
#         r"model\.layers\.([0-9]+)\.attention\.key_layernorm\.weight": (r"layers.\1.attn.k_norm.scale", Transform.SCALE),
#         r"model\.layers\.([0-9]+)\.input_layernorm\.weight": (r"layers.\1.input_layernorm.scale", Transform.SCALE),
#         r"model\.layers\.([0-9]+)\.post_attention_layernorm\.weight": (r"layers.\1.post_attention_layernorm.scale", Transform.SCALE),
#     }

#     # === Layer 0: Dense MLP ===
#     mapping.update({
#         r"model\.layers\.0\.mlp\.gate_proj\.weight": (r"layers.0.mlp.gate_proj.kernel", Transform.LINEAR),
#         r"model\.layers\.0\.mlp\.up_proj\.weight": (r"layers.0.mlp.up_proj.kernel", Transform.LINEAR),
#         r"model\.layers\.0\.mlp\.down_proj\.weight": (r"layers.0.mlp.down_proj.kernel", Transform.LINEAR),
#     })

#     # === Layer 1-19: MoE layers ===
#     # [关键修改]：将原来的 'moe' 全部改为了 'mlp'，因为 JAX 模型里通常 MoE Block 也是放在 mlp 属性下的
#     # [关键修改]：统一了 shared_expert 的路径，去掉了不一致的 shared_experts (复数) 写法
    
#     # 1. Shared Expert
#     mapping.update({
#         r"model\.layers\.([1-9]|1[0-9])\.mlp\.shared_experts\.gate_proj\.weight": (r"layers.\1.mlp.shared_expert.gate_proj.kernel", Transform.LINEAR),
#         r"model\.layers\.([1-9]|1[0-9])\.mlp\.shared_experts\.up_proj\.weight": (r"layers.\1.mlp.shared_expert.up_proj.kernel", Transform.LINEAR),
#         r"model\.layers\.([1-9]|1[0-9])\.mlp\.shared_experts\.down_proj\.weight": (r"layers.\1.mlp.shared_expert.down_proj.kernel", Transform.LINEAR),
#     })

#     # 2. Router / Gate
#     mapping.update({
#         r"model\.layers\.([1-9]|1[0-9])\.mlp\.gate\.weight": (r"layers.\1.mlp.router.w", Transform.MOE_ROUTER),
#         r"model\.layers\.([1-9]|1[0-9])\.mlp\.gate\.expert_bias": (r"layers.\1.mlp.router.expert_bias", Transform.MOE_ROUTER_BIAS),
#     })

#     # 3. Routed Experts
#     mapping.update({
#         r"model\.layers\.([1-9]|1[0-9])\.mlp\.experts\.([0-9]+)\.gate_proj\.weight": (r"layers.\1.mlp.experts.gate_proj.kernel", Transform.MOE_EXPERT_UP),
#         r"model\.layers\.([1-9]|1[0-9])\.mlp\.experts\.([0-9]+)\.up_proj\.weight": (r"layers.\1.mlp.experts.up_proj.kernel", Transform.MOE_EXPERT_UP),
#         r"model\.layers\.([1-9]|1[0-9])\.mlp\.experts\.([0-9]+)\.down_proj\.weight": (r"layers.\1.mlp.experts.down_proj.kernel", Transform.MOE_EXPERT_DOWN),
#     })

#     return mapping, Transform

# def _find_matching_rule(mapping, source_key):
#     """
#     Finds the regex rule that matches the source_key.
#     Returns: (target_pattern, transform_enum, match_object)
#     """
#     for pat, (repl, transform) in mapping.items():
#         match = re.fullmatch(pat, source_key) # 使用 fullmatch 确保完全匹配
#         if match:
#             return repl, transform, match
#     return None, None, None


# def _assign_weights(keys, tensor, state_dict, st_key, transform, sharding_dict):
#     key, *rest = keys
    
#     # [Check] 如果 key 不在 state_dict 中，这通常是 mapping 路径写错了
#     if key not in state_dict:
#         raise KeyError(f"Key '{key}' not found in state_dict node. Available keys: {list(state_dict.keys())}")

#     if not rest:
#         # [关键修复]：检查 transform.value 是否为 None，防止 Scale/Embed/Bias 报错
#         if transform is not None and transform.value is not None:
#             permute, reshape, reshape_first = transform.value
#             if reshape_first and reshape is not None:
#                 tensor = tensor.reshape(reshape)
#             if permute:
#                 tensor = tensor.transpose(permute)
#             if not reshape_first and reshape is not None:
#                 tensor = tensor.reshape(reshape)
        
#         target_shape = state_dict[key].shape
#         if tensor.shape != target_shape:
#             raise ValueError(f"Shape mismatch for {st_key} -> {'.'.join(str(k) for k in keys)}: "
#                              f"Loaded {tensor.shape} vs Expected {target_shape}")
        
#         if sharding_dict is not None:
#             state_dict[key] = jax.device_put(tensor, sharding_dict[key])
#         else:
#             state_dict[key] = jax.device_put(tensor)
#     else:
#         # [关键修复]：如果在递归时发现当前节点已经是叶子节点 (Struct)，说明路径过深
#         current_node = state_dict[key]
#         if hasattr(current_node, 'shape') and not isinstance(current_node, dict):
#             raise TypeError(f"Path too deep! Attempted to access '{rest[0]}' inside '{key}', "
#                             f"but '{key}' is already a leaf node (ShapeDtypeStruct). "
#                             f"Check if you added unnecessary '.kernel' suffix in mapping.")
            
#         next_sharding = sharding_dict[key] if sharding_dict is not None else None
#         _assign_weights(rest, tensor, current_node, st_key, transform, next_sharding)


# def _assign_partial_expert(keys, sub_tensor, expert_idx, state_dict, transform, sharding_dict=None):
#     node = state_dict
#     sharding_node = sharding_dict
    
#     # 遍历路径
#     path_traversed = []
#     for k in keys[:-1]:
#         if k not in node:
#             raise KeyError(f"MoE Path Error: Key '{k}' not found at path {'.'.join(str(p) for p in path_traversed)}. "
#                            f"Available: {list(node.keys())}")
#         node = node[k]
#         path_traversed.append(k)
#         if sharding_node: sharding_node = sharding_node[k]
    
#     last_key = keys[-1]
#     target_tensor = node[last_key]

#     # [关键修复]：同样检查 transform.value
#     if transform is not None and transform.value is not None:
#         permute, reshape, reshape_first = transform.value
#         if permute:
#             sub_tensor = sub_tensor.transpose(permute)

#     if hasattr(target_tensor, 'shape') and not isinstance(target_tensor, (np.ndarray, jax.Array)):
#         full_shape = target_tensor.shape
#         dtype = target_tensor.dtype
#         node[last_key] = np.zeros(full_shape, dtype=dtype)
#         target_tensor = node[last_key]

#     target_tensor[expert_idx] = sub_tensor
#     # """
#     # MoE 专用赋值函数：用于将单个专家的权重填入大 Tensor 的特定切片中。
#     # keys: 指向 stacked tensor 的完整路径 (e.g. layers, 1, moe, experts, gate_proj, kernel)
#     # expert_idx: 当前是第几个专家
#     # """
#     # # 1. 递归找到目标叶子节点
#     # node = state_dict
#     # sharding_node = sharding_dict
    
#     # # 遍历路径直到倒数第二个 (parent)
#     # parent = None
#     # last_key = keys[-1]
    
#     # for k in keys[:-1]:
#     #     parent = node
#     #     node = node[k]
#     #     if sharding_node: sharding_node = sharding_node[k]
    
#     # target_tensor = node[last_key] # 这里可能是 ShapeDtypeStruct，也可能是 Array

#     # # 2. 对子权重进行变换 (Linear Transpose)
#     # # 注意：这里的 transform 是针对"单专家"维度的，比如 (1,0) 转置
#     # if transform is not None:
#     #     permute, reshape, reshape_first = transform.value
#     #     # 忽略 reshape_first 对专家层级的复杂影响，通常专家内部只是简单的 (1,0) 转置
#     #     if permute:
#     #         sub_tensor = sub_tensor.transpose(permute)

#     # # 3. 懒初始化 (Lazy Initialization)
#     # # 如果目标还是 ShapeDtypeStruct (JAX 的占位符)，说明这是第一次写入，需要分配内存
#     # if hasattr(target_tensor, 'shape') and not isinstance(target_tensor, (np.ndarray, jax.Array)):
#     #     full_shape = target_tensor.shape
#     #     dtype = target_tensor.dtype
#     #     # 创建全 0 的 numpy 数组
#     #     # print(f"  [Init MoE Buffer] {'.'.join(str(k) for k in keys)} shape={full_shape}")
#     #     node[last_key] = np.zeros(full_shape, dtype=dtype)
#     #     target_tensor = node[last_key]

#     # # 4. 写入切片
#     # # 假设 JAX 定义的形状是 [num_experts, features_in, features_out]
#     # # 我们把处理好的 sub_tensor 塞进对应的 index
#     # target_tensor[expert_idx] = sub_tensor


# def _stoi(s):
#     try:
#         return int(s)
#     except ValueError:
#         return s


# def create_model_from_safe_tensors(
#     file_dir: str, cfg: model_lib.ModelConfig, mesh: jax.sharding.Mesh | None = None
# ) -> model_lib.Ling2_mini:
    
#     files = list(epath.Path(file_dir).expanduser().glob("*.safetensors"))
#     if not files:
#         raise ValueError(f"No safetensors found in {file_dir}")

#     # 1. 初始化空模型结构
#     print("Initializing model skeleton...")
#     ling2_mini = nnx.eval_shape(lambda: model_lib.Ling2_mini(cfg, rngs=nnx.Rngs(params=0)))
#     graph_def, abs_state = nnx.split(ling2_mini)
#     state_dict = abs_state.to_pure_dict()
#     sharding = nnx.get_named_sharding(abs_state, mesh).to_pure_dict() if mesh is not None else None

#     mapping, TransformEnum = _get_key_and_transform_mapping(cfg)
#     conversion_errors = []

#     print(f"Start loading weights from {len(files)} files...")
    
#     for f in files:
#         with safetensors.safe_open(f, framework="numpy") as sf:
#             for torch_key in sf.keys():
#                 tensor = sf.get_tensor(torch_key)

#                 # 寻找匹配规则
#                 repl_pattern, transform, match = _find_matching_rule(mapping, torch_key)
                
#                 if not match:
#                     # print(f"Skipping unmapped key: {torch_key}")
#                     continue

#                 # === SPECIAL CASE 1: QKV Splitting ===
#                 # 如果是 query_key_value，说明 PyTorch 是合并的，但我们需要拆分给 JAX
#                 if "query_key_value" in torch_key:
#                     # print(f"Splitting QKV: {torch_key}")
#                     layer_idx = int(match.group(1)) # 获取层号
                    
#                     # 计算切分点
#                     head_dim = cfg.head_dim
#                     num_heads = cfg.num_heads
#                     num_kv_heads = cfg.num_kv_heads
                    
#                     q_dim = num_heads * head_dim
#                     kv_dim = num_kv_heads * head_dim
                    
#                     # 切分 PyTorch Tensor (axis=0 通常是 feature 维度)
#                     # Qwen QKV 也就是按 Q, K, V 顺序拼接的
#                     q, k, v = np.split(tensor, [q_dim, q_dim + kv_dim], axis=0)
                    
#                     # 手动构造目标路径，覆盖 mapping 中的默认路径
#                     # 这里假设 JAX 模型里叫 q_proj, k_proj, v_proj
#                     q_path = ["layers", layer_idx, "attn", "q_proj", "w"]
#                     k_path = ["layers", layer_idx, "attn", "k_proj", "w"]
#                     v_path = ["layers", layer_idx, "attn", "v_proj", "w"]
                    
#                     # 我们需要用到 ATTN_QKV 定义的变换规则 (通常包含 reshape)
#                     # 但要注意 Transform.ATTN_QKV 定义的是合并时的规则，这里拆分后可能需要稍微调整
#                     # 通常 Q/K/V 拆开后，只需要 Reshape 成 (heads, head_dim, ...) 并 Permute
#                     # 这里复用 Transform.ATTN_QKV 的逻辑，因为它定义了 ((2,0,1), ...)
                    
#                     # 为简单起见，这里复用 Transform.ATTN_QKV 的参数，但需要注意 num_heads 的不同
#                     # Q 的 Transform
#                     q_trans = (TransformEnum.ATTN_QKV.value[0], (cfg.num_heads, cfg.head_dim, cfg.emb_dim), True)
#                     # K, V 的 Transform (num_kv_heads)
#                     kv_trans = (TransformEnum.ATTN_QKV.value[0], (cfg.num_kv_heads, cfg.head_dim, cfg.emb_dim), True)

#                     # 构造临时的 Enum 值传给 assign
#                     class TempTransform: pass
#                     t_q = TempTransform(); t_q.value = q_trans
#                     t_kv = TempTransform(); t_kv.value = kv_trans

#                     try:
#                         _assign_weights(q_path, q, state_dict, torch_key + "_Q", t_q, sharding)
#                         _assign_weights(k_path, k, state_dict, torch_key + "_K", t_kv, sharding)
#                         _assign_weights(v_path, v, state_dict, torch_key + "_V", t_kv, sharding)
#                     except Exception as e:
#                         conversion_errors.append(f"QKV Split Error {torch_key}: {e}")
                    
#                     continue # 处理完毕，跳过后续通用逻辑

#                 # === SPECIAL CASE 2: MoE Experts (Many-to-One) ===
#                 # 正则中有两个捕获组：(layer_idx, expert_idx)
#                 if len(match.groups()) == 2 and "experts" in torch_key and "shared" not in torch_key:
#                     layer_idx = int(match.group(1))
#                     expert_idx = int(match.group(2))
                    
#                     # 生成 JAX 目标字符串 (使用 repl_pattern，它会自动填入 \1)
#                     # 此时 repl_pattern 类似于 "layers.1.moe.experts.gate_proj.kernel"
#                     # 注意：re.sub 会处理 \1，但我们自己 expand 比较安全
#                     jax_key_str = match.expand(repl_pattern)
                    
#                     keys = [_stoi(k) for k in jax_key_str.split(".")]
                    
#                     try:
#                         _assign_partial_expert(keys, tensor, expert_idx, state_dict, transform, sharding)
#                     except Exception as e:
#                         conversion_errors.append(f"MoE Expert Assign Error {torch_key}: {e}")
                    
#                     continue

#                 # === STANDARD CASE: 1-to-1 Mapping ===
#                 # 生成 JAX Key
#                 jax_key_str = match.expand(repl_pattern)
#                 keys = [_stoi(k) for k in jax_key_str.split(".")]
                
#                 try:
#                     _assign_weights(keys, tensor, state_dict, torch_key, transform, sharding)
#                 except Exception as e:
#                     # 只有当错误不是 "KeyError" (说明模型里没这个层) 时才报错
#                     # 或者根据需要记录
#                     conversion_errors.append(f"Standard Assign Error '{torch_key}' -> '{jax_key_str}': {e}")
        
#         # 释放内存
#         gc.collect()

#     if conversion_errors:
#         full_error_log = "\n".join(conversion_errors[:20]) # 只打印前20个错误防止刷屏
#         raise RuntimeError(f"Encountered {len(conversion_errors)} weight conversion errors. First 20:\n{full_error_log}")

#     # 处理 Word Embeddings 和 LM Head 绑定的情况
#     if cfg.tie_word_embeddings:
#         print("Tying word embeddings to lm_head...")
#         state_dict["lm_head"]["w"] = state_dict["embedder"]["embedding"].T
    
#     gc.collect()
#     print("Merging state dict into model graph...")
#     return nnx.merge(graph_def, state_dict)

# if __name__ == "__main__":
#     import os
#     # 模拟 Config，实际请使用你的 config 加载逻辑
#     try:
#         cfg = model_lib.ModelConfig.ling_minimal()
#     except AttributeError:
#         # 如果 model_lib 没有 ling_minimal，创建一个 dummy config 用于测试语法
#         print("Warning: Using dummy config for syntax check.")
#         class DummyConfig:
#             num_heads = 16
#             num_kv_heads = 4
#             head_dim = 64
#             emb_dim = 1024
#             moe_intermediate_dim = 14336 // 2
#             tie_word_embeddings = False
#         cfg = DummyConfig()

#     script_dir = os.path.dirname(os.path.abspath(__file__))
#     # 假设路径结构
#     project_root = os.path.dirname(os.path.dirname(script_dir))
#     model_path = os.path.join(project_root, "inclusionAI/Ling-mini-2.0")
    
#     print(f"Loading model from: {model_path}")
#     if os.path.exists(model_path):
#         try:
#             model = create_model_from_safe_tensors(model_path, cfg)
#             print("Model created successfully.")
#         except Exception as e:
#             print(f"Loading failed: {e}")
#     else:
#         print(f"Path not found: {model_path}")