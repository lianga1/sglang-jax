import dataclasses
import math
from functools import partial
from typing import Tuple, TypeAlias, Optional

import jax
from flax import nnx
from jax import P
from jax import numpy as jnp
from jax.sharding import PartitionSpec, get_abstract_mesh, reshard
from jaxtyping import Array, ArrayLike

_K_MASK = jnp.finfo(jnp.bfloat16).min
ShardingSpec = PartitionSpec

@dataclasses.dataclass(slots=True, frozen=True)
class ShardingCfg:
    emb_vd: ShardingSpec
    emb_dv: ShardingSpec
    q_weight_ndh: ShardingSpec
    kv_weight_ndh: ShardingSpec
    o_weight_nhd: ShardingSpec
    ffw_weight_df: ShardingSpec
    ffw_weight_fd: ShardingSpec
    rms_norm: ShardingSpec
    act_btd: ShardingSpec
    act_btf: ShardingSpec
    act_btnh: ShardingSpec
    # MoE sharding specs
    gate_weight: ShardingSpec  # [D, E]
    # Expert weights: [Experts, In, Out] or [Experts, Out, In]
    # 这里的维度定义要配合 Einsum 里的 sharding 顺序
    expert_weight_edf: ShardingSpec  # [E, D, F] for Up/Gate
    expert_weight_efd: ShardingSpec  # [E, F, D] for Down

    @staticmethod
    def no_sharding():
        """Configuration with no sharding (all None)."""
        return ShardingCfg(
            emb_vd=P(None, None),
            emb_dv=P(None, None),
            q_weight_ndh=P(None, None, None),
            kv_weight_ndh=P(None, None, None),
            o_weight_nhd=P(None, None, None),
            ffw_weight_df=P(None, None),
            ffw_weight_fd=P(None, None),
            rms_norm=P(None),
            act_btd=P(None, None, None),
            act_btf=P(None, None, None),
            act_btnh=P(None, None, None, None),
            gate_weight=P(None, None),
            expert_weight_edf=P(None, None, None),
            expert_weight_efd=P(None, None, None),
        )

    @staticmethod
    def default():
        return ShardingCfg(
            emb_vd=P("tp", "fsdp"),
            emb_dv=P("fsdp", "tp"),
            q_weight_ndh=P("tp", "fsdp", None),
            kv_weight_ndh=P("tp", "fsdp", None),
            o_weight_nhd=P("tp", None, "fsdp"),
            ffw_weight_df=P("fsdp", "tp"),
            ffw_weight_fd=P("tp", "fsdp"),
            rms_norm=P("tp"),
            act_btd=P("fsdp", None, "tp"),
            act_btf=P("fsdp", None, "tp"),
            act_btnh=P("fsdp", None, "tp", None),
            # MoE Sharding
            gate_weight=P(None, None),  # Router通常较小，复制到所有设备
            # Expert Parallel (EP): 专家维度切分
            # expert_weight_edf=P("ep", "fsdp", "tp"), 
            # expert_weight_efd=P("ep", "tp", "fsdp"),
            expert_weight_edf=P(None,"tp", None),
            expert_weight_efd=P(None, None, "tp"),
        )


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    num_layers: int
    vocab_size: int
    emb_dim: int
    num_heads: int
    head_dim: int
    num_kv_heads: int
    rope_theta: int
    norm_eps: float
    tie_word_embeddings: bool
    
    # === 关键修正 ===
    intermediate_size: int      # Layer 0 (Dense) 的中间维度 (5120)
    moe_intermediate_dim: int   # Layer 1+ (MoE Experts) 的中间维度 (512)
    shared_expert_dim: int      # Layer 1+ (Shared Expert) 的中间维度 (512)
    num_experts: int            # 专家数量 (256)
    num_experts_per_tok: int    # (8)
    
    shd_cfg: ShardingCfg = ShardingCfg.no_sharding()

    @classmethod
    def _from_param(cls, use_sharding: bool, **kwargs):
        if use_sharding:
            kwargs["shd_cfg"] = ShardingCfg.default()
        return cls(**kwargs)

    @classmethod
    def ling_minimal(cls, use_sharding: bool = False):
        # 根据你提供的 config.json 填入数值
        return cls._from_param(
            use_sharding,
            num_layers=20,
            vocab_size=157184,
            emb_dim=2048,            # hidden_size
            num_heads=16,            # num_attention_heads
            head_dim=128,            # head_dim
            num_kv_heads=4,          # num_key_value_heads
            rope_theta=600000,
            norm_eps=1e-06,
            tie_word_embeddings=False,
            
            # === 维度修正 ===
            intermediate_size=5120,               # 对应 config.intermediate_size (用于 Layer 0)
            moe_intermediate_dim=512,             # 对应 config.moe_intermediate_size (用于 Experts)
            shared_expert_dim=512,                # 对应 config.moe_shared_expert_intermediate_size (用于 Shared)
            num_experts=256,                      # 对应 config.num_experts (你之前写了 64)
            num_experts_per_tok=8,
        )

def shard(x: jnp.ndarray, s: ShardingSpec):
    mesh = get_abstract_mesh()
    if not mesh.empty and len(mesh.axis_names) > 0:
        return reshard(x, s)
    return x


class LayerCache(nnx.Module):
    def __init__(self, cfg: ModelConfig, batch_size: int, cache_size: int, dtype: jnp.dtype):
        cache_shape = (batch_size, cache_size, cfg.num_kv_heads, cfg.head_dim)
        self.k_cache = shard(nnx.Cache(jnp.zeros(cache_shape, dtype=dtype)), cfg.shd_cfg.act_btnh)
        self.v_cache = shard(nnx.Cache(jnp.zeros(cache_shape, dtype=dtype)), cfg.shd_cfg.act_btnh)
        self.size = self.k_cache.shape[1]
        # Batch维度sharding
        batch_sharding = P(cfg.shd_cfg.act_btnh[0]) if cfg.shd_cfg.act_btnh else P(None)
        self.start_ind = shard(nnx.Variable(-1 * jnp.ones((batch_size,), dtype=jnp.int32)), batch_sharding)
        self.cur_ind = nnx.Variable(jnp.zeros((), dtype=jnp.int32)) 

Cache: TypeAlias = list[LayerCache]


class Einsum(nnx.Module):
    def __init__(self, einsum_str: str, shape: tuple[int, ...], *, shd: ShardingSpec, rngs: nnx.Rngs):
        self.einsum_str = einsum_str
        self.shape = shape
        self.w = shard(nnx.Param(nnx.initializers.normal()(rngs.params(), shape)), shd)

    @jax.named_scope("einsum")
    def __call__(self, x: ArrayLike) -> Array:
        return jnp.einsum(self.einsum_str, x, self.w.value)


# --- RoPE Utilities ---
def _generate_pos_embeddings(positions: jax.Array, head_dim: int, rope_theta: int) -> tuple[jax.Array, jax.Array]:
    fraction = jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim
    timescale = rope_theta**fraction
    rotational_frequency = 1.0 / timescale
    sinusoid_inp = jnp.einsum("BT,k->BTk", positions, rotational_frequency, precision=jax.lax.Precision.HIGHEST)
    return jnp.sin(sinusoid_inp), jnp.cos(sinusoid_inp)

def apply_rope(x: jax.Array, sin: jax.Array, cos: jax.Array) -> jax.Array:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    sin, cos = sin[:, :, None, :], cos[:, :, None, :]
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1).astype(x.dtype)

# --- Layers ---

class RMSNorm(nnx.Module):
    def __init__(self, dim: int, cfg: ModelConfig, *, rngs: nnx.Rngs):
        self.scale = shard(nnx.Param(nnx.initializers.ones_init()(rngs.params(), dim)), cfg.shd_cfg.rms_norm)
        self.norm_eps = cfg.norm_eps

    @jax.named_scope("rms_norm")
    def __call__(self, x: Array) -> Array:
        dtype = x.dtype
        # High precision for norm calculation
        x_float = jnp.astype(x, jnp.float32)
        rms = jnp.sqrt(jnp.mean(x_float ** 2, axis=-1, keepdims=True) + self.norm_eps)
        return jnp.astype(self.scale.value * x_float / rms, dtype)

class Attention(nnx.Module):
    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs):
        self.shd_cfg = cfg.shd_cfg
        einsum_fn = partial(Einsum, rngs=rngs)
        self.q_proj = einsum_fn(
            "BTD,DNH->BTNH", (cfg.emb_dim, cfg.num_heads, cfg.head_dim), shd=self.shd_cfg.q_weight_ndh
        )
        self.k_proj = einsum_fn(
            "BSD,DKH->BSKH", (cfg.emb_dim, cfg.num_kv_heads, cfg.head_dim), shd=self.shd_cfg.kv_weight_ndh
        )
        self.v_proj = einsum_fn(
            "BSD,DKH->BSKH", (cfg.emb_dim, cfg.num_kv_heads, cfg.head_dim), shd=self.shd_cfg.kv_weight_ndh
        )
        self.o_proj = einsum_fn(
            "BTNH,NHD->BTD", (cfg.num_heads, cfg.head_dim, cfg.emb_dim), shd=self.shd_cfg.o_weight_nhd
        )
        # Norms
        self.q_norm = RMSNorm(cfg.head_dim, cfg, rngs=rngs)
        self.k_norm = RMSNorm(cfg.head_dim, cfg, rngs=rngs)
        
        self.n_rep = cfg.num_heads // cfg.num_kv_heads
        self.scale = cfg.head_dim**-0.5
        self.head_dim = cfg.head_dim
        self.num_kv_heads = cfg.num_kv_heads

    @jax.named_scope("attention")
    def __call__(self, x: Array, cache: LayerCache | None, segment_ids: Array) -> Array:
        # Projection & QK Norm
        q = shard(self.q_norm(self.q_proj(x)), self.shd_cfg.act_btnh)
        k = shard(self.k_norm(self.k_proj(x)), self.shd_cfg.act_btnh)
        v = shard(self.v_proj(x), self.shd_cfg.act_btnh)

        # RoPE
        # 计算位置编码需要 segment_ids
        left_pads = count_left_pads(segment_ids)
        if cache.start_ind.value is not None:
             cache.start_ind.value = jnp.where(cache.start_ind.value < 0, left_pads, cache.start_ind.value)
        
        position_ids = compute_positions_from_segment_ids(segment_ids) + cache.cur_ind.value
        sin, cos = _generate_pos_embeddings(position_ids, self.head_dim, 600000) # 这里硬编码了theta，实际应从cfg传
        q = apply_rope(q, sin, cos)
        k = apply_rope(k, sin, cos)

        # Cache Update
        # slice_indices = jnp.array([0, cache.cur_ind.value, 0, 0], dtype=jnp.bfloat16)
        slice_indices = (0, cache.cur_ind.value, 0, 0)
        cache.v_cache.value = jax.lax.dynamic_update_slice(cache.v_cache.value, v, slice_indices)
        cache.k_cache.value = jax.lax.dynamic_update_slice(cache.k_cache.value, k, slice_indices)
        
        # GQA / Attention
        b, t, n, h = q.shape
        q_gqa = q.reshape((b, t, self.num_kv_heads, self.n_rep, h))
        
        # [B, T, K, G, H] * [B, S, K, H] -> [B, T, S, K, G]
        attn_logits = jnp.einsum("BTKGH,BSKH->BTSKG", q_gqa, cache.k_cache.value) * self.scale
        
        # Masking
        q_pos = cache.cur_ind.value + jnp.arange(t, dtype=jnp.int32)[None, :] - cache.start_ind.value[:, None]
        ts = jnp.arange(cache.size, dtype=jnp.int32)
        kv_segment_ids = (ts[None, :] >= cache.start_ind.value[:, None]) & (ts[None, :] < cache.cur_ind.value + t)
        k_pos = ts[None, :] - cache.start_ind.value[:, None]
        
        causal_mask = k_pos[:, None, :] <= q_pos[:, :, None]
        segment_mask = kv_segment_ids[:, None, :] == segment_ids[:, :, None]
        final_mask = causal_mask & segment_mask
        
        attn_logits = jnp.where(final_mask[:, :, :, None, None], attn_logits, _K_MASK)
        
        attn_weights = jax.nn.softmax(attn_logits.astype(jnp.float32), axis=2).astype(attn_logits.dtype)
        
        # [B, T, S, K, G] * [B, S, K, H] -> [B, T, K, G, H]
        out = jnp.einsum("BTSKG,BSKH->BTKGH", attn_weights, cache.v_cache.value)
        out = out.reshape((b, t, n, h))
        
        cache.cur_ind.value = cache.cur_ind.value + t
        return shard(self.o_proj(out), self.shd_cfg.act_btd)


class MLP(nnx.Module):
    """Standard Dense MLP (SwiGLU)"""
    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs, intermediate_dim: Optional[int] = None):
        self.shd_cfg = cfg.shd_cfg
        dim = intermediate_dim if intermediate_dim is not None else cfg.moe_intermediate_dim # Default fall back
        
        einsum_fn = partial(Einsum, rngs=rngs)
        self.gate_proj = einsum_fn(
            "BTD,DF->BTF", (cfg.emb_dim, dim), shd=self.shd_cfg.ffw_weight_df
        )
        self.up_proj = einsum_fn(
            "BTD,DF->BTF", (cfg.emb_dim, dim), shd=self.shd_cfg.ffw_weight_df
        )
        self.down_proj = einsum_fn(
            "BTF,FD->BTD", (dim, cfg.emb_dim), shd=self.shd_cfg.ffw_weight_fd
        )
        
    @jax.named_scope("mlp")
    def __call__(self, x: ArrayLike) -> Array:
        activations = nnx.silu(self.gate_proj(x)) * self.up_proj(x)
        activations = shard(activations, self.shd_cfg.act_btf)
        return self.down_proj(activations)


class MoEMLP(nnx.Module):
    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs):
        self.shd_cfg = cfg.shd_cfg
        self.num_experts = cfg.num_experts
        self.k = cfg.num_experts_per_tok
        self.dim = cfg.emb_dim
        self.hidden = cfg.moe_intermediate_dim # 使用 512
        
        # 1. Router (Gate)
        # self.router = shard(
        #     nnx.Param(nnx.initializers.normal()(rngs.params(), (self.dim, self.num_experts))),
        #     cfg.shd_cfg.gate_weight
        # )
        init_fn = nnx.initializers.normal(stddev=0.02, dtype=jnp.bfloat16) 

        # 1. Router
        self.router = shard(
            nnx.Param(init_fn(rngs.params(), (self.dim, self.num_experts))),
            cfg.shd_cfg.gate_weight
        )
        self.router_bias = None # 如果模型有bias需添加

        # 2. Shared Expert (Always active)
        # 假设共享专家使用 shared_expert_intermediate_dim，如果cfg没定义则用默认
        shared_dim = getattr(cfg, "shared_expert_intermediate_dim", cfg.moe_intermediate_dim)
        # self.shared_expert = MLP(cfg, rngs=rngs, intermediate_dim=shared_dim)
        self.shared_expert = MLP(cfg, rngs=rngs, intermediate_dim=cfg.shared_expert_dim)
        # 3. Routed Experts (Stacked)
        # weights: [Experts, In, Out] based on loading script logic
        self.experts_gate_proj = shard(
            nnx.Param(init_fn(rngs.params(), (self.num_experts, self.dim, self.hidden))),
            cfg.shd_cfg.expert_weight_edf
        )
        self.experts_up_proj = shard(
            nnx.Param(init_fn(rngs.params(), (self.num_experts, self.dim, self.hidden))),
            cfg.shd_cfg.expert_weight_edf
        )
        self.experts_down_proj = shard(
            nnx.Param(init_fn(rngs.params(), (self.num_experts, self.hidden, self.dim))),
            cfg.shd_cfg.expert_weight_efd
        )

    @jax.named_scope("moe")
    def __call__(self, x: Array) -> Array:
        # x: [B, T, D]
        
        # --- 1. Shared Expert Path ---
        shared_out = self.shared_expert(x)
        
        # --- 2. Router ---
        router_logits = x @ self.router.value # [B, T, E]
        scores = jax.nn.softmax(router_logits.astype(jnp.float32), axis=-1)
        
        # Top-K
        # weights: [B, T, K], indices: [B, T, K]
        topk_weights, topk_ids = jax.lax.top_k(scores, self.k)
        
        # Normalize weights
        topk_weights = topk_weights / jnp.sum(topk_weights, axis=-1, keepdims=True)
        topk_weights = topk_weights.astype(x.dtype)

        # --- 3. Vectorized Routed Computation ---
        routed_out = self._compute_routed_experts(x, topk_ids, topk_weights)
        
        return shared_out + routed_out

    def _compute_routed_experts(self, x: Array, topk_ids: Array, topk_weights: Array) -> Array:
        """
        Executes expert computation by gathering weights.
        Avoids creating [B, T, E] masks which consume O(E) memory.
        """
        # x: [B, T, D]
        # topk_ids: [B, T, K]
        
        # Gather weights: [B, T, K, D, F]
        # 注意：这在 num_experts 很大时比 mask 更省显存，但在此步会产生较大临时张量
        # 如果爆显存，可以使用 scan over K loop
        
        def compute_per_k(k_idx):
            # 获取第 k 个选择的专家 ID: [B, T]
            expert_ids = topk_ids[..., k_idx] 
            # 获取第 k 个选择的权重: [B, T]
            weights = topk_weights[..., k_idx]
            
            # 使用 take 获取对应专家的参数
            # shape: [B, T, D, F]
            # mode='fill' handles OOB if any, typically safe here
            # cur_gate_w = jnp.take(self.experts_gate_proj.value, expert_ids, axis=0)
            # cur_up_w = jnp.take(self.experts_up_proj.value, expert_ids, axis=0)
            # cur_down_w = jnp.take(self.experts_down_proj.value, expert_ids, axis=0)
            axis_b = self.shd_cfg.act_btd[0] 
            axis_t = self.shd_cfg.act_btd[1]
            
            # Expert Weights sharding: [Experts, In, Out] -> 取后两个轴
            # Gate/Up proj: [E, D, F]
            axis_d = self.shd_cfg.expert_weight_edf[1] 

            
            # Down proj: [E, F, D]
            axis_f_down = self.shd_cfg.expert_weight_efd[1]


            # 构造 Target Sharding Specs
            # Gate/Up 结果: [B, T, D, F]
            spec_up = P(axis_b, axis_t, axis_d)
            # Down 结果: [B, T, F, D]
            spec_down = P(axis_b, axis_t, axis_f_down)

            # 使用 .at[...].get(out_sharding=...) 替代 jnp.take
            # 相当于: cur_gate_w = self.experts_gate_proj.value[expert_ids]
            cur_gate_w = self.experts_gate_proj.value.at[expert_ids].get(out_sharding=spec_up)
            cur_up_w = self.experts_up_proj.value.at[expert_ids].get(out_sharding=spec_up)
            cur_down_w = self.experts_down_proj.value.at[expert_ids].get(out_sharding=spec_down)
            # === FIX END ===
            # Forward Computation
            # x expanded: [B, T, 1, D] for broadcasting against [B, T, D, F]
            # but simpler: einsum
            # Gate & Up: [B, T, D] * [B, T, D, F] -> [B, T, F]
            # We use einsum to do the batched dot product
            
            gate_out = jnp.einsum("btd,btdf->btf", x, cur_gate_w,out_sharding=spec_up)
            up_out = jnp.einsum("btd,btdf->btf", x, cur_up_w,out_sharding=spec_up)
            
            hidden = nnx.silu(gate_out) * up_out
            
            # Down: [B, T, F] * [B, T, F, D] -> [B, T, D]
            expert_out = jnp.einsum("btf,btfd->btd", hidden, cur_down_w,out_sharding=spec_down)
            
            return expert_out * weights[..., None]

        # 循环 K 次并累加 (Scan is efficient here)
        # result: [K, B, T, D]
        _, results = jax.lax.scan(
            lambda carry, i: (carry, compute_per_k(i)), # 函数体
            None,                                       # init carry
            jnp.arange(self.k)                          # xs (要循环的序列)
        )
        
        # Sum over K
        return jnp.sum(results, axis=0)


class DecoderLayer(nnx.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int, *, rngs: nnx.Rngs):
        self.input_layernorm = RMSNorm(cfg.emb_dim, cfg, rngs=rngs)
        self.attn = Attention(cfg=cfg, rngs=rngs)
        self.post_attention_layernorm = RMSNorm(cfg.emb_dim, cfg, rngs=rngs)
        
        # Hybrid Architecture Logic:
        # Layer 0 -> Dense MLP
        # Layer 1+ -> MoE MLP

        if layer_idx == 0:
            # Layer 0: Dense MLP
            # [关键] 显式传入 intermediate_dim = 5120
            self.mlp = MLP(cfg=cfg, rngs=rngs, intermediate_dim=cfg.intermediate_size)
        else:
            # Layer 1+: MoE MLP
            self.mlp = MoEMLP(cfg=cfg, rngs=rngs)
        # if layer_idx == 0:
        #     # 使用 moe_intermediate_dim 或者一个特定的 dense dim，这里假设复用
        #     self.mlp = MLP(cfg=cfg, rngs=rngs)
        # else:
        #     self.mlp = MoEMLP(cfg=cfg, rngs=rngs)

    def __call__(self, x: Array, cache: LayerCache | None, segment_ids: Array) -> Array:
        # Pre-Norm Architecture
        norm_x = self.input_layernorm(x)
        attn_out = self.attn(norm_x, cache, segment_ids)
        x = x + attn_out
        
        norm_x2 = self.post_attention_layernorm(x)
        mlp_out = self.mlp(norm_x2)
        x = x + mlp_out
        return x


class Ling2_mini(nnx.Module):
    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs):
        self.embedder = shard(
            nnx.Embed(num_embeddings=cfg.vocab_size, features=cfg.emb_dim, dtype=jnp.bfloat16, rngs=rngs),
            cfg.shd_cfg.emb_vd,
        )
        self.out_emb_shd = None if get_abstract_mesh().empty else cfg.shd_cfg.act_btd
        
        # 传递 layer_idx 以支持混合架构初始化
        self.layers = nnx.List([
            DecoderLayer(cfg=cfg, layer_idx=i, rngs=rngs) 
            for i in range(cfg.num_layers)
        ])
        
        self.final_norm = RMSNorm(cfg.emb_dim, cfg, rngs=rngs)
        self.lm_head = Einsum(
            einsum_str="BTD,DV->BTV", shape=(cfg.emb_dim, cfg.vocab_size), shd=cfg.shd_cfg.emb_dv, rngs=rngs
        )

    def init_cache(
        self, cfg: ModelConfig, batch_size: int, token_len: int, generate_steps: int, dtype: jnp.dtype = jnp.bfloat16
    ) -> Cache:
        # Ensure minimal cache size
        size = max(token_len + generate_steps, 1)
        cache_size = 2 ** math.ceil(math.log2(size))
        return [LayerCache(cfg, batch_size, cache_size, dtype) for _ in range(cfg.num_layers)]

    def __call__(self, tokens, segment_ids, cache, num_right_pads):
        # Embedding
        x = self.embedder.embedding.value.at[(tokens,)].get(out_sharding=self.out_emb_shd)
        
        # Layers
        for i, layer in enumerate(self.layers):
            x = layer(x, cache[i], segment_ids)
            
        # Final
        logits = self.lm_head(self.final_norm(x))
        return logits


# --- Utils for execution ---

def count_left_pads(x: jax.Array) -> int:
    return jnp.sum(jnp.cumsum(x != 0, axis=-1) == 0, -1)

def count_right_pads(x: jax.Array, pad_id) -> int:
    # Avoid zero-size argmin issues
    mask = x == pad_id
    if mask.ndim > 1:
        # Check if entire row is pad
        all_pad = jnp.all(mask, axis=1)
        # Find first non-pad from right
        # flip mask: [True, True, False, False] -> [False, False, True, True]
        # argmin gives index of first False (0) -> 0 if all True? No argmin on bool gives False=0, True=1.
        # So we want first 1 (True) in the flipped mask if we look for pads?
        # Actually logic: argmin(flip(x==pad)) finds first Non-Pad from right.
        flipped = jnp.flip(mask, axis=1)
        return jnp.where(all_pad, x.shape[1], jnp.argmin(flipped.astype(jnp.int32), axis=1))
    return 0

def compute_positions_from_segment_ids(seg_ids):
    # Vectorized position calculation
    return jax.vmap(lambda row: jnp.where(row != 0, jnp.arange(seg_ids.shape[1]) - jnp.argmax(row), 2**30))(seg_ids)

@jax.jit
def forward(model: nnx.Module, cache: Cache, tokens: Array, pad_id: int) -> tuple[Array, nnx.Cache]:
    segment_ids = 1 * (tokens != pad_id)
    # This might need to be passed in from host if dynamic shapes are an issue, 
    # but here we compute it inside jit
    num_right_pads = count_right_pads(tokens, pad_id)
    
    logits = model(tokens, segment_ids, cache, num_right_pads)
    
    # Get logits of the last valid token
    # shape: [B]
    target_ind = tokens.shape[-1] - num_right_pads - 1
    # Gather: logits[batch, target_ind]
    # We need advanced indexing
    batch_inds = jnp.arange(tokens.shape[0],out_sharding=P("fsdp",None))
    last_logits = logits.at[batch_inds, target_ind, :].get(out_sharding=P("fsdp",None))
    
    return last_logits, cache
# import dataclasses
# import math
# from functools import partial
# from typing import Tuple, TypeAlias

# import jax
# from flax import nnx
# from jax import P
# from jax import numpy as jnp
# from jax.sharding import PartitionSpec, get_abstract_mesh, reshard
# from jaxtyping import Array, ArrayLike

# _K_MASK = jnp.finfo(jnp.bfloat16).min
# ShardingSpec = PartitionSpec

# @dataclasses.dataclass(slots=True, frozen=True)
# class ShardingCfg:
#     emb_vd: ShardingSpec
#     emb_dv: ShardingSpec
#     q_weight_ndh: ShardingSpec
#     kv_weight_ndh: ShardingSpec
#     o_weight_nhd: ShardingSpec
#     ffw_weight_df: ShardingSpec
#     ffw_weight_fd: ShardingSpec
#     rms_norm: ShardingSpec
#     act_btd: ShardingSpec
#     act_btf: ShardingSpec
#     act_btnh: ShardingSpec
#     # MoE sharding specs
#     gate_weight: ShardingSpec  # [D, E]
#     expert_weight_edf: ShardingSpec  # [E, D, F] - gate_proj, up_proj
#     expert_weight_efd: ShardingSpec  # [E, F, D] - down_proj

#     @staticmethod
#     def no_sharding():
#         """Configuration with no sharding (all None)."""
#         return ShardingCfg(
#             emb_vd=P(None, None),
#             emb_dv=P(None, None),
#             q_weight_ndh=P(None, None, None),
#             kv_weight_ndh=P(None, None, None),
#             o_weight_nhd=P(None, None, None),
#             ffw_weight_df=P(None, None),
#             ffw_weight_fd=P(None, None),
#             rms_norm=P(None),
#             act_btd=P(None, None, None),
#             act_btf=P(None, None, None),
#             act_btnh=P(None, None, None, None),
#             gate_weight=P(None, None),
#             expert_weight_edf=P(None, None, None),
#             expert_weight_efd=P(None, None, None),
#         )

#     @staticmethod
#     def default():
#         return ShardingCfg(
#             emb_vd=P("tp", "fsdp"),
#             emb_dv=P("fsdp", "tp"),
#             q_weight_ndh=P("tp", "fsdp", None),
#             kv_weight_ndh=P("tp", "fsdp", None),
#             o_weight_nhd=P("tp", None, "fsdp"),
#             ffw_weight_df=P("fsdp", "tp"),
#             ffw_weight_fd=P("tp", "fsdp"),
#             rms_norm=P("tp"),
#             act_btd=P("fsdp", None, "tp"),
#             act_btf=P("fsdp", None, "tp"),
#             act_btnh=P("fsdp", None, "tp", None),
#             gate_weight=P(None, None),  # gate不分片，保证路由一致性
#             expert_weight_edf=P("ep", None, "tp"),  # 专家按ep分片，中间维度按tp分片
#             expert_weight_efd=P("ep", "tp", None),
#         )


# @dataclasses.dataclass(frozen=True)
# class ModelConfig:
#     num_layers: int
#     vocab_size: int
#     emb_dim: int
#     num_heads: int
#     head_dim: int
#     num_kv_heads: int
#     rope_theta: int
#     norm_eps: float
#     tie_word_embeddings: bool
#     # MoE configs
#     moe_intermediate_dim: int = 512
#     num_experts: int = 256
#     num_experts_per_tok: int = 8
#     shd_cfg: ShardingCfg = ShardingCfg.no_sharding()

#     @classmethod
#     def _from_param(cls, use_sharding: bool, **kwargs):
#         if use_sharding:
#             kwargs["shd_cfg"] = ShardingCfg.default()
#         return cls(**kwargs)

#     @classmethod
#     def ling_minimal(cls, use_sharding: bool = False):  # ling-minimal
#         return cls._from_param(
#             use_sharding,
#             num_layers=20,
#             vocab_size=157184,
#             emb_dim=2048,
#             moe_intermediate_dim=512,
#             num_experts=256,
#             num_experts_per_tok=8,
#             num_heads=16,
#             head_dim=128,
#             num_kv_heads=4,
#             norm_eps=1e-06,
#             rope_theta=600000,
#             tie_word_embeddings=False,
#         )



# def shard(x: jnp.ndarray, s: ShardingSpec):
#     mesh = get_abstract_mesh()
#     if not mesh.empty and len(mesh.axis_names) > 0:
#         return reshard(x, s)
#     return x


# class LayerCache(nnx.Module):
#     def __init__(self, cfg: ModelConfig, batch_size: int, cache_size: int, dtype: jnp.dtype):
#         cache_shape = (batch_size, cache_size, cfg.num_kv_heads, cfg.head_dim)
#         self.k_cache = shard(nnx.Cache(jnp.zeros(cache_shape, dtype=dtype)), cfg.shd_cfg.act_btnh)
#         self.v_cache = shard(nnx.Cache(jnp.zeros(cache_shape, dtype=dtype)), cfg.shd_cfg.act_btnh)
#         self.size = self.k_cache.shape[1]
#         batch_sharding = P(cfg.shd_cfg.act_btnh[0]) if cfg.shd_cfg.act_btnh else P(None)
#         self.start_ind = shard(nnx.Variable(-1 * jnp.ones((batch_size,), dtype=jnp.int32)), batch_sharding)
#         self.cur_ind = nnx.Variable(jnp.zeros((), dtype=jnp.int32))  # scalar for compute efficiency.


# Cache: TypeAlias = list[LayerCache]


# class Einsum(nnx.Module):
#     def __init__(self, einsum_str: str, shape: tuple[int, ...], *, shd: ShardingSpec, rngs: nnx.Rngs):
#         self.einsum_str = einsum_str
#         self.shape = shape
#         self.w = shard(nnx.Param(nnx.initializers.normal()(rngs.params(), shape)), shd)

#     @jax.named_scope("einsum")
#     def __call__(self, x: ArrayLike) -> Array:
#         return jnp.einsum(self.einsum_str, x, self.w.value)


# def _generate_pos_embeddings(
#     positions: jax.Array, head_dim: int, rope_theta: int = 1_000_000
# ) -> tuple[jax.Array, jax.Array]:
#     # Forked from: jax-llm-examples/qwen3/qwen3_jax/model.py;l=571
#     fraction = jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim
#     timescale = rope_theta**fraction
#     rotational_frequency = 1.0 / timescale
#     # Use high-precision einsum to prevent catastrophic bfloat16 rounding (ex: 257→256), as sin(257) differs from sin(256).
#     sinusoid_inp = jnp.einsum("BT,k->BTk", positions, rotational_frequency, precision=jax.lax.Precision.HIGHEST)
#     return jnp.sin(sinusoid_inp), jnp.cos(sinusoid_inp)


# def apply_rope(x: jax.Array, sin: jax.Array, cos: jax.Array) -> jax.Array:
#     assert x.ndim == 4 and sin.ndim == 3 and cos.ndim == 3
#     x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
#     # [B, T, head_dim] -> [B, h, T, head_dim]
#     sin, cos = sin[:, :, None, :], cos[:, :, None, :]
#     return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1).astype(x.dtype)


# class RMSNorm(nnx.Module):
#     def __init__(self, dim: int, cfg: ModelConfig, *, rngs: nnx.Rngs):
#         self.scale = shard(nnx.Param(nnx.initializers.ones_init()(rngs.params(), dim)), cfg.shd_cfg.rms_norm)
#         self.norm_eps = cfg.norm_eps

#     @jax.named_scope("rms_norm")
#     def __call__(self, x: Array) -> Array:
#         dtype = x.dtype
#         rms = jnp.sqrt(jnp.mean(jnp.astype(x, jnp.float32) ** 2, axis=-1, keepdims=True) + self.norm_eps)
#         return jnp.astype(self.scale.value * x / rms, dtype)


# def count_left_pads(x: jax.Array) -> int:
#     """Count left padding tokens."""
#     return jnp.sum(jnp.cumsum(x != 0, axis=-1) == 0, -1)


# def count_right_pads(x: jax.Array, pad_id) -> int:
#     result = jnp.where(
#         jnp.all(x == pad_id, axis=1), x.shape[1], jnp.argmin(jnp.flip(x == pad_id, axis=1).astype(jnp.int32), axis=1)
#     )
#     return jnp.max(result)


# def compute_positions_from_segment_ids(seg_ids):
#     return jax.vmap(lambda row: jnp.where(row != 0, jnp.arange(seg_ids.shape[1]) - jnp.argmax(row), 2**30))(seg_ids)



# class Attention(nnx.Module):
#     def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs):
#         self.shd_cfg = cfg.shd_cfg
#         einsum_fn = partial(Einsum, rngs=rngs)
#         self.q_proj = einsum_fn(
#             "BTD,DNH->BTNH", (cfg.emb_dim, cfg.num_heads, cfg.head_dim), shd=self.shd_cfg.q_weight_ndh
#         )
#         self.k_proj = einsum_fn(
#             "BSD,DKH->BSKH", (cfg.emb_dim, cfg.num_kv_heads, cfg.head_dim), shd=self.shd_cfg.kv_weight_ndh
#         )
#         self.v_proj = einsum_fn(
#             "BSD,DKH->BSKH", (cfg.emb_dim, cfg.num_kv_heads, cfg.head_dim), shd=self.shd_cfg.kv_weight_ndh
#         )
#         self.o_proj = einsum_fn(
#             "BTNH,NHD->BTD", (cfg.num_heads, cfg.head_dim, cfg.emb_dim), shd=self.shd_cfg.o_weight_nhd
#         )

#         self.q_norm = RMSNorm(cfg.head_dim, cfg, rngs=rngs)
#         self.k_norm = RMSNorm(cfg.head_dim, cfg, rngs=rngs)
#         self.n_rep = cfg.num_heads // cfg.num_kv_heads
#         self.scale = cfg.head_dim**-0.5

#     @jax.named_scope("attention")
#     def __call__(self, x: Array, cache: LayerCache | None, segment_ids: Array) -> Array:
#         query_proj = shard(self.q_norm(self.q_proj(x)), self.shd_cfg.act_btnh)  # [B, T, N, H]
#         key_proj = shard(self.k_norm(self.k_proj(x)), self.shd_cfg.act_btnh)  # [B, T, K, H]
#         value_proj = shard(self.v_proj(x), self.shd_cfg.act_btnh)  # [B, T, K, H]

#         # RoPE and Cache Logic
#         left_pads = count_left_pads(segment_ids)
#         left_pads = shard(left_pads, P(self.shd_cfg.act_btnh[0]))
#         cache.start_ind.value = jnp.where(cache.start_ind.value < 0, left_pads, cache.start_ind.value)
#         position_ids = compute_positions_from_segment_ids(segment_ids) + cache.cur_ind.value
#         sin, cos = _generate_pos_embeddings(position_ids, self.head_dim)
#         query_proj = apply_rope(query_proj, sin, cos)
#         key_proj = apply_rope(key_proj, sin, cos)

#         # Update K/V cache [B, S, K, H]
#         slice_indices = (0, cache.cur_ind.value, 0, 0)
#         cache.v_cache.value = jax.lax.dynamic_update_slice(cache.v_cache.value, value_proj, slice_indices)
#         cache.k_cache.value = jax.lax.dynamic_update_slice(cache.k_cache.value, key_proj, slice_indices)

#         b, t, n, h = query_proj.shape

#         # GQA reshape and attention logits
#         query_proj_gqa = query_proj.reshape((b, t, self.num_kv_heads, self.n_rep, h))
#         attn_logits = jnp.einsum("BTKGH,BSKH->BTSKG", query_proj_gqa, cache.k_cache.value) * self.scale

#         # Masking and Softmax
#         q_pos = cache.cur_ind.value + jnp.arange(t, dtype=jnp.int32)[None, :] - cache.start_ind.value[:, None]
#         ts = jnp.arange(cache.size, dtype=jnp.int32)  # (cache.size,)
#         kv_segment_ids = (ts[None, :] >= cache.start_ind.value[:, None]) & (ts[None, :] < cache.cur_ind.value + t)
#         k_pos = ts[None, :] - cache.start_ind.value[:, None]  # (b, cache.size)
#         causal_mask = k_pos[:, None, :] <= q_pos[:, :, None]
#         segment_mask = kv_segment_ids[:, None, :] == segment_ids[:, :, None]
#         final_mask = causal_mask & segment_mask  # (B, T, S)
#         attn_mask = final_mask[:, :, :, None, None]
#         attn_logits = jnp.where(attn_mask, attn_logits, _K_MASK)

#         # Softmax
#         attn_weights = jax.nn.softmax(attn_logits.astype(jnp.float32), axis=2).astype(attn_logits.dtype)
#         qkv = jnp.einsum("BTSKG,BSKH->BTKGH", attn_weights, cache.v_cache.value)
#         qkv = qkv.reshape((b, t, n, h))

#         cache.cur_ind.value = cache.cur_ind.value + t
#         return shard(self.o_proj(qkv), self.shd_cfg.act_btd)

#     @property
#     def head_dim(self):
#         return self.o_proj.shape[1]

#     @property
#     def num_heads(self):
#         return self.q_proj.shape[1]

#     @property
#     def num_kv_heads(self):
#         return self.k_proj.shape[1]


# class MoEGate(nnx.Module):
#     """MoE门控层：计算专家路由权重"""
#     def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs):
#         self.num_experts = cfg.num_experts
#         self.num_experts_per_tok = cfg.num_experts_per_tok
#         # Gate weight: [emb_dim, num_experts]
#         self.gate = shard(
#             nnx.Param(nnx.initializers.normal()(rngs.params(), (cfg.emb_dim, cfg.num_experts))),
#             cfg.shd_cfg.gate_weight
#         )

#     @jax.named_scope("moe_gate")
#     def __call__(self, x: Array) -> Tuple[Array, Array]:
#         """
#         Args:
#             x: [batch, seq, emb_dim]
#         Returns:
#             topk_weights: [batch * seq, num_experts_per_tok]
#             topk_ids: [batch * seq, num_experts_per_tok]
#         """
#         batch, seq, dim = x.shape
#         x_flat = x.reshape(-1, dim)  # [B*T, D]
        
#         # 计算门控logits并softmax
#         logits = x_flat @ self.gate.value  # [B*T, E]
#         scores = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
        
#         # Top-K选择
#         topk_weights, topk_ids = jax.lax.top_k(scores, self.num_experts_per_tok)
        
#         # 归一化权重
#         topk_weights = topk_weights / jnp.sum(topk_weights, axis=-1, keepdims=True)
#         topk_weights = topk_weights.astype(x.dtype)
        
#         return topk_weights, topk_ids


# class MoEMLP(nnx.Module):
#     """MoE MLP层：包含多个专家"""
#     def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs):
#         self.shd_cfg = cfg.shd_cfg
#         self.num_experts = cfg.num_experts
#         self.num_experts_per_tok = cfg.num_experts_per_tok
#         self.emb_dim = cfg.emb_dim
#         self.intermediate_dim = cfg.moe_intermediate_dim
        
#         # 门控层
#         self.gate = MoEGate(cfg, rngs=rngs)
        
#         # 专家参数: [num_experts, emb_dim, intermediate_dim]
#         self.gate_proj = shard(
#             nnx.Param(nnx.initializers.normal()(
#                 rngs.params(), 
#                 (cfg.num_experts, cfg.emb_dim, cfg.moe_intermediate_dim)
#             )),
#             cfg.shd_cfg.expert_weight_edf
#         )
#         self.up_proj = shard(
#             nnx.Param(nnx.initializers.normal()(
#                 rngs.params(), 
#                 (cfg.num_experts, cfg.emb_dim, cfg.moe_intermediate_dim)
#             )),
#             cfg.shd_cfg.expert_weight_edf
#         )
#         # [num_experts, intermediate_dim, emb_dim]
#         self.down_proj = shard(
#             nnx.Param(nnx.initializers.normal()(
#                 rngs.params(), 
#                 (cfg.num_experts, cfg.moe_intermediate_dim, cfg.emb_dim)
#             )),
#             cfg.shd_cfg.expert_weight_efd
#         )

#     @jax.named_scope("moe_mlp")
#     def __call__(self, x: Array) -> Array:
#         """
#         Args:
#             x: [batch, seq, emb_dim]
#         Returns:
#             output: [batch, seq, emb_dim]
#         """
#         batch, seq, dim = x.shape
#         x_flat = x.reshape(-1, dim)  # [B*T, D]
        
#         # 获取路由权重和专家ID
#         topk_weights, topk_ids = self.gate(x)  # [B*T, K], [B*T, K]
        
#         # 简化版MoE计算（非优化版本，用于验证）
#         output = self._naive_moe_forward(x_flat, topk_weights, topk_ids)
        
#         return output.reshape(batch, seq, dim)

#     def _naive_moe_forward(self, x: Array, topk_weights: Array, topk_ids: Array) -> Array:
#         """简化版MoE前向：逐token计算（仅用于功能验证）"""
#         num_tokens = x.shape[0]
        
#         def compute_token(token_idx):
#             token = x[token_idx]  # [D]
#             weights = topk_weights[token_idx]  # [K]
#             expert_ids = topk_ids[token_idx]  # [K]
            
#             def compute_expert(k):
#                 expert_id = expert_ids[k]
#                 weight = weights[k]
                
#                 # 获取专家权重
#                 gate_w = self.gate_proj.value[expert_id]  # [D, F]
#                 up_w = self.up_proj.value[expert_id]  # [D, F]
#                 down_w = self.down_proj.value[expert_id]  # [F, D]
                
#                 # SwiGLU计算
#                 gate_out = nnx.silu(token @ gate_w)  # [F]
#                 up_out = token @ up_w  # [F]
#                 hidden = gate_out * up_out  # [F]
#                 expert_out = hidden @ down_w  # [D]
                
#                 return weight * expert_out
            
#             # 对所有选中的专家求和
#             expert_outputs = jax.vmap(compute_expert)(jnp.arange(self.num_experts_per_tok))
#             return jnp.sum(expert_outputs, axis=0)
        
#         # 对所有token进行计算
#         outputs = jax.vmap(compute_token)(jnp.arange(num_tokens))
#         return outputs


# class MLP(nnx.Module):
#     def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs):
#         self.shd_cfg = cfg.shd_cfg
#         # 保留原有的dense MLP实现，用于非MoE层
#         einsum_fn = partial(Einsum, rngs=rngs)
#         self.gate_proj = einsum_fn(
#             "BTD,DF->BTF", (cfg.emb_dim, cfg.moe_intermediate_dim), shd=self.shd_cfg.ffw_weight_df
#         )
#         self.up_proj = einsum_fn(
#             "BTD,DF->BTF", (cfg.emb_dim, cfg.moe_intermediate_dim), shd=self.shd_cfg.ffw_weight_df
#         )
#         self.down_proj = einsum_fn(
#             "BTF,FD->BTD", (cfg.moe_intermediate_dim, cfg.emb_dim), shd=self.shd_cfg.ffw_weight_fd
#         )
        
#     @jax.named_scope("feed_forward")
#     def __call__(self, x: ArrayLike) -> Array:
#         activations = nnx.silu(self.gate_proj(x)) * self.up_proj(x)
#         activations = shard(activations, self.shd_cfg.act_btf)
#         outputs = self.down_proj(activations)
#         return outputs


# class DecoderLayer(nnx.Module):
#     def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs, use_moe: bool = True):
#         self.input_layernorm = RMSNorm(cfg.emb_dim, cfg, rngs=rngs)
#         self.attn = Attention(cfg=cfg, rngs=rngs)
#         self.post_attention_layernorm = RMSNorm(cfg.emb_dim, cfg, rngs=rngs)
#         # 根据配置选择MoE或普通MLP
#         if use_moe and cfg.num_experts > 1:
#             self.mlp = MoEMLP(cfg=cfg, rngs=rngs)
#         else:
#             self.mlp = MLP(cfg=cfg, rngs=rngs)

#     def __call__(self, x: Array, cache: LayerCache | None, segment_ids: Array) -> Array:
#         inputs_normalized = self.input_layernorm(x)
#         attn_output = x + self.attn(inputs_normalized, cache, segment_ids)
#         outputs = attn_output + self.mlp(self.post_attention_layernorm(attn_output))
#         return outputs


# class Ling2_mini(nnx.Module):
#     def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs):
#         self.embedder = shard(
#             nnx.Embed(num_embeddings=cfg.vocab_size, features=cfg.emb_dim, dtype=jnp.bfloat16, rngs=rngs),
#             cfg.shd_cfg.emb_vd,
#         )
#         self.out_emb_shd = None if get_abstract_mesh().empty else cfg.shd_cfg.act_btd
#         self.layers = nnx.List([DecoderLayer(cfg=cfg, rngs=rngs) for _ in range(cfg.num_layers)])
#         self.final_norm = RMSNorm(cfg.emb_dim, cfg, rngs=rngs)
#         self.lm_head = Einsum(
#             einsum_str="BTD,DV->BTV", shape=(cfg.emb_dim, cfg.vocab_size), shd=cfg.shd_cfg.emb_dv, rngs=rngs
#         )

#     def init_cache(
#         self, cfg: ModelConfig, batch_size: int, token_len: int, generate_steps: int, dtype: jnp.dtype = jnp.bfloat16
#     ) -> Cache:
#         cache_size = 2 ** math.ceil(math.log2(max(token_len + generate_steps, 1)))  # Pad for a sharding-friendly size.
#         return [LayerCache(cfg, batch_size, cache_size, dtype) for _ in range(cfg.num_layers)]

#     def __call__(self, tokens, segment_ids, cache, num_right_pads):
#         x = self.embedder.embedding.value.at[(tokens,)].get(out_sharding=self.out_emb_shd)
#         for i, layer in enumerate(self.layers):
#             x = layer(x, cache[i], segment_ids)
#         logits = self.lm_head(self.final_norm(x))
#         return logits


# @jax.jit
# def forward(model: nnx.Module, cache: Cache, tokens: Array, pad_id: int) -> tuple[Array, nnx.Cache]:
#     segment_ids = 1 * (tokens != pad_id)
#     num_right_pads = count_right_pads(tokens, pad_id)
#     logits = model(tokens, segment_ids, cache, num_right_pads)
#     target_ind = tokens.shape[-1] - num_right_pads - 1
#     return logits[:, target_ind], cache
