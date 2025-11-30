"""
Ling-mini-2.0 Model Implementation

This module provides a JAX-based implementation of the Ling-mini-2.0 model,
which uses the BailingMoeV2 (Mixture of Experts) architecture.

Architecture:
- 20 Transformer decoder layers
- Hidden size: 2048
- Attention heads: 16 (4 KV heads for GQA)
- MoE: 256 experts, 8 experts per token, 1 shared expert
- RoPE position encoding with QK-Norm
"""

import logging
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from transformers import PretrainedConfig

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.layers.embeddings import Embed, ParallelLMHead, RotaryEmbedding
from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.layers.moe import EPMoE, GateLogit, TopK
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.mem_cache.memory_pool import KVCache
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

logger = logging.getLogger(__name__)

init_fn = nnx.initializers.uniform()


class LingMiniAttention(nnx.Module):
    """Multi-head attention layer with QK-Norm and RoPE."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position_embeddings: int,
        rope_theta: float = 600000.0,
        rope_scaling: dict[str, Any] | None = None,
        head_dim: int | None = None,
        rms_norm_eps: float = None,
        use_qk_norm: bool = True,
        rotary_dim: int = 0,
        layer_id: int = 0,
        attention_bias: bool = False,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs = None,
        mesh: jax.sharding.Mesh = None,
    ):
        self.layer_id = layer_id
        assert num_heads % num_kv_heads == 0

        self.head_dim = head_dim or hidden_size // num_heads
        self.q_head_num = num_heads
        self.kv_head_num = num_kv_heads

        self.q_size = num_heads * self.head_dim
        self.kv_size = num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.use_qk_norm = use_qk_norm

        if use_qk_norm:
            self.q_norm = RMSNorm(
                self.head_dim,
                epsilon=rms_norm_eps,
                param_dtype=dtype,
                rngs=rngs,
            )
            self.k_norm = RMSNorm(
                self.head_dim,
                epsilon=rms_norm_eps,
                param_dtype=dtype,
                rngs=rngs,
            )
        else:
            self.q_norm = None
            self.k_norm = None

        # Use single QKV projection like BailingMoeV2
        self.qkv_proj = LinearBase(
            input_size=hidden_size,
            output_size=(num_heads + 2 * num_kv_heads) * self.head_dim,
            use_bias=attention_bias,
            kernel_axes=(None, "tensor"),
            rngs=rngs,
            params_dtype=dtype,
            mesh=mesh,
        )

        self.c_proj = LinearBase(
            input_size=num_heads * self.head_dim,
            output_size=hidden_size,
            use_bias=attention_bias,
            kernel_axes=("tensor", None),
            rngs=rngs,
            params_dtype=dtype,
            mesh=mesh,
        )

        self.rotary_emb = RotaryEmbedding(
            head_size=self.head_dim,
            rotary_dim=rotary_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
            is_neox_style=True,
            dtype=dtype,
        )

        self.attn = RadixAttention(
            num_heads=num_heads,
            head_dim=self.head_dim,
            scaling=self.scaling,
            num_kv_heads=num_kv_heads,
            layer_id=layer_id,
        )

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ) -> jax.Array:
        # QKV projection
        qkv, _ = self.qkv_proj(hidden_states)
        qkv = qkv.reshape(-1, self.q_head_num + 2 * self.kv_head_num, self.head_dim)

        # Split Q, K, V
        query_states = qkv[:, : self.q_head_num, :]
        key_states = qkv[:, self.q_head_num : self.q_head_num + self.kv_head_num, :]
        value_states = qkv[:, self.q_head_num + self.kv_head_num :, :]

        # Apply QK-Norm if enabled
        if self.use_qk_norm and self.q_norm is not None and self.k_norm is not None:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        # Apply RoPE
        query_states, key_states = self.rotary_emb(positions, query_states, key_states)

        # Attention
        attn_output, kv_fused = self.attn(
            query_states, key_states, value_states, forward_batch, token_to_kv_pool
        )

        # Output projection
        output, _ = self.c_proj(attn_output)
        return output, kv_fused


class LingMiniMLP(nnx.Module):
    """Standard MLP with SiLU activation."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        layer_id: int = 0,
        rngs: nnx.Rngs = None,
        dtype: jnp.dtype = jnp.bfloat16,
        mesh: jax.sharding.Mesh = None,
    ) -> None:
        self.layer_id = layer_id

        self.gate_proj = LinearBase(
            input_size=hidden_size,
            output_size=intermediate_size,
            kernel_axes=(None, "tensor"),
            use_bias=False,
            params_dtype=dtype,
            rngs=rngs,
            mesh=mesh,
        )

        self.up_proj = LinearBase(
            input_size=hidden_size,
            output_size=intermediate_size,
            kernel_axes=(None, "tensor"),
            use_bias=False,
            params_dtype=dtype,
            rngs=rngs,
            mesh=mesh,
        )

        self.down_proj = LinearBase(
            input_size=intermediate_size,
            output_size=hidden_size,
            kernel_axes=("tensor", None),
            use_bias=False,
            params_dtype=dtype,
            rngs=rngs,
            mesh=mesh,
        )

        self.act_fn = jax.nn.silu

    def __call__(self, hidden_states: jnp.ndarray):
        a1, _ = self.gate_proj(hidden_states)
        a2, _ = self.up_proj(hidden_states)
        intermediate_parallel = a2 * self.act_fn(a1)
        output, _ = self.down_proj(intermediate_parallel)
        return output


class LingMiniDecoderLayer(nnx.Module):
    """Decoder layer with support for both dense and MoE configurations."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs = None,
        mesh: jax.sharding.Mesh = None,
    ):
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size

        # RoPE configuration
        rope_theta = getattr(config, "rope_theta", 600000.0)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 32768)
        self.head_dim = getattr(config, "head_dim", None)
        use_qk_norm = getattr(config, "use_qk_norm", True)

        # Calculate rotary dimension
        if hasattr(config, "partial_rotary_factor"):
            rotary_dim = int(self.head_dim * config.partial_rotary_factor)
        elif hasattr(config, "rotary_dim"):
            rotary_dim = config.rotary_dim
        else:
            rotary_dim = self.head_dim

        # Attention layer
        self.self_attn = LingMiniAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            head_dim=self.head_dim,
            rms_norm_eps=config.rms_norm_eps,
            use_qk_norm=use_qk_norm,
            rotary_dim=rotary_dim,
            layer_id=layer_id,
            attention_bias=getattr(config, "attention_bias", False),
            dtype=dtype,
            rngs=rngs,
            mesh=mesh,
        )

        # Determine if this layer uses MoE or dense MLP
        first_k_dense_replace = getattr(config, "first_k_dense_replace", 1)

        if layer_id < first_k_dense_replace:
            # Use standard MLP for first k layers
            self.mlp = LingMiniMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                layer_id=layer_id,
                dtype=dtype,
                rngs=rngs,
                mesh=mesh,
            )
            self.is_moe_layer = False
            self.moe_gate = None
        else:
            # Use MoE for later layers
            num_shared_experts = getattr(config, "num_shared_experts", 1)
            router_dtype = getattr(config, "router_dtype", None)

            if router_dtype is None:
                router_dtype = jnp.bfloat16
            elif router_dtype == "fp32":
                router_dtype = jnp.float32
            else:
                router_dtype = jnp.bfloat16

            # MoE gate
            self.moe_gate = GateLogit(
                input_size=config.hidden_size,
                num_experts=config.num_experts,
                enable_expert_bias=getattr(config, "moe_router_enable_expert_bias", True),
                weight_dtype=router_dtype,
                score_func=getattr(config, "score_function", "sigmoid"),
            )

            # Top-K selection
            self.topk = TopK(
                topk=config.num_experts_per_tok,
                renormalize=getattr(config, "norm_topk_prob", True),
                num_expert_group=getattr(config, "n_group", 8),
                topk_group=getattr(config, "topk_group", 4),
                routed_scaling_factor=getattr(config, "routed_scaling_factor", 2.5),
            )

            # Expert-parallel MoE
            self.mlp = EPMoE(
                config=config,
                num_experts=config.num_experts,
                num_experts_per_tok=config.num_experts_per_tok,
                intermediate_dim=config.moe_intermediate_size,
                mesh=mesh,
                ep_size=getattr(config, "ep_size", 1),
                weight_dtype=dtype,
                dtype=dtype,
                layer_id=layer_id,
            )

            # Shared experts
            if num_shared_experts > 0:
                self.shared_experts = LingMiniMLP(
                    hidden_size=config.hidden_size,
                    intermediate_size=getattr(
                        config,
                        "moe_shared_expert_intermediate_size",
                        config.moe_intermediate_size,
                    )
                    * num_shared_experts,
                    layer_id=layer_id,
                    dtype=dtype,
                    rngs=rngs,
                    mesh=mesh,
                )
            else:
                self.shared_experts = None

            self.is_moe_layer = True

        # Layer norms
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype,
            rngs=rngs,
        )

        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        residual: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states += residual
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

        # Self attention
        hidden_states, kv_fused = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            token_to_kv_pool=token_to_kv_pool,
        )

        hidden_states += residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        # MLP or MoE
        if self.is_moe_layer:
            # Add shared experts if present
            if self.shared_experts is not None:
                shared_output = self.shared_experts(hidden_states)
            else:
                shared_output = None

            # MoE routing
            router_logits = self.moe_gate(hidden_states)

            # Get correction bias
            correction_bias = (
                self.moe_gate.bias.value if self.moe_gate.bias is not None else None
            )

            # Top-K selection
            topk_weights, topk_ids = self.topk(router_logits, correction_bias)

            # Apply MoE
            hidden_states = self.mlp(hidden_states, topk_weights, topk_ids)

            # Add shared expert output
            if shared_output is not None:
                hidden_states = hidden_states + shared_output
        else:
            # Standard MLP
            hidden_states = self.mlp(hidden_states)

        return hidden_states, residual, kv_fused


class LingMiniModel(nnx.Module):
    """Complete Ling-mini-2.0 transformer model."""

    def __init__(
        self,
        config: PretrainedConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs = None,
        mesh: jax.sharding.Mesh = None,
    ):
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Token embeddings
        self.embed_tokens = Embed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            rngs=rngs,
            dtype=dtype,
            param_dtype=dtype,
            kernel_axes=("tensor", None),
            mesh=mesh,
        )

        # Decoder layers
        self.layers = nnx.data(
            [
                LingMiniDecoderLayer(
                    config=config,
                    layer_id=i,
                    dtype=dtype,
                    rngs=rngs,
                    mesh=mesh,
                )
                for i in range(config.num_hidden_layers)
            ]
        )

        # Final layer norm
        self.norm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ) -> tuple[jax.Array, list]:
        hidden_states = self.embed_tokens(forward_batch.input_ids)
        residual = None
        layers_kv_fused = []

        for layer in self.layers:
            hidden_states, residual, kv_fused = layer(
                forward_batch.positions,
                hidden_states,
                forward_batch,
                token_to_kv_pool,
                residual,
            )
            layers_kv_fused.append(kv_fused)

        if residual is not None:
            hidden_states += residual

        hidden_states = self.norm(hidden_states)
        return hidden_states, layers_kv_fused


class LingMiniForCausalLM(nnx.Module):
    """Causal language model for Ling-mini-2.0."""

    def __init__(
        self,
        config: PretrainedConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs = None,
        mesh: jax.sharding.Mesh = None,
    ):
        self.mesh = mesh
        self.config = config
        self.dtype = dtype

        logger.info("LingMiniForCausalLM config dtype: %s", self.dtype)

        # Main model
        self.model = LingMiniModel(config, dtype=self.dtype, rngs=rngs, mesh=mesh)

        # LM head
        if not getattr(self.config, "tie_word_embeddings", False):
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                dtype=self.dtype,
                param_dtype=self.dtype,
                kernel_axes=("tensor", None),
                rngs=rngs,
            )

        # Logits processor
        self.logits_processor = LogitsProcessor(config.vocab_size, mesh=self.mesh)

    def load_weights(self, model_config: ModelConfig, rng_key: jax.Array):
        """Load weights from HuggingFace safetensors files."""
        self.rng = nnx.Rngs(rng_key)

        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )

        weight_mappings = self._create_ling_mini_weight_mappings()
        loader.load_weights_from_safetensors(weight_mappings)
        logger.info("Ling-mini-2.0 weights loaded successfully!")

    def _create_ling_mini_weight_mappings(self) -> dict:
        """Create weight mappings from HF format to JAX format."""
        mappings = {
            "model.word_embeddings.weight": WeightMapping(
                target_path="model.embed_tokens.embedding",
                sharding=("tensor", None),
                transpose=False,
            ),
            "model.norm.weight": WeightMapping(
                target_path="model.norm.scale",
                sharding=(None,),
                transpose=False,
            ),
        }

        # LM head
        if not getattr(self.config, "tie_word_embeddings", False):
            mappings["lm_head.weight"] = WeightMapping(
                target_path="lm_head.embedding",
                sharding=("tensor", None),
                transpose=False,
            )

        # Decoder layers
        num_layers = self.config.num_hidden_layers
        first_k_dense_replace = self.config.first_k_dense_replace

        for layer_idx in range(num_layers):
            layer_mappings = self._create_layer_mappings(
                layer_idx, layer_idx < first_k_dense_replace
            )
            mappings.update(layer_mappings)

        return mappings

    def _create_layer_mappings(self, layer_idx: int, is_mlp_layer: bool) -> dict:
        """Create weight mappings for a single layer."""
        prefix = f"model.layers.{layer_idx}"
        target_prefix = f"model.layers.{layer_idx}"

        mappings = {
            f"{prefix}.input_layernorm.weight": WeightMapping(
                target_path=f"{target_prefix}.input_layernorm.scale",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.post_attention_layernorm.weight": WeightMapping(
                target_path=f"{target_prefix}.post_attention_layernorm.scale",
                sharding=(None,),
                transpose=False,
            ),
            # QKV projection (single weight for Q, K, V)
            f"{prefix}.attention.query_key_value.weight": WeightMapping(
                target_path=[
                    f"{target_prefix}.self_attn.q_proj.weight",
                    f"{target_prefix}.self_attn.k_proj.weight",
                    f"{target_prefix}.self_attn.v_proj.weight",
                ],
                sharding=(None, "tensor"),
                transpose=True,
                kv_head_padding=True,
            ),
            # Output projection
            f"{prefix}.attention.dense.weight": WeightMapping(
                target_path=f"{target_prefix}.self_attn.c_proj.weight",
                sharding=("tensor", None),
                transpose=True,
            ),
        }

        # QK-Norm layers
        if getattr(self.config, "use_qk_norm", True):
            mappings[f"{prefix}.attention.query_layernorm.weight"] = WeightMapping(
                target_path=f"{target_prefix}.self_attn.q_norm.scale",
                sharding=(None,),
                transpose=False,
            )
            mappings[f"{prefix}.attention.key_layernorm.weight"] = WeightMapping(
                target_path=f"{target_prefix}.self_attn.k_norm.scale",
                sharding=(None,),
                transpose=False,
            )

        # MLP or MoE
        if is_mlp_layer:
            # Standard MLP
            mlp_mappings = {
                f"{prefix}.mlp.gate_proj.weight": WeightMapping(
                    target_path=f"{target_prefix}.mlp.gate_proj.weight",
                    sharding=(None, "tensor"),
                    transpose=True,
                ),
                f"{prefix}.mlp.up_proj.weight": WeightMapping(
                    target_path=f"{target_prefix}.mlp.up_proj.weight",
                    sharding=(None, "tensor"),
                    transpose=True,
                ),
                f"{prefix}.mlp.down_proj.weight": WeightMapping(
                    target_path=f"{target_prefix}.mlp.down_proj.weight",
                    sharding=("tensor", None),
                    transpose=True,
                ),
            }
            mappings.update(mlp_mappings)
        else:
            # MoE gate
            mappings[f"{prefix}.mlp.gate.weight"] = WeightMapping(
                target_path=f"{target_prefix}.moe_gate.kernel",
                sharding=(None, None),
                transpose=True,
            )

            # Expert bias
            if getattr(self.config, "moe_router_enable_expert_bias", False):
                mappings[f"{prefix}.mlp.gate.expert_bias"] = WeightMapping(
                    target_path=f"{target_prefix}.moe_gate.bias",
                    sharding=(None,),
                    transpose=False,
                )

            # Shared experts
            num_shared_experts = getattr(self.config, "num_shared_experts", 0)
            if num_shared_experts > 0:
                shared_experts_mappings = {
                    f"{prefix}.mlp.shared_experts.gate_proj.weight": WeightMapping(
                        target_path=f"{target_prefix}.shared_experts.gate_proj.weight",
                        sharding=(None, "tensor"),
                        transpose=True,
                    ),
                    f"{prefix}.mlp.shared_experts.up_proj.weight": WeightMapping(
                        target_path=f"{target_prefix}.shared_experts.up_proj.weight",
                        sharding=(None, "tensor"),
                        transpose=True,
                    ),
                    f"{prefix}.mlp.shared_experts.down_proj.weight": WeightMapping(
                        target_path=f"{target_prefix}.shared_experts.down_proj.weight",
                        sharding=("tensor", None),
                        transpose=True,
                    ),
                }
                mappings.update(shared_experts_mappings)

            # Expert weights
            num_experts = getattr(self.config, "num_experts", 256)
            for expert_type in ["gate_proj", "up_proj", "down_proj"]:
                target_name = {
                    "gate_proj": "wi_0",
                    "up_proj": "wi_1",
                    "down_proj": "wo",
                }[expert_type]

                expert_keys = [
                    f"{prefix}.mlp.experts.{i}.{expert_type}.weight"
                    for i in range(num_experts)
                ]

                # Determine sharding
                if expert_type == "down_proj":
                    sharding = ("expert", "tensor", None)
                else:
                    sharding = ("expert", None, "tensor")

                mappings[f"__MOE_EXPERTS__{prefix}.mlp.{target_name}"] = WeightMapping(
                    target_path=[f"{target_prefix}.mlp.{target_name}"] + expert_keys,
                    sharding=sharding,
                    transpose=True,
                )

        return mappings

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        logits_metadata: LogitsMetadata,
    ):
        """Forward pass."""
        hidden_states, layers_kv_fused = self.model(forward_batch, token_to_kv_pool)

        # Compute logits
        if not getattr(self.config, "tie_word_embeddings", False):
            output = self.logits_processor(
                hidden_states, self.lm_head, logits_metadata
            )
        else:
            output = self.logits_processor(
                hidden_states, self.model.embed_tokens, logits_metadata
            )

        return output, layers_kv_fused, True


# Export the main model class
EntryClass = [LingMiniForCausalLM]


def params(model: LingMiniForCausalLM) -> dict:
    """
    Extract parameters from the model.

    Args:
        model: The LingMiniForCausalLM model

    Returns:
        Dictionary of model parameters
    """
    # Extract parameters using nnx.state
    # This returns a PyTree of all trainable parameters
    return nnx.state(model)


def run_model(
    model: LingMiniForCausalLM,
    input_ids: jnp.ndarray,
    positions: jnp.ndarray | None = None,
):
    """
    Run the model with given inputs.

    Args:
        model: The LingMiniForCausalLM model
        input_ids: Token IDs of shape (batch_size, seq_len)
        positions: Optional position IDs

    Returns:
        Model output dictionary
    """
    from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
    from sgl_jax.srt.layers.logits_processor import LogitsMetadata

    batch_size, seq_len = input_ids.shape

    # Create positions if not provided
    if positions is None:
        positions = jnp.arange(seq_len, dtype=jnp.int32)
        positions = jnp.broadcast_to(positions, (batch_size, seq_len))

    # Create forward batch
    forward_batch = ForwardBatch(
        input_ids=input_ids,
        positions=positions,
    )

    # Create logits metadata
    logits_metadata = LogitsMetadata(
        batch_size=batch_size,
        num_tokens=seq_len,
        return_logprobs=False,
    )

    # Run model
    # Note: token_to_kv_pool should be properly initialized in real usage
    output, _, _ = model(forward_batch, None, logits_metadata)

    return output
