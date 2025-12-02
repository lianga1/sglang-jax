#!/usr/bin/env python3
"""
Simplified JAX Implementation for Ling-mini-2.0 Model

This is a more concise version showing the essential steps to:
1. Load the model config from config.json
2. Initialize the JAX model
3. Load weights from safetensors
4. Run inference
"""

import json
import logging
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import nnx
from transformers import PretrainedConfig

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.model_loader.loader import JAXModelLoader
# from sgl_jax.srt.utils.jax_utils import create_mesh

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_ling_mini_model(
    model_path: str = "/home/gcpuser/sky_workdir/sglang-jax/inclusionAI/Ling-mini-2.0",
    dtype: jnp.dtype = jnp.bfloat16,
) -> tuple[ModelConfig, nnx.Module]:
    """
    Load Ling-mini-2.0 model and weights using JAX.

    Args:
        model_path: Path to the model directory
        dtype: Data type for the model

    Returns:
        Tuple of (model_config, loaded_model)
    """
    logger.info(f"Loading Ling-mini-2.0 from: {model_path}")

    # Create model configuration
    model_config = ModelConfig(
        model_path=model_path,
        trust_remote_code=True,
        dtype="bfloat16",
    )

    logger.info(f"Model config created:")
    logger.info(f"  - Architecture: {model_config.hf_config.architectures}")
    logger.info(f"  - Hidden size: {model_config.hidden_size}")
    logger.info(f"  - Layers: {model_config.num_hidden_layers}")
    logger.info(f"  - Heads: {model_config.num_attention_heads} (Q), {model_config.num_key_value_heads} (KV)")
    logger.info(f"  - Vocab size: {model_config.vocab_size}")
    logger.info(f"  - Experts: {model_config.hf_config.num_experts} total, {model_config.hf_config.num_experts_per_tok} per token")
    logger.info(f"  - Data type: {dtype}")
    logger.info(f"  - EP size :{model_config.ep_size}")

    # Create JAX mesh for parallel execution
    import jax.sharding as shd
    mesh = jax.make_mesh(axis_shapes=(4, 1), axis_names=("tensor", "expert"),axis_types=(shd.AxisType.Explicit, shd.AxisType.Explicit))

    # Initialize RNG
    rng = nnx.Rngs(jax.random.PRNGKey(42))

    # Create model loader
    from sgl_jax.srt.configs.load_config import LoadConfig, LoadFormat
    load_config = LoadConfig(
        load_format=LoadFormat.JAX,
        download_dir=None,
        ignore_patterns=None,
        model_loader_extra_config=None,
    )

    loader = JAXModelLoader(load_config, rng, mesh)

    # Load model
    logger.info("Initializing model structure...")
    model = loader.load_model(model_config=model_config)

    logger.info("Model loaded successfully!")
    return model_config, model


def run_simple_inference(
    model_config: ModelConfig,
    model: nnx.Module,
    prompt_tokens: jnp.ndarray,
) -> jnp.ndarray:
    """
    Run a simple forward pass on the model.

    Args:
        model_config: The loaded model configuration
        model: The loaded JAX model
        prompt_tokens: Token IDs of shape (batch_size, seq_len)

    Returns:
        Logits from the model
    """
    from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
    from sgl_jax.srt.layers.logits_processor import LogitsMetadata

    batch_size, seq_len = prompt_tokens.shape

    # Create position IDs
    positions = jnp.arange(seq_len, dtype=jnp.int32)
    positions = jnp.broadcast_to(positions, (batch_size, seq_len))

    # Create forward batch
    forward_batch = ForwardBatch(
        input_ids=prompt_tokens,
        positions=positions,
    )

    # Create logits metadata
    logits_metadata = LogitsMetadata(
        batch_size=batch_size,
        num_tokens=seq_len,
        return_logprobs=False,
    )

    # Run model
    logger.info(f"Running forward pass...")
    logger.info(f"  - Input shape: {prompt_tokens.shape}")
    logger.info(f"  - Position shape: {positions.shape}")

    logits, _, _ = model(forward_batch, None, logits_metadata)

    logger.info(f"  - Output shape: {logits.shape}")
    logger.info(f"  - Vocab size: {model_config.vocab_size}")

    return logits


def decode_tokens_to_text(tokens: jnp.ndarray, tokenizer=None) -> str:
    """
    Decode token IDs to text.

    Args:
        tokens: Token IDs
        tokenizer: Optional tokenizer object (if available)

    Returns:
        Decoded text string
    """
    if tokenizer is not None:
        return tokenizer.decode(tokens.tolist())

    # Fallback: just return token IDs
    return f"Tokens: {tokens.tolist()}"


def main():
    """Main function demonstrating the complete workflow."""

    print("="*70)
    print("Ling-mini-2.0 JAX Inference Demo")
    print("="*70)
    print()

    # Step 1: Load model configuration
    print("Step 1: Loading model configuration...")
    model_path = "/home/gcpuser/sky_workdir/sglang-jax/inclusionAI/Ling-mini-2.0"

    # Verify config.json exists
    config_path = Path(model_path) / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config_data = json.load(f)
        print(f"  ✓ Config file found: {config_path}")
        print(f"  ✓ Architecture: {config_data.get('architectures')}")
        print(f"  ✓ Model type: {config_data.get('model_type')}")
    else:
        print(f"  ✗ Config file not found: {config_path}")
        return

    # Step 2: Load the model
    print()
    print("Step 2: Loading JAX model...")
    try:
        model_config, model = load_ling_mini_model(model_path)
        print("  ✓ Model loaded successfully!")
    except Exception as e:
        print(f"  ✗ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 3: Create example input
    print()
    print("Step 3: Creating example input...")
    batch_size = 2
    seq_len = 15
    vocab_size = model_config.vocab_size

    # Create random token IDs (in practice, use a real tokenizer)
    example_prompt = jax.random.randint(
        jax.random.PRNGKey(42),
        (batch_size, seq_len),
        0,
        min(1000, vocab_size),  # Use small vocab range for demo
        dtype=jnp.int32
    )

    print(f"  ✓ Created example tokens:")
    print(f"    - Batch size: {batch_size}")
    print(f"    - Sequence length: {seq_len}")
    print(f"    - Token range: [0, {min(1000, vocab_size)})")
    print(f"    - Shape: {example_prompt.shape}")
    print(f"    - First sequence: {example_prompt[0].tolist()}")

    # Step 4: Run inference
    print()
    print("Step 4: Running inference...")
    try:
        logits = run_simple_inference(model_config, model, example_prompt)
        print("  ✓ Inference completed!")
        print(f"    - Output shape: {logits.shape}")
        print(f"    - Expected shape: ({batch_size}, {seq_len}, {vocab_size})")

        # Get top predictions for the last token
        last_token_logits = logits[:, -1, :]
        top_k = 5
        top_tokens = jnp.argsort(last_token_logits, axis=-1)[..., -top_k:][..., ::-1]
        top_probs = jax.nn.softmax(last_token_logits)[..., -top_k:][..., ::-1]

        print()
        print("  Top 5 predictions for last token (first batch):")
        for i, (token, prob) in enumerate(zip(top_tokens[0], top_probs[0])):
            print(f"    {i+1}. Token {int(token):6d} (prob: {float(prob):.4f})")

    except Exception as e:
        print(f"  ✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()

    # Step 5: Model information summary
    print()
    print("="*70)
    print("Model Summary")
    print("="*70)
    print()
    print(f"Model: Ling-mini-2.0 (BailingMoeV2)")
    print(f"Location: {model_path}")
    print()
    print("Architecture:")
    print(f"  • Layers: {model_config.num_hidden_layers}")
    print(f"  • Hidden size: {model_config.hidden_size}")
    print(f"  • Attention heads: {model_config.num_attention_heads}")
    print(f"  • KV heads: {model_config.num_key_value_heads}")
    print(f"  • Head dimension: {model_config.head_dim}")
    print()
    print("MoE Configuration:")
    hf_config = model_config.hf_config
    print(f"  • Total experts: {hf_config.num_experts}")
    print(f"  • Experts per token: {hf_config.num_experts_per_tok}")
    print(f"  • Shared experts: {hf_config.num_shared_experts}")
    print(f"  • Expert groups: {hf_config.n_group}")
    print(f"  • Top-K groups: {hf_config.topk_group}")
    print()
    print("Tokenizer:")
    print(f"  • Vocabulary size: {model_config.vocab_size}")
    print(f"  • PAD token ID: {hf_config.pad_token_id}")
    print(f"  • EOS token ID: {hf_config.eos_token_id}")
    print()
    print("Computation:")
    print(f"  • Data type: bfloat16")
    print(f"  • Max positions: {hf_config.max_position_embeddings}")
    print(f"  • RoPE theta: {hf_config.rope_theta}")
    print(f"  • Use QK-Norm: {hf_config.use_qk_norm}")
    print()
    print("="*70)
    print("Demo completed!")
    print("="*70)


if __name__ == "__main__":
    # Show JAX info
    print("JAX Information:")
    print(f"  Version: {jax.__version__}")
    print(f"  Backend: {jax.default_backend()}")
    print(f"  Devices: {jax.devices()}")
    print()

    main()
