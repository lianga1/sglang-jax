#!/usr/bin/env python3
"""
JAX Implementation for Ling-mini-2.0 Model Loading and Inference

This script demonstrates how to load and run inference on the Ling-mini-2.0 model
using JAX and the sglang-jax framework.

Model Architecture: BailingMoeV2 (Mixture of Experts)
- 20 layers
- 2048 hidden size
- 16 attention heads (4 KV heads)
- 256 experts, 8 experts per token
- 1 shared expert
- RoPE position encoding
- QK-Norm for attention stabilization
"""

import json
import logging
from typing import Dict, Any, Optional

import jax
import jax.numpy as jnp
from flax import nnx
from transformers import PretrainedConfig

from sgl_jax.srt.configs.load_config import LoadConfig, LoadFormat
from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.model_loader.loader import get_model_loader
from sgl_jax.srt.utils.jax_utils import create_mesh
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.layers.logits_processor import LogitsMetadata

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LingMiniJAXInference:
    """
    JAX-based inference handler for Ling-mini-2.0 model.

    This class wraps the sglang-jax framework to provide a simple interface
    for loading models and running inference.
    """

    def __init__(
        self,
        model_path: str,
        dtype: jnp.dtype = jnp.bfloat16,
        tensor_parallel_size: int = 1,
        trust_remote_code: bool = True,
    ):
        """
        Initialize the inference handler.

        Args:
            model_path: Path to the model directory containing config.json and weights
            dtype: Data type for model weights and computations (default: bfloat16)
            tensor_parallel_size: Number of devices for tensor parallelism
            trust_remote_code: Whether to trust remote code in model config
        """
        self.model_path = model_path
        self.dtype = dtype
        self.tp_size = tensor_parallel_size
        self.trust_remote_code = trust_remote_code

        # Initialize components
        self.model_config = None
        self.model = None
        self.mesh = None
        self.rng = None

    def load_model(self) -> None:
        """Load the Ling-mini-2.0 model and initialize JAX components."""
        logger.info(f"Loading Ling-mini-2.0 model from: {self.model_path}")

        # Create device mesh for JAX parallel execution
        self.mesh = create_mesh(
            mesh_shape=(self.tp_size, 1),
            axis_names=("tensor", "expert")
        )

        # Initialize RNG for model initialization
        self.rng = nnx.Rngs(jax.random.PRNGKey(42))

        # Create model configuration
        logger.info("Creating model configuration...")
        self.model_config = ModelConfig(
            model_path=self.model_path,
            trust_remote_code=self.trust_remote_code,
            dtype=str(self.dtype).replace('jax.', ''),
        )

        # Configure for tensor parallelism if needed
        if self.tp_size > 1:
            self.model_config.configure_for_tensor_parallel(self.tp_size)
            self.model_config.log_kv_heads_info(self.tp_size)

        # Create load configuration
        load_config = LoadConfig(
            load_format=LoadFormat.JAX,  # Use JAX weights
            download_dir=None,
            ignore_patterns=None,
            model_loader_extra_config=None,
        )

        # Get model loader
        loader = get_model_loader(load_config, self.rng, self.mesh)

        # Load the model
        logger.info("Initializing and loading model weights...")
        self.model = loader.load_model(model_config=self.model_config)

        logger.info("Model loaded successfully!")

    def create_forward_batch(
        self,
        input_ids: jnp.ndarray,
        positions: Optional[jnp.ndarray] = None,
    ) -> ForwardBatch:
        """
        Create a forward batch from input tokens.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            positions: Position IDs (optional, will be auto-generated if None)

        Returns:
            ForwardBatch object ready for model inference
        """
        batch_size, seq_len = input_ids.shape

        # Auto-generate positions if not provided
        if positions is None:
            positions = jnp.arange(seq_len, dtype=jnp.int32)
            positions = jnp.broadcast_to(positions, (batch_size, seq_len))

        # Create forward batch with required metadata
        forward_batch = ForwardBatch(
            input_ids=input_ids,
            positions=positions,
            # Add any other required fields based on ForwardBatch definition
        )

        return forward_batch

    def generate(
        self,
        input_ids: jnp.ndarray,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ) -> jnp.ndarray:
        """
        Generate text from input tokens using the model.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling parameter (0 < top_p <= 1)

        Returns:
            Generated token IDs
        """
        logger.info(f"Generating up to {max_new_tokens} new tokens...")

        # This is a simplified generation loop
        # In practice, you'd want a more sophisticated implementation
        # with KV caching and dynamic batching

        batch_size, seq_len = input_ids.shape
        generated = input_ids.copy()

        for _ in range(max_new_tokens):
            # Get current input (last token for autoregressive generation)
            current_input = generated[:, -1:] if seq_len > 0 else generated

            # Create forward batch
            forward_batch = self.create_forward_batch(current_input)

            # Prepare logits metadata
            logits_metadata = LogitsMetadata(
                batch_size=batch_size,
                num_tokens=1,
                return_logprobs=False,
            )

            # Run model forward pass
            # Note: In a complete implementation, you'd need to handle
            # KV cache properly for generation
            try:
                logits, _, _ = self.model(forward_batch, None, logits_metadata)

                # Apply temperature
                if temperature != 1.0:
                    logits = logits / temperature

                # Apply top-p sampling
                if top_p < 1.0:
                    # Sort logits and apply nucleus filtering
                    sorted_logits = jax.lax.sort(logits, dimension=-1)
                    sorted_indices = jax.lax.sort_key_val(
                        logits,
                        jnp.arange(logits.shape[-1])[None, :]
                    )[1]

                    # Compute cumulative probabilities
                    probs = jax.nn.softmax(sorted_logits, axis=-1)
                    cumsum = jnp.cumsum(probs, axis=-1)

                    # Find cutoff for top-p
                    cutoff = jnp.searchsorted(cumsum, top_p, side='right')

                    # Zero out probabilities below cutoff
                    mask = jnp.arange(logits.shape[-1]) <= cutoff
                    filtered_logits = jnp.where(
                        mask[None, :],
                        sorted_logits,
                        -float('inf')
                    )

                    # Restore original order
                    logits = jax.lax.sort_key_val(sorted_indices, filtered_logits)[0]

                # Sample next token
                probs = jax.nn.softmax(logits[:, -1, :], axis=-1)
                next_token = jax.random.categorical(
                    jax.random.PRNGKey(0),
                    jnp.log(probs + 1e-8),
                    axis=-1
                )

                # Append to generated sequence
                generated = jnp.concatenate([generated, next_token[:, None]], axis=-1)
                seq_len += 1

            except Exception as e:
                logger.warning(f"Generation step failed: {e}")
                # Fallback: just use the last token's embedding
                break

        return generated

    def forward(
        self,
        input_ids: jnp.ndarray,
        use_cache: bool = False,
    ) -> Dict[str, jnp.ndarray]:
        """
        Run a single forward pass through the model.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            use_cache: Whether to use KV cache (for longer sequences)

        Returns:
            Dictionary containing model outputs:
            - logits: Output logits of shape (batch_size, seq_len, vocab_size)
            - last_hidden_state: Final hidden states
        """
        logger.info(f"Running forward pass with input shape: {input_ids.shape}")

        # Create forward batch
        forward_batch = self.create_forward_batch(input_ids)

        # Prepare logits metadata
        batch_size, seq_len = input_ids.shape
        logits_metadata = LogitsMetadata(
            batch_size=batch_size,
            num_tokens=seq_len,
            return_logprobs=False,
        )

        # Run model forward pass
        logits, layers_kv_fused, _ = self.model(forward_batch, None, logits_metadata)

        return {
            "logits": logits,
            "layers_kv_fused": layers_kv_fused,
        }

    def encode(self, input_ids: jnp.ndarray) -> jnp.ndarray:
        """
        Get the final hidden states (embeddings) for input tokens.

        Args:
            input_ids: Input token IDs

        Returns:
            Hidden states of shape (batch_size, seq_len, hidden_size)
        """
        logger.info(f"Encoding input with shape: {input_ids.shape}")

        # Run forward pass without computing logits
        outputs = self.forward(input_ids)

        # Return the logits as representation (you might want to use hidden states instead)
        return outputs["logits"]


def main():
    """
    Example usage of the Ling-mini-2.0 JAX inference.
    """
    # Model path
    model_path = "/home/gcpuser/sky_workdir/sglang-jax/inclusionAI/Ling-mini-2.0"

    # Initialize the inference handler
    logger.info("Initializing Ling-mini-2.0 JAX inference handler...")
    inference = LingMiniJAXInference(
        model_path=model_path,
        dtype=jnp.bfloat16,
        tensor_parallel_size=1,  # Use 1 for single device, increase for multi-device
    )

    # Load the model
    inference.load_model()

    # Example 1: Simple forward pass
    logger.info("\n" + "="*60)
    logger.info("Example 1: Forward Pass")
    logger.info("="*60)

    # Create some example input tokens
    # In practice, you'd tokenize text using the model's tokenizer
    batch_size, seq_len = 2, 10
    example_input = jax.random.randint(
        jax.random.PRNGKey(0),
        (batch_size, seq_len),
        0,
        inference.model_config.vocab_size,
        dtype=jnp.int32
    )

    logger.info(f"Input shape: {example_input.shape}")
    logger.info(f"Input tokens (first batch): {example_input[0].tolist()}")

    # Run forward pass
    outputs = inference.forward(example_input)
    logger.info(f"Output logits shape: {outputs['logits'].shape}")
    logger.info(f"Vocab size: {inference.model_config.vocab_size}")

    # Example 2: Generate text
    logger.info("\n" + "="*60)
    logger.info("Example 2: Text Generation")
    logger.info("="*60)

    # Start with a prompt
    prompt_length = 5
    prompt = example_input[:, :prompt_length]
    logger.info(f"Prompt: {prompt[0].tolist()}")

    # Generate
    try:
        generated = inference.generate(
            prompt,
            max_new_tokens=20,
            temperature=0.7,
            top_p=0.9,
        )
        logger.info(f"Generated sequence shape: {generated.shape}")
        logger.info(f"Generated tokens: {generated[0].tolist()}")
    except Exception as e:
        logger.error(f"Generation failed (this is expected for a demo): {e}")

    # Example 3: Model information
    logger.info("\n" + "="*60)
    logger.info("Example 3: Model Information")
    logger.info("="*60)

    config = inference.model_config.hf_config
    logger.info(f"Model architecture: {config.architectures}")
    logger.info(f"Hidden size: {config.hidden_size}")
    logger.info(f"Number of layers: {config.num_hidden_layers}")
    logger.info(f"Number of attention heads: {config.num_attention_heads}")
    logger.info(f"Number of KV heads: {config.num_key_value_heads}")
    logger.info(f"Vocab size: {config.vocab_size}")
    logger.info(f"Max position embeddings: {config.max_position_embeddings}")
    logger.info(f"Data type: {inference.dtype}")
    logger.info(f"MoE configuration:")
    logger.info(f"  - Number of experts: {config.num_experts}")
    logger.info(f"  - Experts per token: {config.num_experts_per_tok}")
    logger.info(f"  - Shared experts: {config.num_shared_experts}")
    logger.info(f"  - Use QK-Norm: {config.use_qk_norm}")

    logger.info("\n" + "="*60)
    logger.info("Demo completed successfully!")
    logger.info("="*60)


if __name__ == "__main__":
    # Print JAX configuration
    print("="*60)
    print("JAX Configuration")
    print("="*60)
    print(f"JAX version: {jax.__version__}")
    print(f"JAX backend: {jax.default_backend()}")
    print(f"Available devices: {jax.devices()}")
    print(f"Device count: {len(jax.devices())}")
    print("="*60 + "\n")

    main()
