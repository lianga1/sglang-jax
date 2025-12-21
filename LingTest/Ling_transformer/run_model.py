import math
import time
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from typing import List, Tuple, Optional

class GreedySampler:
    """Greedy sampling strategy."""
    def __init__(self, temperature: float = 1.0, top_p: float = 0.9, top_k: int = 50):
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

    def __call__(self, logits: torch.Tensor, key: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Sample next tokens from logits.

        Args:
            logits: [batch_size, vocab_size] or [batch_size, 1, vocab_size]
            key: Random key (not used in greedy sampling, kept for API compatibility)

        Returns:
            next_tokens: [batch_size, 1]
        """
        if logits.dim() == 3:
            logits = logits.squeeze(1)  # Remove sequence dimension if present

        # Apply temperature
        if self.temperature != 1.0:
            logits = logits / self.temperature

        # Top-k filtering
        if self.top_k > 0:
            top_k = min(self.top_k, logits.size(-1))
            top_k_logits, top_k_indices = torch.topk(logits, top_k)
            # Create a mask for top-k tokens
            mask = torch.full_like(logits, float('-inf'))
            mask.scatter_(-1, top_k_indices, top_k_logits)
            logits = mask

        # Top-p (nucleus) filtering
        if self.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > self.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # Apply mask
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits = logits.masked_fill(indices_to_remove, float('-inf'))

        # Sample
        probs = torch.softmax(logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1)

        return next_tokens


def tokenize(tokenizer, input: List[str]) -> torch.Tensor:
    """
    Tokenize input texts with the same padding strategy as JAX version.
    Right-align, left-padding to the max token length.
    """
    pad_idx = tokenizer.pad_token_id
    lines = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": l}], tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        for l in input
    ]
    lines = [tokenizer.encode(line) for line in lines]
    max_len = max(len(line) for line in lines)
    return torch.tensor([np.pad(l, (max_len - len(l), 0), constant_values=pad_idx) for l in lines])


def run_model():
    """
    Run model inference with step-by-step forward passes exposed.
    Based on JAX version but using transformers library.
    """
    # Model configuration
    model_ckpt_path = "inclusionAI/Ling-mini-2.0"

    # Load tokenizer and model
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt_path)
    config = AutoConfig.from_pretrained(model_ckpt_path,trust_remote_code = True)
    config._attn_implementation = "eager"
    model = AutoModelForCausalLM.from_pretrained(
        model_ckpt_path,
        torch_dtype=torch.float32,
        config = config,
        device_map="auto",
        trust_remote_code=True
        
    )

    # Set model to evaluation mode
    model.eval()

    # Input queries
    query = [
        "Why is the sky blue instead of any other color like purple?",
        "Who am I?",
    ]

    # Tokenize input (same as JAX version)
    print("\nTokenizing input...")
    tokens = tokenize(tokenizer, query)
    batch_size, token_len = tokens.shape
    print(f"Batch size: {batch_size}, Token length: {token_len}")

    # Move tokens to device
    tokens = tokens.to(model.device)

    # Generation parameters
    generate_steps = 100

    # Initialize past key values (cache) for transformers
    # In transformers, we can use model.generate or manually handle past_key_values
    # For step-by-step control, we'll use past_key_values

    # Set up random key for sampling (not used in greedy but kept for API compatibility)
    key = torch.tensor(0)

    # Create sampler
    sampler = GreedySampler(temperature=1.0, top_p=0.8, top_k=10)

    # Initialize KV cache
    # For transformers, we initialize past_key_values as None and it will be created automatically
    past_key_values = None

    # ============================================================================
    # PREFILL PHASE
    # ============================================================================
    print("\n=== PREFILL PHASE ===")

    # Run forward pass with input tokens
    # In transformers, we need to prepare inputs with past_key_values
    input_token_count = token_len

    with torch.no_grad():
        # Forward pass for prefill
        outputs = model(
            input_ids=tokens,
            past_key_values=past_key_values,
            use_cache=True,
        )

        logits = outputs.logits
        past_key_values = outputs.past_key_values

        print(f"Logits shape: {logits.shape}")

        # Get logits of the last valid token (same as JAX version)
        # Right-aligned: the last token before padding is the actual last token
        target_ind = token_len - 1
        last_logits = logits[:, target_ind, :]
        print(f"Last logits shape: {last_logits.shape}")

        # Sample next tokens
        next_tokens = sampler(last_logits, key=key)
        print(f"Next tokens (first step): {next_tokens.squeeze(-1).tolist()}")

    # ============================================================================
    # DECODE PHASE
    # ============================================================================
    print("\n=== DECODE PHASE ===")

    tokens_list = [next_tokens]
    finished = torch.zeros((batch_size,), dtype=torch.bool)

    for i in range(generate_steps):
        print(f"\n--- Step {i} ---")

        with torch.no_grad():
            # Forward pass with only the new token and past key values
            outputs = model(
                input_ids=next_tokens,
                past_key_values=past_key_values,
                use_cache=True,
            )

            logits = outputs.logits
            past_key_values = outputs.past_key_values

            print(f"Logits shape: {logits.shape}")
            print(f"Logits stats: Min={logits.min().item():.4f}, Max={logits.max().item():.4f}, NaN?={torch.isnan(logits).any().item()}")

            # Get logits of the last token (should be shape [batch_size, vocab_size])
            last_logits = logits[:, 0, :]  # Only one token generated in this step
            print(f"Last logits shape: {last_logits.shape}")

            # Sample next tokens
            next_tokens = sampler(last_logits, key=key)
            print(f"Next tokens: {next_tokens.squeeze(-1).tolist()}")

            # Check for EOS
            # finished = finished | (next_tokens.squeeze(-1) == tokenizer.eos_token_id)
            # print(f"Finished: {finished.sum().item()}/{batch_size} sequences")

            # Append to tokens list
            tokens_list.append(next_tokens)

            # Break if all sequences are finished
            # if finished.all():
            #     print("\nAll sequences finished early!")
            #     break

    # ============================================================================
    # DECODE OUTPUT
    # ============================================================================
    print("\n=== DECODING OUTPUT ===")

    # Concatenate all tokens (prefill + decode)
    all_output_tokens = torch.cat(tokens_list, dim=1)
    print(f"All output tokens shape: {all_output_tokens.shape}")

    # Decode each sequence
    for i, q in enumerate(query):
        print(f"\n{'='*60}")
        print(f"User: {q}")
        print(f"{'='*60}")

        seq_tokens = all_output_tokens[i].cpu().numpy()

        # Find EOS token and truncate
        eos_indices = np.where(seq_tokens == tokenizer.eos_token_id)[0]
        if eos_indices.size > 0:
            seq_tokens = seq_tokens[:eos_indices[0]]

        # Decode
        decoded = tokenizer.decode(seq_tokens, skip_special_tokens=True)
        print(f"Answer: {decoded}")
        print(f"{'='*60}")

    return all_output_tokens


if __name__ == "__main__":
    run_model()