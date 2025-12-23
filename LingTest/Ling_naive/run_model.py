

import math
import time

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from huggingface_hub import snapshot_download
from jax import P
from jax.sharding import AxisType
from transformers import AutoTokenizer

import modeling as modeling
import params2 as params
from sampler import GreedySampler, Sampler
from jax import NamedSharding

def tokenize(tokenizer, input: list[str], shd: P | None = None):
    pad_idx = tokenizer.pad_token_id
    lines = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": l}], tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        for l in input
    ]
    lines = [tokenizer.encode(line) for line in lines]
    max_len = max(len(line) for line in lines)  # Right-align, left-padding to the max token length.
    return jnp.array([np.pad(l, (max_len - len(l), 0), constant_values=pad_idx) for l in lines], out_sharding=shd)


def run_model():
    # For sharding, you can use one of the following:
    # model_ckpt_path = snapshot_download("Qwen/Qwen3-0.6B")
    model_ckpt_path = 'inclusionAI/Ling-mini-2.0'
    config = modeling.ModelConfig.ling_minimal(use_sharding=True)
    mesh, batch_shd = None, None
    # mesh = jax.make_mesh((2, 2), ("fsdp", "tp"), axis_types=(AxisType.Explicit, AxisType.Explicit))
    # Enable sharding below if you have mtuliple devices.
    # model_ckpt_path = snapshot_download("Qwen/Qwen3-4B")
    # config = modeling.ModelConfig.qwen3_4b(use_sharding=True)
    mesh = jax.make_mesh((2,2), ("fsdp", "tp"), axis_types=(AxisType.Explicit, AxisType.Explicit))
    batch_shd = P("fsdp", None)
    jax.set_mesh(mesh)
    

    query = [
        "Why is the sky blue instead of any other color like purple?",
        "Who am I?",
    ]

    tokenizer = AutoTokenizer.from_pretrained(model_ckpt_path)
    tokens = tokenize(tokenizer, query, batch_shd)
    batch_size, token_len = tokens.shape
    input_sharding = NamedSharding(mesh, P('fsdp', None))
    # input_sharding = None
    # 执行搬运
    tokens = jax.device_put(tokens, input_sharding)
    generate_steps = 100
    model = params.create_model_from_safe_tensors(model_ckpt_path, config, mesh)
    cache = model.init_cache(config, batch_size, token_len, generate_steps)

    key = jax.random.key(0)
    sampler = Sampler(temperature=1.0, top_p=0.8, top_k=10)
    jit_sampler = jax.jit(sampler)

    # prefill
# with jax.disable_jit():
    logits, cache = modeling.forward(model, cache, tokens, tokenizer.pad_token_id)
    next_tokens = jit_sampler(logits, key=key)

    # decode
    tokens_list = [next_tokens]
    finished = jnp.zeros((batch_size,), dtype=jnp.bool_)
    for i in range(generate_steps):
        logits, cache = modeling.forward(model, cache, next_tokens, tokenizer.pad_token_id)
        # print("Step:", i)
        # print("Logits:", logits)
        # print(f"Step {i}: cur_ind = {cache[0].cur_ind.value}") # 检查是否在增加
        # print(f"Logits stats: Min={logits.min()}, Max={logits.max()}, NaN?={jnp.any(jnp.isnan(logits))}")
        # print("Cache keys shape:", )
        
        next_tokens = jit_sampler(logits, key=key)

        # print("Next tokens:", next_tokens)
        finished = finished | (next_tokens.squeeze(-1) == tokenizer.eos_token_id)
        tokens_list.append(next_tokens)
        if finished.all():
            break

    all_output_tokens = jax.device_get(jnp.concatenate(tokens_list, axis=-1))
    for i, q in enumerate(query):
        print(f"User:\n {q}")
        seq_tokens = all_output_tokens[i]
        eos_idx = np.where(seq_tokens == tokenizer.eos_token_id)[0]
        if eos_idx.size > 0:
            seq_tokens = seq_tokens[: eos_idx[0]]
        decoded = tokenizer.decode(seq_tokens, skip_special_tokens=True)
        print(f"Answer:\n {decoded}\n\n")


if __name__ == "__main__":
    run_model()


__all__ = ["run_model"]
