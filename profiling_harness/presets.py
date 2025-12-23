from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import jax
import jax.numpy as jnp
from flax import nnx

# Ensure the workspace root is on sys.path so we can import LingTest.Ling_naive.modeling
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))

from LingTest.Ling_naive import modeling  # noqa: E402


@dataclass
class HarnessContext:
    cfg: modeling.ModelConfig
    model: modeling.Ling2_mini
    rngs: nnx.Rngs
    pad_id: int
    mesh: Any | None = None


def make_context(*, use_sharding: bool = False, pad_id: int = 0, mesh_shape: tuple[int, int] | None = None) -> HarnessContext:
    """Build a default Ling-minimal model for profiling.

    Args:
        use_sharding: Whether to request the model's default sharding cfg.
        pad_id: Token id to treat as padding when building inputs.
        mesh_shape: Optional 2D mesh (rows, cols) to create and set; axis names are ("fsdp", "tp").
    """

    cfg = modeling.ModelConfig.ling_minimal(use_sharding=use_sharding)
    rngs = nnx.Rngs(0)

    mesh = None
    if mesh_shape is not None:
        mesh = jax.make_mesh(mesh_shape, ("fsdp", "tp"))
        jax.set_mesh(mesh)

    model = modeling.Ling2_mini(cfg, rngs=rngs)
    return HarnessContext(cfg=cfg, model=model, rngs=rngs, pad_id=pad_id, mesh=mesh)


def build_forward_inputs(ctx: HarnessContext, batch_size: int, seq_len: int, generate_steps: int) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Create inputs for modeling.forward with random tokens and a fresh cache."""

    key = jax.random.key(0)
    tokens = jax.random.randint(key, (batch_size, seq_len), minval=1, maxval=ctx.cfg.vocab_size, dtype=jnp.int32)
    cache = ctx.model.init_cache(ctx.cfg, batch_size, seq_len, generate_steps)
    args = (ctx.model, cache, tokens, ctx.pad_id)
    return args, {}


DEFAULT_INPUT_BUILDERS: dict[str, Callable[..., tuple[tuple[Any, ...], dict[str, Any]]]] = {
    "LingTest.Ling_naive.modeling.forward": build_forward_inputs,
    "modeling.forward": build_forward_inputs,
}
