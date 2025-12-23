from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp

from .presets import DEFAULT_INPUT_BUILDERS, HarnessContext
from .resolve import resolve_dotted
from .types import ExecutionCase, ExecutionResult, HarnessConfig, TargetSpec


def _shape_summary(obj: Any) -> Any:
    if hasattr(obj, "shape"):
        try:
            return tuple(int(x) for x in obj.shape)
        except Exception:
            return str(getattr(obj, "shape", ""))
    return type(obj).__name__


def _tree_shapes(x: Any) -> Any:
    return jax.tree_util.tree_map(_shape_summary, x, is_leaf=lambda v: isinstance(v, (int, float, str)))


def _select_builder(spec: TargetSpec) -> Callable[..., tuple[tuple[Any, ...], dict[str, Any]]]:
    if spec.input_builder:
        return resolve_dotted(spec.input_builder)
    if spec.name in DEFAULT_INPUT_BUILDERS:
        return DEFAULT_INPUT_BUILDERS[spec.name]
    raise ValueError(f"No input builder for target {spec.name}; provide --input-builder or add mapping.")


def _resolve_target(name: str) -> Callable[..., Any]:
    return resolve_dotted(name)


def _maybe_make_jaxpr(target: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any], enabled: bool) -> str | None:
    if not enabled:
        return None
    try:
        jaxpr = jax.make_jaxpr(target)(*args, **kwargs)
        return str(jaxpr)
    except Exception as exc:
        return f"<jaxpr capture failed: {exc}>"


def _execute_single(
    *,
    target_name: str,
    target: Callable[..., Any],
    builder: Callable[..., tuple[tuple[Any, ...], dict[str, Any]]],
    ctx: HarnessContext,
    case: ExecutionCase,
    sweep,
    x_func: Callable[[ExecutionResult], float] | None,
    y_func: Callable[[ExecutionResult], float] | None,
) -> ExecutionResult:
    args, kwargs = builder(ctx, case.batch_size, case.seq_len, case.generate_steps)

    callable_to_run: Callable[..., Any]
    if sweep.jit:
        callable_to_run = jax.jit(target)
    else:
        callable_to_run = target

    # Compile / warmup
    compile_time_s: float | None = None
    start = time.perf_counter()
    out = callable_to_run(*args, **kwargs)
    jax.block_until_ready(out)
    compile_time_s = time.perf_counter() - start

    # Additional warmup iterations (already compiled)
    for _ in range(max(sweep.warmup - 1, 0)):
        tmp = callable_to_run(*args, **kwargs)
        jax.block_until_ready(tmp)

    # Timed runs
    times = []
    last_out = out
    for _ in range(sweep.runs):
        t0 = time.perf_counter()
        last_out = callable_to_run(*args, **kwargs)
        jax.block_until_ready(last_out)
        times.append(time.perf_counter() - t0)

    wall_time_s = float(sum(times) / len(times)) if times else compile_time_s or 0.0

    result = ExecutionResult(
        target=target_name,
        case=case,
        wall_time_s=wall_time_s,
        compile_time_s=compile_time_s,
        input_shapes=_tree_shapes(args),
        output_shapes=_tree_shapes(last_out),
        jaxpr=_maybe_make_jaxpr(target, args, kwargs, sweep.capture_jaxpr),
    )

    if x_func:
        try:
            result.x_value = float(x_func(result))
        except Exception as exc:
            result.notes["x_error"] = str(exc)
    if y_func:
        try:
            result.y_value = float(y_func(result))
        except Exception as exc:
            result.notes["y_error"] = str(exc)

    return result


def run_targets(config: HarnessConfig, ctx: HarnessContext) -> list[ExecutionResult]:
    cases = [ExecutionCase(b, s, config.sweep.generate_steps) for b in config.sweep.batch_sizes for s in config.sweep.seq_lens]
    results: list[ExecutionResult] = []

    for spec in config.targets:
        target = _resolve_target(spec.name)
        builder = _select_builder(spec)
        for case in cases:
            results.append(
                _execute_single(
                    target_name=spec.name,
                    target=target,
                    builder=builder,
                    ctx=ctx,
                    case=case,
                    sweep=config.sweep,
                    x_func=config.x_func,
                    y_func=config.y_func,
                )
            )
    return results
