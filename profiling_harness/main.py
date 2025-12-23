from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

from .presets import make_context
from .resolve import resolve_dotted
from .runner import run_targets
from .types import HarnessConfig, SweepSpec, TargetSpec


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Profile Ling_naive forward/attention and build roofline inputs.")
    p.add_argument("--targets", nargs="+", required=True, help="Dotted paths of callables to profile.")
    p.add_argument("--builder", help="Optional dotted path input builder to use for all targets.")
    p.add_argument("--batch", nargs="+", type=int, default=[1], help="Batch sizes to sweep.")
    p.add_argument("--seq", nargs="+", type=int, default=[64], help="Sequence lengths to sweep.")
    p.add_argument("--generate-steps", type=int, default=1, help="Generate steps passed to cache init.")
    p.add_argument("--warmup", type=int, default=1, help="Warmup runs (after compile).")
    p.add_argument("--runs", type=int, default=3, help="Timed runs per case.")
    p.add_argument("--no-jit", action="store_true", help="Disable jax.jit wrapping of targets.")
    p.add_argument("--no-capture-jaxpr", action="store_true", help="Skip jaxpr capture.")
    p.add_argument("--pad-id", type=int, default=0, help="Pad token id for input builders.")
    p.add_argument("--use-sharding", action="store_true", help="Use ModelConfig default sharding and create a mesh if provided.")
    p.add_argument("--mesh", nargs=2, type=int, help="Optional mesh shape rows cols; axis names are (fsdp,tp).")
    p.add_argument("--x-func", help="Dotted path to function that maps ExecutionResult -> x value.")
    p.add_argument("--y-func", help="Dotted path to function that maps ExecutionResult -> y value.")
    p.add_argument("--output-dir", default="profiling_runs", help="Directory to write CSV and jaxpr dumps.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    builder_override = args.builder
    targets: List[TargetSpec] = []
    for name in args.targets:
        targets.append(TargetSpec(name=name, input_builder=builder_override))

    sweep = SweepSpec(
        batch_sizes=args.batch,
        seq_lens=args.seq,
        generate_steps=args.generate_steps,
        jit=not args.no_jit,
        warmup=args.warmup,
        runs=args.runs,
        capture_jaxpr=not args.no_capture_jaxpr,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x_func = resolve_dotted(args.x_func) if args.x_func else None
    y_func = resolve_dotted(args.y_func) if args.y_func else None

    mesh_shape = tuple(args.mesh) if args.mesh else None
    ctx = make_context(use_sharding=args.use_sharding, pad_id=args.pad_id, mesh_shape=mesh_shape)

    cfg = HarnessConfig(targets=targets, sweep=sweep, output_dir=output_dir, pad_id=args.pad_id, x_func=x_func, y_func=y_func)
    results = run_targets(cfg, ctx)

    csv_path = output_dir / "results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].to_row().keys()) if results else [])
        writer.writeheader()
        for r in results:
            writer.writerow(r.to_row())
    # Dump jaxpr for first case of each target
    dumped = set()
    for r in results:
        if r.target in dumped:
            continue
        dumped.add(r.target)
        if r.jaxpr:
            safe_name = r.target.replace(".", "_")
            (output_dir / f"jaxpr_{safe_name}.txt").write_text(r.jaxpr)

    print(f"Wrote {len(results)} rows to {csv_path}")


if __name__ == "__main__":
    main()
