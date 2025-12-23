# Profiling Harness for `Ling_naive`

This lightweight harness profiles a list of callables (e.g., `LingTest.Ling_naive.modeling.forward`) by:
- Running once to capture shapes and an optional `jaxpr`.
- Sweeping batch/sequence sizes, timing compiled execution, and logging results to CSV.
- Optionally computing custom `x`/`y` metrics (for roofline) via user-provided functions.

## Quick start
```bash
cd /home/gcpuser/sky_workdir/sglang-jax
python -m profiling_harness.main \
  --targets LingTest.Ling_naive.modeling.forward \
  --batch 1 2 \
  --seq 64 128 \
  --generate-steps 1 \
  --runs 3 \
  --warmup 1 \
  --output-dir profiling_runs
```
This uses random tokens, builds a random `Ling2_mini` model, runs `forward` under `jax.jit`, writes `results.csv`, and dumps one `jaxpr` per target.

## Custom `x`/`y` (roofline)
Pass functions that accept an `ExecutionResult` and return a float:
```bash
python -m profiling_harness.main \
  --targets LingTest.Ling_naive.modeling.forward \
  --x-func mymodule.compute_x \
  --y-func mymodule.compute_y
```
Your module must be importable on `PYTHONPATH`.

## Selecting input builders
`modeling.forward` is wired to the built-in builder that allocates random tokens and a fresh cache. For other callables, supply `--builder dotted.path.to.builder` where the builder signature is `(ctx, batch_size, seq_len, generate_steps) -> (args, kwargs)`.

## Sharding / mesh
Use `--use-sharding` to request the model's default sharding config. Optionally create a mesh with `--mesh ROWS COLS` (axis names are `fsdp`, `tp`).

## Output
- `results.csv` with timing and optional x/y values.
- `jaxpr_<target>.txt` for the first case per target (if capture enabled).

## Notes
- The harness initializes weights randomly; to profile real checkpoints, swap in your own builder that loads weights and tokens.
- All timings call `block_until_ready` to measure executed time, not enqueue time.
