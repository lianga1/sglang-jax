from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


@dataclass
class TargetSpec:
    """Specification for a callable to profile."""

    name: str
    input_builder: str | None = None
    description: str | None = None


@dataclass
class SweepSpec:
    """Parameter sweep across batch/seq dimensions."""

    batch_sizes: Sequence[int]
    seq_lens: Sequence[int]
    generate_steps: int = 1
    jit: bool = True
    warmup: int = 1
    runs: int = 3
    capture_jaxpr: bool = True


@dataclass
class ExecutionCase:
    batch_size: int
    seq_len: int
    generate_steps: int

    def to_dict(self) -> dict[str, int]:
        return {
            "batch_size": self.batch_size,
            "seq_len": self.seq_len,
            "generate_steps": self.generate_steps,
        }


@dataclass
class ExecutionResult:
    target: str
    case: ExecutionCase
    wall_time_s: float
    compile_time_s: float | None
    input_shapes: Any
    output_shapes: Any
    x_value: float | None = None
    y_value: float | None = None
    jaxpr: str | None = None
    notes: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        row = {
            "target": self.target,
            "batch_size": self.case.batch_size,
            "seq_len": self.case.seq_len,
            "generate_steps": self.case.generate_steps,
            "wall_time_s": self.wall_time_s,
            "compile_time_s": self.compile_time_s,
            "x_value": self.x_value,
            "y_value": self.y_value,
        }
        row.update({f"note_{k}": v for k, v in self.notes.items()})
        return row


@dataclass
class HarnessConfig:
    targets: Sequence[TargetSpec]
    sweep: SweepSpec
    output_dir: Path
    pad_id: int = 0
    x_func: Callable[[ExecutionResult], float] | None = None
    y_func: Callable[[ExecutionResult], float] | None = None

