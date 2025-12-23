from __future__ import annotations

import importlib
from typing import Any


def resolve_dotted(path: str) -> Any:
    """Import and return an attribute given a dotted path.

    Example: ``profiling_harness.presets.build_forward_inputs``
    """

    if ":" in path:
        module_path, attr = path.split(":", 1)
    else:
        parts = path.rsplit(".", 1)
        if len(parts) != 2:
            raise ValueError(f"Dotted path must include module and attribute: {path}")
        module_path, attr = parts
    module = importlib.import_module(module_path)
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise ImportError(f"Cannot find attribute {attr} in module {module_path}") from exc
