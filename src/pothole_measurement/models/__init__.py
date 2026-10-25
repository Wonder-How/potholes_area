"""ACSH architecture, adapted from the authors' segmentation refinement code.

Import lazily so geometry-only workflows do not need torch or Ultralytics.
"""

def register_legacy_modules():
    """Resolve module names embedded in original authors' trusted checkpoints."""
    import sys
    from . import acmix_modules, base_architecture, refine_architecture
    for module in (acmix_modules, base_architecture, refine_architecture):
        name = module.__name__.rsplit(".", 1)[-1]
        existing = sys.modules.get(name)
        if existing is not None and existing is not module:
            raise RuntimeError(f"Legacy checkpoint module {name!r} is already loaded from another package")
        sys.modules[name] = module


def load_model(weights):
    """Load a local trusted checkpoint, with the original ACSH classes registered."""
    from pathlib import Path
    path = Path(weights).expanduser().resolve(strict=True)
    register_legacy_modules()
    from ultralytics import YOLO
    return YOLO(str(path))
