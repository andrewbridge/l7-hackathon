"""Loaders that convert each raw dataset into the symptompred schema.

Each module exposes `load(path=None, **options) -> pd.DataFrame`. The default paths assume the
project layout (data in the repository root).
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from . import brazil, carbon_health, israel  # noqa: E402

__all__ = ["israel", "carbon_health", "brazil", "PROJECT_ROOT"]
