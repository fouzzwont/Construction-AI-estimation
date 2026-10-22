"""Central configuration loader.

Resolves the project root, loads ``config.yaml`` and exposes typed accessors
plus the standard folder layout.  The loader is deliberately forgiving: it
walks upwards from the module location looking for ``config.yaml`` so the
project can be imported from anywhere (CLI, Flask app, notebooks).
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_CONFIG_FILENAME = "config.yaml"


def find_project_root(start: Path | None = None) -> Path:
    """Return the directory that contains ``config.yaml``."""
    here = (start or Path(__file__).resolve().parent).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / _CONFIG_FILENAME).is_file():
            return candidate
    # Fall back to the package parent (src/ -> project root)
    return Path(__file__).resolve().parent.parent


class Config:
    """Thin wrapper around the parsed YAML configuration."""

    def __init__(self, data: dict[str, Any], root: Path, config_path: Path):
        self.data = data
        self.root = root
        self.config_path = config_path

    # ------------------------------------------------------------------ #
    # access helpers
    # ------------------------------------------------------------------ #
    def get(self, dotted: str, default: Any = None) -> Any:
        """Fetch ``a.b.c`` from the config, returning ``default`` if missing."""
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def section(self, name: str) -> dict[str, Any]:
        value = self.data.get(name)
        return copy.deepcopy(value) if isinstance(value, dict) else {}

    def path(self, dotted: str, default: str | None = None) -> Path:
        """Resolve a configured path against the project root."""
        raw = self.get(dotted, default)
        if raw is None:
            raise KeyError(f"No path configured under '{dotted}'")
        p = Path(raw)
        return p if p.is_absolute() else (self.root / p)

    # ------------------------------------------------------------------ #
    # standard layout
    # ------------------------------------------------------------------ #
    def ensure_dirs(self) -> None:
        for key in (
            "paths.raw_pdfs",
            "paths.processed_images",
            "paths.dataset_root",
            "paths.models",
            "paths.output",
            "paths.estimates",
            "paths.reports",
            "paths.logs",
            "paths.uploads",
        ):
            self.path(key).mkdir(parents=True, exist_ok=True)
        for split in ("images", "labels"):
            for part in ("train", "val", "test"):
                (self.path("paths.dataset_root") / split / part).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # convenience accessors
    # ------------------------------------------------------------------ #
    @property
    def labels(self) -> list[str]:
        return list(self.get("annotation.labels", []))

    @property
    def device(self) -> str:
        """Resolve ``auto`` into the best available torch device."""
        raw = str(self.get("inference.device", "auto"))
        if raw != "auto":
            return raw
        try:
            import torch

            return "cuda:0" if torch.cuda.is_available() else "cpu"
        except Exception:  # pragma: no cover - torch always present in practice
            return "cpu"

    def market_multiplier(self) -> float:
        market = self.get("rates.market", "india_urban")
        table = self.get("rates.market_multipliers", {}) or {}
        return float(table.get(market, 1.0))


def _default_config_path() -> Path:
    return find_project_root() / _CONFIG_FILENAME


def load_config(config_path: str | Path | None = None) -> Config:
    """Load the project configuration, falling back to built-in defaults."""
    if config_path is None:
        path = _default_config_path()
    else:
        path = Path(config_path)
        if not path.is_absolute():
            path = find_project_root() / path
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found: {path}")

    if path.is_file():
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        root = path.parent
    else:  # pragma: no cover - only hit when no config.yaml exists anywhere
        logger.warning("No config.yaml found, using defaults")
        data, root = {}, find_project_root()

    return Config(data, root, path)


_CONFIG: Config | None = None


def get_config(reload: bool = False) -> Config:
    """Return the process-wide cached configuration."""
    global _CONFIG
    if _CONFIG is None or reload:
        _CONFIG = load_config()
    return _CONFIG
