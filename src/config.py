"""Central configuration loader.

Resolves the project root, loads ``config.yaml`` and exposes typed accessors
plus the standard folder layout.  The loader is deliberately forgiving: it
walks upwards from the module location looking for ``config.yaml`` so the
project can be imported from anywhere (CLI, Flask app, notebooks).
"""

from __future__ import annotations

import copy
import logging
import os
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_CONFIG_FILENAME = "config.yaml"

# Environment overrides, so a deployment can be tuned without editing
# config.yaml (the free Render plan has 512 MB RAM; a full-resolution A1 sheet
# does not fit). The mapping is explicit because config keys contain
# underscores themselves, which makes a generic prefix rule ambiguous.
_ENV_OVERRIDES: dict[str, str] = {
    "CAE_DPI": "pdf_processing.dpi",
    "CAE_MAX_RENDER_DIMENSION": "pdf_processing.max_render_dimension",
    "CAE_MAX_PAGES": "pdf_processing.max_pages",
    "CAE_CV_MAX_DIMENSION": "pdf_processing.cv_max_dimension",
    "CAE_CV_MODE": "pdf_processing.cv_mode",
    "CAE_QUALITY": "pdf_processing.quality",
    "CAE_UPLOAD_LIMIT_MB": "deployment.upload_limit_mb",
    "CAE_MAX_CONCURRENT_JOBS": "deployment.max_concurrent_jobs",
}


def _coerce(raw: str) -> Any:
    text = raw.strip()
    low = text.lower()
    if low in {"true", "yes", "on"}:
        return True
    if low in {"false", "no", "off"}:
        return False
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            continue
    return text


def apply_env_overrides(data: dict[str, Any]) -> list[str]:
    """Apply ``CAE_*`` environment variables to a parsed config, in place."""
    applied: list[str] = []
    for env_key, dotted in _ENV_OVERRIDES.items():
        raw = os.environ.get(env_key)
        if raw is None or not raw.strip():
            continue
        node = data
        parts = dotted.split(".")
        for part in parts[:-1]:
            child = node.get(part)
            if not isinstance(child, dict):
                child = {}
                node[part] = child
            node = child
        node[parts[-1]] = _coerce(raw)
        applied.append(f"{env_key}={raw} -> {dotted}")
    return applied


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

    applied = apply_env_overrides(data)
    if applied:
        logger.info("Config overridden by environment: %s", ", ".join(applied))

    return Config(data, root, path)


_CONFIG: Config | None = None


def get_config(reload: bool = False) -> Config:
    """Return the process-wide cached configuration."""
    global _CONFIG
    if _CONFIG is None or reload:
        _CONFIG = load_config()
    return _CONFIG
