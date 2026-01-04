# app/utils/config_loader.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import yaml


# --------------------------------------------------
# Repo root resolution
# app/utils/config_loader.py → repo root is 2 levels up
# --------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_config_path(rel_or_abs_path: str) -> Path:
    """
    Resolve config paths in a robust, enterprise-safe way.

    Resolution order:
    1. Absolute path
    2. <repo_root>/<path>
    3. <repo_root>/app/<path>
    4. <repo_root>/app/config/<filename>  (for 'config/*.yaml')
    """
    p = Path(rel_or_abs_path)

    # 1️⃣ Absolute path
    if p.is_absolute():
        return p

    # 2️⃣ repo_root/<path>
    candidate = REPO_ROOT / p
    if candidate.exists():
        return candidate

    # 3️⃣ repo_root/app/<path>
    candidate = REPO_ROOT / "app" / p
    if candidate.exists():
        return candidate

    # 4️⃣ repo_root/app/config/<filename>
    # Allows callers to use: load_yaml_config("config/prompts.yaml")
    if p.parts and p.parts[0] == "config":
        candidate = REPO_ROOT / "app" / "config" / p.name
        if candidate.exists():
            return candidate

    # Fallthrough → explicit error later
    return candidate


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing YAML config: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML structure (expected dict): {path}")

    return data


def load_yaml_config(rel_or_abs_path: str) -> Dict[str, Any]:
    """
    Public API used across agents.

    Example:
        load_yaml_config("config/prompts.yaml")
        load_yaml_config("app/config/mcp_governance.yaml")
    """
    resolved_path = resolve_config_path(rel_or_abs_path)
    return _read_yaml(resolved_path)