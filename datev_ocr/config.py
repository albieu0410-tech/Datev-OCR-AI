from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import json

from .core import CFG_FILE, DEFAULTS


ConfigDict = dict[str, Any]


def _normalize_config(overrides: Mapping[str, Any] | None = None) -> ConfigDict:
    data: ConfigDict = dict(DEFAULTS)
    if overrides:
        for key, value in overrides.items():
            data[key] = value
    return data


def config_path(path: str | Path | None = None) -> Path:
    if path is None:
        return Path(CFG_FILE)
    return Path(path)


def load_config(path: str | Path | None = None) -> ConfigDict:
    cfg_file = config_path(path)
    if not cfg_file.exists():
        return _normalize_config()
    try:
        with cfg_file.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        payload = {}
    return _normalize_config(payload)


def save_config(config: Mapping[str, Any], path: str | Path | None = None) -> Path:
    cfg_file = config_path(path)
    cfg_file.parent.mkdir(parents=True, exist_ok=True)
    cfg = _normalize_config(config)
    with cfg_file.open("w", encoding="utf-8") as handle:
        json.dump(cfg, handle, indent=2, ensure_ascii=False)
    return cfg_file


__all__ = ["load_config", "save_config", "config_path"]
