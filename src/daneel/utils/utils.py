"""Generic utility functions for Daneel."""

from __future__ import annotations

import copy
import logging
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Sequence, Union

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO") -> None:
    """Configures application-wide logging format and severity level.

    Args:
        log_level: Logging severity level (e.g., 'DEBUG', 'INFO', 'WARNING').
    """
    numeric_level = getattr(logging, str(log_level).upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=False,
    )


def deep_merge(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Recursively merges an override dictionary into a base configuration dictionary.

    Args:
        base: The base/default configuration dictionary.
        override: The overriding dictionary.

    Returns:
        A new deeply merged dictionary.
    """
    merged = copy.deepcopy(base)
    if not override:
        return merged

    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


# Default aliases mapping flat/synonym keys to canonical sections
DEFAULT_DATA_KEYS: Dict[str, str] = {
    "csv_path": "csv_path",
    "data_path": "csv_path",
    "dataset_path": "csv_path",
    "dataset": "csv_path",
    "n_bins": "n_bins",
    "test_size": "test_size",
    "balanced": "balanced",
    "samples_per_class": "samples_per_class",
    "data_random_state": "random_state",
}

DEFAULT_OUTPUT_KEYS: Dict[str, str] = {
    "report_path": "report_path",
    "output_path": "report_path",
    "output_file": "report_path",
    "report_file": "report_path",
    "model_path": "model_path",
    "model_save_path": "model_path",
}


def normalize_config(
    raw_config: Dict[str, Any],
    default_config: Dict[str, Any],
    model_keys: Optional[Dict[str, str]] = None,
    data_keys: Optional[Dict[str, str]] = None,
    output_keys: Optional[Dict[str, str]] = None,
    wrapper_keys: Sequence[str] = ("classifier", "rf", "random_forest", "cnn", "detection"),
) -> Dict[str, Any]:
    """Normalizes various YAML schemas into standard {data, model, output, logging} structure.

    Supports:
      1. Standard structure: {data: {...}, model: {...}, output: {...}}
      2. Nested under a wrapper key ('classifier', 'rf', 'cnn', 'detection', etc.)
      3. Flat parameters at root level.
    """
    config: Dict[str, Any] = {"data": {}, "model": {}, "output": {}, "logging": {}}

    # Check for wrapper sections
    target_dict = raw_config
    for key in wrapper_keys:
        if key in raw_config and isinstance(raw_config[key], dict):
            target_dict = raw_config[key]
            break

    # If target or raw has structured keys, merge them
    for section in ("data", "model", "output", "logging"):
        if section in target_dict and isinstance(target_dict[section], dict):
            config[section].update(target_dict[section])
        elif section in raw_config and isinstance(raw_config[section], dict):
            config[section].update(raw_config[section])

    d_keys = data_keys if data_keys is not None else DEFAULT_DATA_KEYS
    m_keys = model_keys if model_keys is not None else {}
    o_keys = output_keys if output_keys is not None else DEFAULT_OUTPUT_KEYS

    # Extract flat keys from target_dict or raw_config
    for source in (raw_config, target_dict):
        for k, v in source.items():
            if k in d_keys and d_keys[k] not in config["data"]:
                config["data"][d_keys[k]] = v
            elif k in m_keys and m_keys[k] not in config["model"]:
                config["model"][m_keys[k]] = v
            elif k in o_keys and o_keys[k] not in config["output"]:
                config["output"][o_keys[k]] = v

    # Clean up empty sections so deep_merge will use defaults
    clean_config = {k: v for k, v in config.items() if v}
    return deep_merge(default_config, clean_config)
