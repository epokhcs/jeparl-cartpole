"""Configuration management utilities."""

import yaml
from typing import Any, Dict
from pathlib import Path


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries.

    Args:
        base: Base dictionary
        override: Dictionary with override values

    Returns:
        Merged dictionary
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dicts
            result[key] = deep_merge(result[key], value)
        else:
            # Override value
            result[key] = value

    return result


class Config:
    """Configuration container with dot notation access."""

    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize configuration from dictionary.

        Args:
            config_dict: Configuration dictionary
        """
        self._config = config_dict
        self._make_recursive(config_dict)

    def _make_recursive(self, d: Dict[str, Any]) -> None:
        """Recursively convert nested dicts to Config objects."""
        for key, value in d.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def __getitem__(self, key: str) -> Any:
        """Dictionary-style access."""
        return self._config[key]

    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        return key in self._config

    def get(self, key: str, default: Any = None) -> Any:
        """Get value with default."""
        return self._config.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Convert back to dictionary."""
        return self._config


def load_config(config_path: str) -> Config:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Config object with dot notation access
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Handle defaults (simplified, not full Hydra)
    if 'defaults' in config_dict:
        base_configs = config_dict.pop('defaults')
        merged_config = {}

        # Load base configs
        for base_config in base_configs:
            if isinstance(base_config, str):
                base_path = config_path.parent / base_config
                # Add .yaml extension if not present
                if not base_path.suffix:
                    base_path = base_path.with_suffix('.yaml')
                with open(base_path, 'r') as f:
                    base_dict = yaml.safe_load(f)
                    if base_dict:
                        merged_config = deep_merge(merged_config, base_dict)

        # Override with current config (deep merge)
        merged_config = deep_merge(merged_config, config_dict)
        config_dict = merged_config

    return Config(config_dict)


def save_config(config: Config, output_path: str) -> None:
    """Save configuration to YAML file.

    Args:
        config: Config object to save
        output_path: Path to output YAML file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False)
