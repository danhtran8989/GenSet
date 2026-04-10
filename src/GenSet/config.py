# configs/config.py
import os
import yaml
from pathlib import Path
from typing import Dict, List, Any
from dotenv import load_dotenv

load_dotenv()

CONFIG_PATH = Path("configs/configs.yml")


class Config:
    _config: Dict[str, Any] = None

    @classmethod
    def _load(cls):
        """Load YAML config once"""
        if cls._config is None:
            if not CONFIG_PATH.exists():
                raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")

            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cls._config = yaml.safe_load(f)

    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        cls._load()
        keys = key.split(".")
        value = cls._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value

    # ====================== API Keys (from .env) ======================
    @staticmethod
    def get_api_keys(prefix: str) -> List[str]:
        """Load MISTRAL_API_KEY_1, MISTRAL_API_KEY_2, ... from .env"""
        keys = []
        i = 1
        while True:
            key = os.getenv(f"{prefix}_{i}")
            if key is None:
                break
            if key.strip():
                keys.append(key.strip())
            i += 1
        return keys

    # ====================== Convenience Getters ======================
    @classmethod
    def get_base_url(cls, platform: str) -> str:
        platform = platform.lower()
        base_key = cls.get(f"platforms.{platform}.base_url_key")
        if base_key:
            return cls.get(f"base_urls.{base_key}")
        return None

    @classmethod
    def get_default_model(cls, platform: str) -> str:
        platform = platform.lower()
        model_key = cls.get(f"platforms.{platform}.default_model_key")
        if model_key:
            return cls.get(f"default_models.{model_key}")
        return None

    @classmethod
    def get_output_dir(cls) -> str:
        return cls.get("output.default_dir", "output")

    @classmethod
    def get_default_output_file(cls) -> str:
        return cls.get("output.default_file", "dataset.tsv")

    @classmethod
    def get_generation_default(cls, key: str, default: Any = None) -> Any:
        return cls.get(f"generation_defaults.{key}", default)

    @classmethod
    def print_config(cls):
        cls._load()
        print("=== GenSet Configuration (YAML + .env) ===")
        print(yaml.dump(cls._config, default_flow_style=False, sort_keys=False))
        print("\nAPI Keys loaded from .env:")
        for p in ["mistral", "ollama", "openai", "gemini"]:
            keys = cls.get_api_keys(p.upper() + "_API_KEY")
            print(f"  {p}: {len(keys)} key(s)")


# Backward compatibility (optional)
def get_mistral_api_keys():
    return Config.get_api_keys("MISTRAL_API_KEY")


if __name__ == "__main__":
    Config.print_config()