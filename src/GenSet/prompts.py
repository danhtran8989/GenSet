"""
Language-specific system and user prompts for dataset generation.
Supports English and Vietnamese with easy extensibility.

Prompt templates are now stored in configs/promps-configs.yml
so they can be edited without touching Python code.
"""

import yaml
from pathlib import Path
from typing import List, Dict

# ====================== LOAD CONFIG ======================
def _load_config() -> Dict:
    """Load prompts from YAML config (relative to project root)."""
    # src/GenSet/prompts.py → parents[2] = project root
    config_path = Path(__file__).resolve().parents[2] / "configs" / "promps-configs.yml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"❌ Config file not found: {config_path}\n"
            "Please create configs/promps-configs.yml with the prompt templates."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


CONFIG = _load_config()

# ====================== PROMPTS FROM YAML ======================
SYSTEM_PROMPTS: Dict[str, str] = CONFIG.get("system_prompts", {})
USER_PROMPTS: Dict[str, Dict] = CONFIG.get("user_prompts", {})
DOMAIN_TRANSLATIONS: Dict[str, Dict[str, str]] = CONFIG.get("domain_translations", {})

LANGUAGE_CHOICES: List[str] = CONFIG.get("language_choices", ["english", "vietnamese"])
LANGUAGE_DISPLAY: Dict[str, str] = CONFIG.get("language_display", {})


# ====================== PUBLIC API ======================
def get_system_prompt(language: str = "english") -> str:
    """Return the system prompt for the given language."""
    language = language.lower()
    return SYSTEM_PROMPTS.get(language, SYSTEM_PROMPTS.get("english", ""))


def get_user_prompt(
    language: str = "english",
    labels: List[str] = None,
    domain: str = "general customer reviews",
    min_sentences: int = 1,
    max_sentences: int = 3,
) -> str:
    """Return the formatted user prompt."""
    if labels is None or not labels:
        labels = ["positive", "negative"]

    language = language.lower()
    template = USER_PROMPTS.get(language, USER_PROMPTS.get("english", {})).get("template", "")

    # Get translated domain if available
    domain_trans = DOMAIN_TRANSLATIONS.get(language, {}).get(domain, domain)

    labels_str = "/".join(labels)
    example_label = labels[0]  # Use first label as example in prompt

    return template.format(
        labels=labels_str,
        domain=domain_trans,
        min_sentences=min_sentences,
        max_sentences=max_sentences,
        example_label=example_label
    )


# For backward compatibility and CLI
__all__ = [
    "get_system_prompt",
    "get_user_prompt",
    "LANGUAGE_CHOICES",
    "LANGUAGE_DISPLAY",
    "DOMAIN_TRANSLATIONS",
    "SYSTEM_PROMPTS",
    "USER_PROMPTS",
]