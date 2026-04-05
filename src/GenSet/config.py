from dotenv import load_dotenv
import os
from typing import List, Optional

load_dotenv()


def get_api_keys(prefix: str) -> List[str]:
    keys = []
    index = 1
    while True:
        key = os.getenv(f"{prefix}_{index}")
        if key is None:
            break
        if key.strip():
            keys.append(key.strip())
        index += 1
    return keys


class Config:
    MISTRAL_API_KEYS: List[str] = get_api_keys("MISTRAL_API_KEY")
    OLLAMA_API_KEYS: List[str] = get_api_keys("OLLAMA_API_KEY")

    MISTRAL_BASE_URL = "https://api.mistral.ai/v1"
    OLLAMA_BASE_URL = "https://ollama.com"

    DEFAULT_MISTRAL_MODEL = "mistral-small-latest"
    DEFAULT_OLLAMA_MODEL = "gemma3:4b-cloud"

    DEFAULT_OUTPUT_DIR = "output"
    DEFAULT_OUTPUT_FILE = "dataset.tsv"

    @classmethod
    def normalize_output_path(cls, output_file: Optional[str] = None) -> str:
        output_file = output_file or cls.DEFAULT_OUTPUT_FILE
        output_file = os.path.expanduser(output_file)
        if not os.path.dirname(output_file):
            output_file = os.path.join(cls.DEFAULT_OUTPUT_DIR, output_file)
        return os.path.abspath(output_file)

    @classmethod
    def ensure_output_dir(cls, output_path: str) -> None:
        directory = os.path.dirname(output_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

    @classmethod
    def print_config(cls) -> None:
        print("=== Dataset Generator (CSV + Structured Prompts) ===")
        print(f"Mistral API Keys : {len(cls.MISTRAL_API_KEYS)}")
        print(f"Ollama API Keys  : {len(cls.OLLAMA_API_KEYS)}")
        print(f"Default Mistral  : {cls.DEFAULT_MISTRAL_MODEL}")
        print(f"Default Ollama   : {cls.DEFAULT_OLLAMA_MODEL}")
        print(f"Default output   : {cls.normalize_output_path(cls.DEFAULT_OUTPUT_FILE)}")
