"""
Module to fetch available models from Mistral and Ollama Cloud APIs.
"""

import requests
from typing import Dict, List, Optional
from .config import Config


class ModelsManager:
    """Manages fetching and caching available models from different platforms."""
    
    # Cache for models
    _mistral_models_cache: Optional[List[str]] = None
    _ollama_models_cache: Optional[List[str]] = None
    
    @staticmethod
    def get_mistral_models() -> List[str]:
        """
        Fetch available models from Mistral API.
        
        Returns:
            List of available Mistral model names.
            Falls back to default models if API call fails.
        """
        if ModelsManager._mistral_models_cache is not None:
            return ModelsManager._mistral_models_cache
        
        models = []
        
        # Try to fetch from Mistral API if API key is available
        if Config.MISTRAL_API_KEYS:
            try:
                api_key = Config.MISTRAL_API_KEYS[0]
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "User-Agent": "GenSet/1.0"
                }
                
                response = requests.get(
                    f"{Config.MISTRAL_BASE_URL}/models",
                    headers=headers,
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if "data" in data:
                        models = [model["id"] for model in data["data"]]
                        ModelsManager._mistral_models_cache = models
                        return models
            except Exception as e:
                print(f"Warning: Could not fetch Mistral models from API: {e}")
        
        # Fallback to default/known Mistral models
        default_models = [
            # "mistral-large-latest",
            # "mistral-medium-latest",
            "mistral-small-latest",
            # "mistral-tiny-latest",
        ]
        ModelsManager._mistral_models_cache = default_models
        return default_models
    
    @staticmethod
    def get_ollama_models() -> List[str]:
        """
        Fetch available models from Ollama Cloud API.
        
        Returns:
            List of available Ollama model names.
            Falls back to default models if API call fails.
        """
        if ModelsManager._ollama_models_cache is not None:
            return ModelsManager._ollama_models_cache
        
        models = []
        
        try:
            # Try to fetch from Ollama Cloud/registry
            response = requests.get(
                "https://registry.ollama.ai/api/tags",
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if "results" in data:
                    models = [model["name"] for model in data["results"]]
                    ModelsManager._ollama_models_cache = models
                    return models
        except Exception as e:
            print(f"Warning: Could not fetch Ollama models from registry: {e}")
        
        # Try alternative Ollama API endpoint
        try:
            response = requests.get(
                "https://api.ollama.ai/models",
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list):
                    models = [model.get("name") for model in data if "name" in model]
                    if models:
                        ModelsManager._ollama_models_cache = models
                        return models
        except Exception as e:
            print(f"Warning: Could not fetch Ollama models from alternative endpoint: {e}")
        
        # Fallback to popular/known Ollama models
        default_models = [
            # "gemma3:4b-cloud",
            "gemma3:12b-cloud",
            "gpt-oss:20b-cloud",
        ]
        ModelsManager._ollama_models_cache = default_models
        return default_models
    
    @staticmethod
    def get_all_models() -> Dict[str, List[str]]:
        """
        Fetch available models from all platforms.
        
        Returns:
            Dictionary with platform names as keys and list of models as values.
        """
        return {
            "mistral": ModelsManager.get_mistral_models(),
            "ollama": ModelsManager.get_ollama_models(),
        }
    
    @staticmethod
    def clear_cache() -> None:
        """Clear the cached models."""
        ModelsManager._mistral_models_cache = None
        ModelsManager._ollama_models_cache = None


# Convenience functions
def get_mistral_models() -> List[str]:
    """Get available Mistral models."""
    return ModelsManager.get_mistral_models()


def get_ollama_models() -> List[str]:
    """Get available Ollama models."""
    return ModelsManager.get_ollama_models()


def get_all_models() -> Dict[str, List[str]]:
    """Get available models from all platforms."""
    return ModelsManager.get_all_models()
