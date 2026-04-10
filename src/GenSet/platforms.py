from typing import Dict, Any
from .config import Config


def get_platform_config(platform: str):
    """Return base_url and next API key for a platform"""
    platform = platform.lower()
    base_url = Config.get_base_url(platform)
    api_keys = Config.get_api_keys(f"{platform.upper()}_API_KEY")
    api_key = api_keys[0] if api_keys else None

    return base_url, api_key


def build_request(platform: str, model: str, system_prompt: str, user_prompt: str, temperature: float = 0.7):
    """Build URL, headers, and payload for each platform"""
    platform = platform.lower()
    base_url, api_key = get_platform_config(platform)

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    if platform == "ollama":
        url = f"{base_url.rstrip('/')}/api/chat"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "stream": False,
            "format": "json",
        }
    elif platform in ("mistral", "openai"):
        url = f"{base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "max_tokens": 1200,
        }
    elif platform == "gemini":
        url = f"{base_url.rstrip('/')}/models/{model}:generateContent?key={api_key}"
        headers.pop("Authorization", None)
        payload = {
            "contents": [{"parts": [{"text": f"{system_prompt}\n\n{user_prompt}"}]}],
            "generationConfig": {"temperature": temperature}
        }
    else:
        raise NotImplementedError(f"Platform {platform} not supported yet.")

    return url, headers, payload