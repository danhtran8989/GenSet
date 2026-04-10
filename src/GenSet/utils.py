import json
import re
from typing import Dict, Optional

def clean_and_parse_json(text: str) -> Optional[Dict]:
    """Robust JSON extractor with fallback strategies"""
    if not text:
        return None
    text = text.strip()

    # 1. Direct JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Extract first JSON object
    match = re.search(r"\{[\s\S]*?\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # 3. Fallback regex for text + label
    text_match = re.search(r'"text"\s*:\s*"([^"]+)"', text, re.IGNORECASE | re.DOTALL)
    label_match = re.search(r'"label"\s*:\s*"([^"]+)"', text, re.IGNORECASE | re.DOTALL)

    if text_match and label_match:
        return {
            "text": text_match.group(1).strip(),
            "label": label_match.group(1).strip(),
        }

    return None