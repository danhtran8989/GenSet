import csv
import json
import os
import re
import time
from typing import Dict, List, Literal, Optional

import requests
from rich.console import Console

from .config import Config
from .prompts import get_system_prompt, get_user_prompt

console = Console()


class DatasetGenerator:
    def __init__(self):
        self.mistral_keys = Config.MISTRAL_API_KEYS
        self.ollama_keys = Config.OLLAMA_API_KEYS
        self.current_mistral_idx = 0
        self.current_ollama_idx = 0
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "GenSet/1.0"})

    def _get_next_mistral_key(self) -> str:
        if not self.mistral_keys:
            raise ValueError("No Mistral API keys loaded in .env!")
        key = self.mistral_keys[self.current_mistral_idx]
        self.current_mistral_idx = (self.current_mistral_idx + 1) % len(self.mistral_keys)
        return key

    def _get_next_ollama_config(self):
        if not self.ollama_keys:
            return Config.OLLAMA_BASE_URL, None
        key = self.ollama_keys[self.current_ollama_idx]
        self.current_ollama_idx = (self.current_ollama_idx + 1) % len(self.ollama_keys)
        return Config.OLLAMA_BASE_URL, key

    def _clean_and_parse_json(self, text: str) -> Optional[Dict]:
        if not text:
            return None

        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{[\s\S]*?\}", text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        text_match = re.search(r'"text"\s*:\s*"([^"]+)"', text, re.IGNORECASE | re.DOTALL)
        label_match = re.search(r'"label"\s*:\s*"([^"]+)"', text, re.IGNORECASE | re.DOTALL)
        if text_match and label_match:
            return {
                "text": text_match.group(1).strip(),
                "label": label_match.group(1).strip(),
            }

        return None

    def generate_multilingual_example(
        self,
        platform: Literal["mistral", "ollama"] = "mistral",
        labels: List[str] = None,
        domain: str = "general customer reviews",
        min_sentences: int = 1,
        max_sentences: int = 3,
        model: str = None,
        temperature: float = 0.7,
    ) -> Dict:
        """Generate both English and Vietnamese text for the same label."""
        if labels is None:
            labels = ["positive", "negative"]

        english_example = self.generate_example(
            platform=platform,
            language="english",
            labels=labels,
            domain=domain,
            min_sentences=min_sentences,
            max_sentences=max_sentences,
            model=model,
            temperature=temperature,
        )

        vietnamese_example = self.generate_example(
            platform=platform,
            language="vietnamese",
            labels=labels,
            domain=domain,
            min_sentences=min_sentences,
            max_sentences=max_sentences,
            model=model,
            temperature=temperature,
        )

        return {
            "english_text": english_example.get("text", ""),
            "vietnamese_text": vietnamese_example.get("text", ""),
            "label": english_example.get("label", "unknown"),
        }

    def generate_example(
        self,
        platform: Literal["mistral", "ollama"] = "mistral",
        language: str = "english",
        labels: List[str] = None,
        domain: str = "general customer reviews",
        min_sentences: int = 1,
        max_sentences: int = 3,
        model: str = None,
        temperature: float = 0.7,
    ) -> Dict:
        if labels is None:
            labels = ["positive", "negative"]

        system_prompt = get_system_prompt(language)
        user_prompt = get_user_prompt(language, labels, domain, min_sentences, max_sentences)

        if platform == "mistral":
            model = model or Config.DEFAULT_MISTRAL_MODEL
            url = f"{Config.MISTRAL_BASE_URL}/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._get_next_mistral_key()}"
            }
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": temperature,
                "max_tokens": 1200,
            }
        else:
            model = model or Config.DEFAULT_OLLAMA_MODEL
            base_url, api_key = self._get_next_ollama_config()
            url = f"{base_url.rstrip('/')}/api/chat"
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

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

        response = self.session.post(url, json=payload, headers=headers, timeout=120)
        response.raise_for_status()

        try:
            data = response.json()
        except json.JSONDecodeError:
            raise ValueError("API response did not contain valid JSON")

        if platform == "mistral":
            raw_text = data["choices"][0]["message"]["content"]
        else:
            raw_text = data.get("message", {}).get("content", "")

        parsed = self._clean_and_parse_json(raw_text)
        if parsed and "text" in parsed and "label" in parsed:
            return {"text": str(parsed["text"]).strip(), "label": str(parsed["label"]).strip().lower()}

        console.print("\n[yellow]JSON parsing failed. Using raw response (sample may be noisy).[/yellow]")
        return {"text": raw_text.strip(), "label": "unknown"}

    def create_dataset(
        self,
        num_samples: int = 100,
        platform: Literal["mistral", "ollama"] = "mistral",
        language: str = "english",
        labels: List[str] = None,
        domain: str = "general customer reviews",
        min_sentences: int = 1,
        max_sentences: int = 3,
        model: str = None,
        temperature: float = 0.7,
        output_file: str = Config.DEFAULT_OUTPUT_FILE,
        delay: float = 0.8,
        multilingual: bool = False,
        balance_labels: bool = False,
    ) -> str:
        if labels is None:
            labels = ["positive", "negative"]

        output_file = Config.normalize_output_path(output_file)
        Config.ensure_output_dir(output_file)

        delimiter = "\t" if output_file.lower().endswith(".tsv") else ","
        file_exists = os.path.exists(output_file)

        # Prepare CSV headers based on multilingual flag
        if multilingual:
            headers = ["English_Text", "Vietnamese_Text", "Label"]
        else:
            headers = ["text", "label"]

        if not file_exists:
            with open(output_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f, delimiter=delimiter)
                writer.writerow(headers)

        console.print(f"[bold green]Starting generation of {num_samples} samples → {output_file}[/bold green]")
        console.print(f"Platform : {platform.upper()} | Language: {language} | Domain: {domain}")
        console.print(f"Labels   : {labels} | Temperature: {temperature}")
        if multilingual:
            console.print(f"[cyan]Mode     : MULTILINGUAL (English + Vietnamese)[/cyan]")
        if balance_labels:
            console.print(f"[cyan]Mode     : BALANCED LABELS[/cyan]")

        # Label distribution tracking for balancing
        label_counts = {label: 0 for label in labels}
        sample_queue = []

        # If balance_labels is enabled, distribute samples evenly across labels
        if balance_labels:
            samples_per_label = num_samples // len(labels)
            remainder = num_samples % len(labels)
            for label in labels:
                count = samples_per_label + (1 if remainder > 0 else 0)
                sample_queue.extend([label] * count)
                remainder -= 1
            # Shuffle to avoid predictable label order
            import random
            random.shuffle(sample_queue)
        else:
            sample_queue = [None] * num_samples

        success_count = 0
        for i, target_label in enumerate(sample_queue):
            try:
                if multilingual:
                    example = self.generate_multilingual_example(
                        platform=platform,
                        labels=labels,
                        domain=domain,
                        min_sentences=min_sentences,
                        max_sentences=max_sentences,
                        model=model,
                        temperature=temperature,
                    )
                    # Override label if balance_labels is enabled
                    if balance_labels and target_label:
                        example["label"] = target_label

                    with open(output_file, "a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f, delimiter=delimiter)
                        writer.writerow([example.get("english_text", ""), example.get("vietnamese_text", ""), example["label"]])
                else:
                    example = self.generate_example(
                        platform=platform,
                        language=language,
                        labels=labels,
                        domain=domain,
                        min_sentences=min_sentences,
                        max_sentences=max_sentences,
                        model=model,
                        temperature=temperature,
                    )
                    # Override label if balance_labels is enabled
                    if balance_labels and target_label:
                        example["label"] = target_label

                    with open(output_file, "a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f, delimiter=delimiter)
                        writer.writerow([example["text"], example["label"]])

                success_count += 1
                time.sleep(delay)

            except requests.exceptions.RequestException as e:
                console.print(f"[red]API Error (sample {i+1}): {e}[/red]")
                time.sleep(3)
            except Exception as e:
                console.print(f"[red]Unexpected error (sample {i+1}): {e}[/red]")
                time.sleep(2)
                continue

        console.print(f"\n[bold green]✅ Generation completed! {success_count}/{num_samples} samples saved to {output_file}[/bold green]")
        return output_file
