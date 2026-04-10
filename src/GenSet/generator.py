import time
import random
import requests
from typing import List, Literal, Optional, Dict
from rich.console import Console

from .config import Config
from .prompts import get_system_prompt, get_user_prompt
from .platforms import build_request
from .utils import clean_and_parse_json
from .dataset_writer import DatasetWriter

console = Console()


class DatasetGenerator:
    def __init__(self):
        self.key_indices: Dict[str, int] = {"mistral": 0, "ollama": 0, "openai": 0, "gemini": 0}
        self.session = requests.Session()  # Reuse TCP connections for faster repeated calls

    def _get_next_key(self, platform: str) -> Optional[str]:
        """Rotate through API keys for the given platform (prepared for future use)."""
        keys = getattr(Config, f"{platform.upper()}_API_KEYS", [])
        if not keys:
            return None
        idx = self.key_indices[platform]
        key = keys[idx]
        self.key_indices[platform] = (idx + 1) % len(keys)
        return key

    def _extract_content(self, platform: str, data: Dict) -> str:
        """Extract the generated text from the platform-specific response format."""
        if platform in ("mistral", "openai"):
            return data.get("choices", [{}])[0].get("message", {}).get("content", "")
        elif platform == "ollama":
            return data.get("message", {}).get("content", "")
        elif platform == "gemini":
            return (
                data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
            )
        return str(data)

    def _call_llm(
        self,
        platform: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
    ) -> str:
        """Make the API request and return the raw generated text."""
        # NOTE: Key rotation is prepared but not passed yet because build_request()
        # does not accept an 'api_key' argument. Update .platforms/build_request
        # if you want automatic key rotation to work.
        self._get_next_key(platform)  # still rotates the index (no-op for now)

        url, headers, payload = build_request(
            platform=platform,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
        )

        response = self.session.post(url, json=payload, headers=headers, timeout=120)
        response.raise_for_status()
        data = response.json()

        return self._extract_content(platform, data)

    def generate_example(
        self,
        platform: Literal["mistral", "ollama", "openai", "gemini"] = "mistral",
        language: str = "english",
        labels: Optional[List[str]] = None,
        domain: str = "general customer reviews",
        min_sentences: int = 1,
        max_sentences: int = 3,
        model: Optional[str] = None,
        temperature: float = 0.7,
        forced_label: Optional[str] = None,
    ) -> Dict:
        """Generate one labeled example (single language)."""
        if labels is None:
            labels = ["positive", "negative"]

        system_prompt = get_system_prompt(language)
        user_prompt = get_user_prompt(language, labels, domain, min_sentences, max_sentences)

        # Force the exact label at generation time (much more reliable than post-processing)
        if forced_label:
            user_prompt += (
                f"\n\nCRITICAL: The 'label' field in your JSON MUST be exactly '{forced_label}'. "
                f"Do not choose any other label."
            )

        model = model or Config.get_default_model(platform)

        console.print(f"[cyan]→ {platform.upper()} | Model: {model} | Language: {language}[/cyan]")

        raw_text = self._call_llm(
            platform=platform,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
        )

        parsed = clean_and_parse_json(raw_text)
        if parsed and "text" in parsed and "label" in parsed:
            return {
                "text": parsed["text"].strip(),
                "label": parsed["label"].strip().lower(),
            }

        # Fallback
        if forced_label:
            console.print(f"[yellow]Warning: JSON parse failed → forcing label '{forced_label}'[/yellow]")
            return {"text": raw_text.strip(), "label": forced_label}

        console.print("[yellow]Warning: JSON parsing failed, using raw text[/yellow]")
        return {"text": raw_text.strip(), "label": "unknown"}

    def generate_multilingual_example(
        self,
        forced_label: Optional[str] = None,
        **kwargs,
    ) -> Dict:
        """Generate paired English + Vietnamese examples with the same label."""
        eng = self.generate_example(language="english", forced_label=forced_label, **kwargs)
        vie = self.generate_example(language="vietnamese", forced_label=forced_label, **kwargs)

        return {
            "english_text": eng.get("text", ""),
            "vietnamese_text": vie.get("text", ""),
            "label": eng.get("label", "unknown"),
        }

    def create_dataset(
        self,
        num_samples: int = 100,
        platform: Literal["mistral", "ollama", "openai", "gemini"] = "mistral",
        language: str = "english",
        labels: Optional[List[str]] = None,
        domain: str = "general customer reviews",
        min_sentences: int = 1,
        max_sentences: int = 3,
        model: Optional[str] = None,
        temperature: float = 0.7,
        output_file: Optional[str] = None,
        delay: float = 0.8,
        multilingual: bool = False,
        balance_labels: bool = False,
    ) -> str:
        """Generate a full dataset with optional label balancing and multilingual support."""
        if labels is None:
            labels = ["positive", "negative"]

        writer = DatasetWriter(output_file, multilingual=multilingual)

        console.print(f"[bold green]Starting dataset generation: {num_samples} samples[/bold green]")
        lang_display = "multilingual (EN/VI)" if multilingual else language
        console.print(
            f"Platform: {platform.upper()} | Language: {lang_display} | "
            f"Domain: {domain[:20]}... | Balanced: {balance_labels}"
        )

        # Build a shuffled queue of target labels (guarantees exact count)
        if balance_labels:
            if not labels:
                console.print("[red]Error: balance_labels=True but no labels provided[/red]")
                return writer.output_file

            samples_per_label = num_samples // len(labels)
            remainder = num_samples % len(labels)
            sample_queue = []
            for i, label in enumerate(labels):
                count = samples_per_label + (1 if i < remainder else 0)
                sample_queue.extend([label] * count)
            random.shuffle(sample_queue)
            sample_queue = sample_queue[:num_samples]
        else:
            sample_queue = [None] * num_samples

        success = 0
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
                        forced_label=target_label,
                    )
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
                        forced_label=target_label,
                    )

                writer.write(example)
                success += 1
                time.sleep(delay)

            except Exception as e:
                console.print(f"[red]Error at sample {i + 1}: {e}[/red]")
                time.sleep(3)

        console.print(
            f"[bold green]✅ Done! {success}/{num_samples} samples saved to {writer.output_file}[/bold green]"
        )
        return writer.output_file