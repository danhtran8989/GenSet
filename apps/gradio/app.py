import sys
from pathlib import Path
import argparse
import os
from typing import List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import gradio as gr
from GenSet import Config, DatasetGenerator, LANGUAGE_CHOICES, get_all_models


def parse_labels(label_string: str) -> List[str]:
    labels = [label.strip() for label in label_string.split(",") if label.strip()]
    return labels or ["positive", "negative"]


def normalize_weights(weights: List[float]) -> List[float]:
    total = sum(weights) or 1.0
    return [w / total for w in weights]


def get_api_key_label(platform: str, idx: int) -> str:
    return f"{platform}_{idx + 1}"


# ====================== SAMPLE ======================
def create_sample(platforms, languages, domain, labels, mistral_models, ollama_models, temperature):
    if not platforms or not languages:
        return "", "", "⚠️ Select platform and language"
    
    models_dict = {}
    if mistral_models: models_dict["mistral"] = mistral_models
    if ollama_models:  models_dict["ollama"] = ollama_models

    platform = platforms[0]
    language = languages[0]
    model = models_dict.get(platform, [None])[0]

    try:
        generator = DatasetGenerator()
        ex = generator.generate_example(
            platform=platform, language=language, labels=parse_labels(labels),
            domain=domain, model=model, temperature=temperature
        )
        return ex.get("text", ""), ex.get("label", ""), f"✓ {platform}/{model} ({language})"
    except Exception as e:
        return "", "", f"✗ Error: {str(e)}"


# ====================== WORKER (now per API Key) ======================
def create_dataset_worker(
    platform: str,
    api_key_idx: int,
    languages: List[str],
    lang_weights: List[float],
    models: List[str],
    model_weights: List[float],
    domain: str,
    labels: str,
    num_samples: int,
    temperature: float,
    output_file: str,
    balance_labels: bool,
):
    temp_info = []
    try:
        generator = DatasetGenerator()
        norm_lang = normalize_weights(lang_weights)
        norm_model = normalize_weights(model_weights)

        # Each "worker" is now one API key → we give it proportional samples
        # For simplicity, we distribute total samples across all selected API keys
        # (The main function will calculate how many samples per API key)

        combos = []
        for l_idx, lang in enumerate(languages):
            for m_idx, model in enumerate(models):
                weight = norm_lang[l_idx] * norm_model[m_idx]
                combos.append((lang, model, weight))

        total_weight = sum(w for _, _, w in combos) or 1.0

        # This function will be called with pre-calculated samples_per_key
        # But for now we keep the logic here (samples_per_key will be passed later)

        # Note: We'll adjust samples in the main function

        return f"API {get_api_key_label(platform, api_key_idx)} ready", temp_info   # placeholder

    except Exception as e:
        return f"Error with {platform}_{api_key_idx+1}", temp_info


# Better approach: calculate samples per API key in main function
def create_dataset(
    platforms: List[str],
    languages: List[str],
    lang_weight: float,
    mistral_models: List[str],
    mistral_model_weight: float,
    ollama_models: List[str],
    ollama_model_weight: float,
    mistral_api_select: List[str],
    mistral_api_weight: float,
    ollama_api_select: List[str],
    ollama_api_weight: float,
    domain: str,
    labels: str,
    num_samples: int,
    temperature: float,
    output_file: str,
    balance_labels: bool,
):
    if not platforms:
        return "⚠️ Please select at least one platform (Mistral or Ollama)"
    if not languages:
        return "⚠️ Please select at least one language"
    
    # Prepare normalized weights
    lang_weights = [float(lang_weight)] * len(languages)
    norm_lang = normalize_weights(lang_weights)

    # Build platform config: (models, model_weight)
    platform_config = {}
    if "mistral" in platforms and mistral_models:
        platform_config["mistral"] = (mistral_models, float(mistral_model_weight))
    if "ollama" in platforms and ollama_models:
        platform_config["ollama"] = (ollama_models, float(ollama_model_weight))

    if not platform_config:
        return "⚠️ Please select at least one model for the chosen platform(s)"

    # Build API config: (selected_api_keys, api_weight)
    api_config = {}
    if "mistral" in platforms and mistral_api_select:
        api_config["mistral"] = (mistral_api_select, float(mistral_api_weight))
    if "ollama" in platforms and ollama_api_select:
        api_config["ollama"] = (ollama_api_select, float(ollama_api_weight))

    if not api_config:
        return "⚠️ Please select at least one API key for the chosen platform(s)"

    results = []
    all_temp_info = []
    total_generated = 0

    try:
        # Calculate total platform weight
        total_platform_weight = sum(
            platform_config[p][1] for p in platform_config
        ) or 1.0
        norm_platform = {p: w / total_platform_weight for p, (_, w) in platform_config.items()}

        # Process each platform
        for platform in platforms:
            if platform not in platform_config or platform not in api_config:
                continue

            models, model_weight = platform_config[platform]
            api_keys, api_weight = api_config[platform]

            # Normalize API key weights (equal weight if only one slider for now)
            norm_api = normalize_weights([api_weight] * len(api_keys))

            # Samples for this platform
            platform_samples = max(1, int(num_samples * norm_platform.get(platform, 0) + 0.5))

            results.append(f"→ {platform.upper()}: allocating {platform_samples} samples")

            # Distribute platform samples across API keys
            for api_idx, (api_key_label, api_key_weight) in enumerate(zip(api_keys, norm_api)):
                key_samples = max(1, int(platform_samples * api_key_weight + 0.5))

                # Distribute key samples across languages and models
                combo_list = []
                for l_idx, lang in enumerate(languages):
                    for m_idx, model in enumerate(models):
                        combo_weight = norm_lang[l_idx] * (model_weight / 100.0)  # normalize model weight
                        combo_samples = max(1, int(key_samples * combo_weight + 0.5))
                        if combo_samples > 0:
                            combo_list.append((lang, model, combo_samples, api_idx))

                # Generate for this API key
                generator = DatasetGenerator()
                for lang, model, combo_samples, api_idx in combo_list:
                    if combo_samples <= 0:
                        continue

                    safe_model = model.replace("/", "_").replace(":", "_").replace(".", "_")
                    temp_file = output_file.replace(
                        ".tsv", f"_{platform}_{lang}_{safe_model}_k{api_idx}.tsv"
                    )

                    result_path = generator.create_dataset(
                        num_samples=combo_samples,
                        platform=platform,
                        language=lang,
                        labels=parse_labels(labels),
                        domain=domain,
                        model=model,
                        temperature=temperature,
                        output_file=temp_file,
                        delay=0.8,
                        multilingual=False,
                        balance_labels=balance_labels,
                    )

                    all_temp_info.append((result_path, platform, model, api_idx))
                    total_generated += combo_samples

                results.append(f"   ✓ {api_key_label}: generated {key_samples} samples")

    except Exception as e:
        return f"✗ Fatal error during generation: {str(e)}"

    # Merge all temporary files
    success, merge_msg = merge_tsv_files(all_temp_info, output_file)
    results.append(merge_msg)

    final_count = total_generated if success else 0
    return (
        f"✅ Dataset created successfully!\n\n"
        + "\n".join(results)
        + f"\n\nTotal rows in final dataset: {final_count}"
    )
def merge_tsv_files(temp_info: List[Tuple], output_file: str) -> Tuple[bool, str]:
    try:
        all_dfs = []
        for tf, platform, model_name, api_idx in temp_info:
            if os.path.exists(tf):
                df = pd.read_csv(tf, sep='\t')
                df['created_by'] = get_api_key_label(platform, api_idx)
                df['created_by_model'] = f"{platform}/{model_name}"
                all_dfs.append(df)

        merged = pd.concat(all_dfs, ignore_index=True)
        Config.ensure_output_dir(output_file)
        merged.to_csv(output_file, sep='\t', index=False)

        removed = sum(1 for tf,_,_,_ in temp_info if os.path.exists(tf) and os.remove(tf) is None)

        return True, f"✓ Final dataset has {len(merged)} rows\n✓ Cleaned {removed} temporary files"
    except Exception as e:
        return False, f"Merge error: {str(e)}"


def get_download_file(output_file: str):
    try:
        path = Config.normalize_output_path(output_file)
        return path if os.path.exists(path) else None
    except:
        return None


# ====================== BUILD UI ======================
def build_interface():
    with gr.Blocks(title="GenSet - API Key Weighted Generator") as demo:
        gr.Markdown("# GenSet Dataset Generator")
        gr.Markdown("Generate synthetic datasets with weighted support for multiple models, languages, and API keys.")

        with gr.Row():
            # Left Column - Selection & Weights
            with gr.Column(scale=1):
                gr.Markdown("### Platforms")
                platforms = gr.CheckboxGroup(
                    choices=["mistral", "ollama"],
                    value=["mistral"],
                    label="Select Platforms"
                )

                gr.Markdown("### Languages")
                languages = gr.CheckboxGroup(
                    choices=LANGUAGE_CHOICES,
                    value=["english"],
                    label="Languages"
                )
                lang_weight = gr.Slider(0, 100, value=50, step=1, label="Language Weight (%)")

                gr.Markdown("### Mistral Models")
                mistral_models = gr.CheckboxGroup(
                    label="Mistral Models",
                    choices=[],
                    value=[]
                )
                mistral_model_weight = gr.Slider(
                    0, 100, value=50, step=1,
                    label="Mistral Models Weight (%)"
                )

                gr.Markdown("### Ollama Models")
                ollama_models = gr.CheckboxGroup(
                    label="Ollama Models",
                    choices=[],
                    value=[]
                )
                ollama_model_weight = gr.Slider(
                    0, 100, value=50, step=1,
                    label="Ollama Models Weight (%)"
                )

            # Middle Column - API Keys & Generation Settings
            with gr.Column(scale=1):
                gr.Markdown("### Mistral API Keys")
                mistral_api_select = gr.CheckboxGroup(
                    label="Select Mistral API Keys",
                    choices=[],
                    value=[]
                )
                mistral_api_weight = gr.Slider(
                    0, 100, value=50, step=1,
                    label="Mistral API Keys Weight (%)"
                )

                gr.Markdown("### Ollama API Keys")
                ollama_api_select = gr.CheckboxGroup(
                    label="Select Ollama API Keys",
                    choices=[],
                    value=[]
                )
                ollama_api_weight = gr.Slider(
                    0, 100, value=50, step=1,
                    label="Ollama API Keys Weight (%)"
                )

                gr.Markdown("### Generation Settings")
                domain = gr.Textbox(
                    "general customer reviews",
                    label="Domain"
                )
                labels = gr.Textbox(
                    "positive,negative",
                    label="Labels (comma-separated)"
                )
                temperature = gr.Slider(
                    0.0, 1.0, value=0.7, step=0.05,
                    label="Temperature"
                )
                num_samples = gr.Slider(
                    1, 10000, value=500, step=10,
                    label="Total Number of Samples"
                )
                balance_labels = gr.Checkbox(
                    False,
                    label="Balance Labels"
                )

            # Right Column - Output & Controls
            with gr.Column(scale=1):
                output_file = gr.Textbox(
                    str(Config.normalize_output_path(Config.DEFAULT_OUTPUT_FILE)),
                    label="Output File (.tsv)"
                )

                with gr.Row():
                    create_btn = gr.Button("📊 Create Dataset", variant="primary", size="lg")
                    sample_btn = gr.Button("🎲 Generate Sample")

                status_box = gr.Textbox(
                    label="Status",
                    lines=15,
                    show_copy_button=True
                )

                gr.Markdown("### Download")
                dl_btn = gr.Button("⬇️ Download Dataset")
                dl_file = gr.File()
                dl_status = gr.Textbox()

        # ====================== Callbacks ======================

        # Update models when platforms change
        def update_models(selected_platforms):
            all_m = get_all_models()
            mistral_choices = all_m.get("mistral", [])
            ollama_choices = all_m.get("ollama", [])
            return (
                gr.update(choices=mistral_choices, value=mistral_choices[:2] if mistral_choices else []),
                gr.update(choices=ollama_choices, value=ollama_choices[:1] if ollama_choices else [])
            )

        platforms.change(
            update_models,
            inputs=platforms,
            outputs=[mistral_models, ollama_models]
        )

        # Set API key choices from config
        mistral_api_choices = [f"mistral_{i+1}" for i in range(len(Config.MISTRAL_API_KEYS))]
        ollama_api_choices = [f"ollama_{i+1}" for i in range(len(Config.OLLAMA_API_KEYS))]

        mistral_api_select.choices = mistral_api_choices
        ollama_api_select.choices = ollama_api_choices

        # Default selection if keys exist
        if mistral_api_choices:
            mistral_api_select.value = mistral_api_choices[:1]
        if ollama_api_choices:
            ollama_api_select.value = ollama_api_choices[:1]

        # ====================== Create Dataset Event ======================
        create_btn.click(
            fn=create_dataset,
            inputs=[
                platforms,
                languages,
                lang_weight,
                mistral_models,
                mistral_model_weight,      # ← Fixed: now a real component
                ollama_models,
                ollama_model_weight,       # ← Fixed
                mistral_api_select,
                mistral_api_weight,        # ← Fixed
                ollama_api_select,
                ollama_api_weight,         # ← Fixed
                domain,
                labels,
                num_samples,
                temperature,
                output_file,
                balance_labels,
            ],
            outputs=status_box
        )

        # ====================== Sample Generation ======================
        sample_btn.click(
            fn=create_sample,
            inputs=[
                platforms,
                languages,
                domain,
                labels,
                mistral_models,
                ollama_models,
                temperature
            ],
            outputs=[
                gr.Textbox(label="Generated Text", lines=8, show_copy_button=True),
                gr.Textbox(label="Label", show_copy_button=True),
                gr.Textbox(label="Status")
            ]
        )

        # Load models on startup
        demo.load(
            update_models,
            inputs=platforms,
            outputs=[mistral_models, ollama_models]
        )

    return demo
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    demo = build_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=args.share, debug=args.debug)