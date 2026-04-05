import sys
from pathlib import Path
import argparse
import os
from typing import List, Tuple

import pandas as pd
import gradio as gr

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

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
    if ollama_models: models_dict["ollama"] = ollama_models

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

# ====================== DATASET CREATION ======================
def create_dataset(
    platforms: List[str],
    languages: List[str],
    lang_weight: float,
    mistral_models: List[str],
    mistral_model_weight: float,
    ollama_models: List[str],
    ollama_model_weight: float,
    mistral_api_keys: List[str],
    mistral_api_weight: float,
    ollama_api_keys: List[str],
    ollama_api_weight: float,
    domain: str,
    labels: str,
    num_samples: int,
    temperature: float,
    output_file: str,
    balance_labels: bool,
):
    if not platforms or not languages:
        return "⚠️ Select at least one platform and language"

    lang_weights = [float(lang_weight)] * len(languages)

    platform_config = {}
    if "mistral" in platforms and mistral_models:
        platform_config["mistral"] = (mistral_models, [float(mistral_model_weight)] * len(mistral_models))
    if "ollama" in platforms and ollama_models:
        platform_config["ollama"] = (ollama_models, [float(ollama_model_weight)] * len(ollama_models))

    if not platform_config:
        return "⚠️ Select at least one model"

    api_config = {}
    if "mistral" in platforms and mistral_api_keys:
        api_config["mistral"] = (mistral_api_keys, [float(mistral_api_weight)] * len(mistral_api_keys))
    if "ollama" in platforms and ollama_api_keys:
        api_config["ollama"] = (ollama_api_keys, [float(ollama_api_weight)] * len(ollama_api_keys))

    if not api_config:
        return "⚠️ Select at least one API key for the chosen platform(s)"

    results = []
    all_temp_info = []

    try:
        total_api_weight = 0.0
        api_tasks = []

        for platform in platforms:
            if platform not in api_config: continue
            key_names, key_weights = api_config[platform]
            norm_key_weights = normalize_weights(key_weights)
            for i, (key_name, weight) in enumerate(zip(key_names, norm_key_weights)):
                total_api_weight += weight
                api_tasks.append((platform, i, weight, key_name))

        for platform, api_idx, weight, key_label in api_tasks:
            samples_for_this_key = max(1, int(num_samples * (weight / total_api_weight) + 0.5))

            models, model_w = platform_config.get(platform, ([], []))
            if not models: continue

            norm_model = normalize_weights(model_w)
            norm_lang = normalize_weights(lang_weights)

            generator = DatasetGenerator()
            combo_samples_list = []

            for l_idx, lang in enumerate(languages):
                for m_idx, model in enumerate(models):
                    combo_weight = norm_lang[l_idx] * norm_model[m_idx]
                    combo_samples = max(1, int(samples_for_this_key * combo_weight + 0.5))
                    combo_samples_list.append((lang, model, combo_samples))

            for lang, model, combo_samples in combo_samples_list:
                if combo_samples <= 0: continue
                safe_model = model.replace("/", "_").replace(":", "_").replace(".", "_")
                temp_file = output_file.replace(".tsv", f"_{platform}_{lang}_{safe_model}_k{api_idx}.tsv")

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

            results.append(f"✓ {key_label}: generated {samples_for_this_key} samples")

    except Exception as e:
        return f"✗ Fatal error: {str(e)}"

    success, merge_msg = merge_tsv_files(all_temp_info, output_file)
    results.append(merge_msg)
    return "✅ Dataset created successfully!\n\n" + "\n".join(results) + f"\n\nTotal rows: {num_samples}"


def merge_tsv_files(temp_info: List[Tuple], output_file: str) -> Tuple[bool, str]:
    try:
        all_dfs = []
        for tf, platform, model_name, api_idx in temp_info:
            if os.path.exists(tf):
                df = pd.read_csv(tf, sep='\t')
                df['created_by'] = get_api_key_label(platform, api_idx)
                df['created_by_model'] = f"{platform}/{model_name}"
                all_dfs.append(df)

        if not all_dfs:
            return False, "No files generated"

        merged = pd.concat(all_dfs, ignore_index=True)
        Config.ensure_output_dir(output_file)
        merged.to_csv(output_file, sep='\t', index=False)

        removed = sum(1 for tf, _, _, _ in temp_info if os.path.exists(tf) and os.remove(tf) is None)
        return True, f"✓ Final dataset has {len(merged)} rows\n✓ Cleaned {removed} temporary files"
    except Exception as e:
        return False, f"Merge error: {str(e)}"


# ====================== BUILD UI ======================
def build_interface():
    with gr.Blocks(title="GenSet - Dataset Generator") as demo:
        gr.Markdown("# GenSet Dataset Generator")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Platforms")
                platforms = gr.CheckboxGroup(["mistral", "ollama"], value=["mistral"], label="Platforms")

                gr.Markdown("### Languages")
                languages = gr.CheckboxGroup(LANGUAGE_CHOICES, value=["english"], label="Languages")
                lang_weight = gr.Slider(0, 100, 50, step=1, label="Language Weight (%)")

                gr.Markdown("### Mistral Models")
                mistral_models = gr.CheckboxGroup(choices=[], value=[], label="Mistral Models")
                mistral_model_weight = gr.Slider(0, 100, 50, step=1, label="Mistral Models Weight (%)")

                gr.Markdown("### Ollama Models")
                ollama_models = gr.CheckboxGroup(choices=[], value=[], label="Ollama Models")
                ollama_model_weight = gr.Slider(0, 100, 50, step=1, label="Ollama Models Weight (%)")

            with gr.Column(scale=1):
                gr.Markdown("### Mistral API Keys")
                mistral_api_select = gr.CheckboxGroup(choices=[], value=[], label="Select Mistral API Keys")
                mistral_api_weight = gr.Slider(0, 100, 50, step=1, label="Total Mistral API Weight (%)")

                gr.Markdown("### Ollama API Keys")
                ollama_api_select = gr.CheckboxGroup(choices=[], value=[], label="Select Ollama API Keys")
                ollama_api_weight = gr.Slider(0, 100, 50, step=1, label="Total Ollama API Weight (%)")

                gr.Markdown("### Generation")
                domain = gr.Textbox("general customer reviews", label="Domain")
                labels = gr.Textbox("positive,negative", label="Labels (comma-separated)")
                temperature = gr.Slider(0.0, 1.0, 0.7, step=0.05, label="Temperature")
                num_samples = gr.Slider(10, 10000, 500, step=10, label="Total Samples")
                balance_labels = gr.Checkbox(False, label="Balance Labels")

            with gr.Column(scale=1):
                output_file = gr.Textbox(
                    value=str(Config.normalize_output_path(Config.DEFAULT_OUTPUT_FILE)),
                    label="Output File (.tsv)"
                )
                create_btn = gr.Button("📊 Create Dataset", variant="primary", size="lg")
                status_box = gr.Textbox(label="Status", lines=15, show_copy_button=True)

        # Update models when platform changes
        def update_models(selected_platforms):
            all_m = get_all_models()
            m_models = all_m.get("mistral", ["mistral-small-latest"])
            o_models = all_m.get("ollama", ["gemma3:12b-cloud"])
            return (
                gr.update(choices=m_models, value=m_models[:2]),
                gr.update(choices=o_models, value=o_models[:1])
            )

        platforms.change(update_models, platforms, [mistral_models, ollama_models])

        # Set API key choices safely (list of strings only)
        mistral_api_choices = [f"mistral_{i+1}" for i in range(len(Config.MISTRAL_API_KEYS))]
        ollama_api_choices = [f"ollama_{i+1}" for i in range(len(Config.OLLAMA_API_KEYS))]

        mistral_api_select.choices = mistral_api_choices
        ollama_api_select.choices = ollama_api_choices

        # Create Dataset
        create_btn.click(
            create_dataset,
            inputs=[
                platforms, languages, lang_weight,
                mistral_models, mistral_model_weight,
                ollama_models, ollama_model_weight,
                mistral_api_select, mistral_api_weight,
                ollama_api_select, ollama_api_weight,
                domain, labels, num_samples, temperature,
                output_file, balance_labels
            ],
            outputs=status_box
        )

        # Sample button
        gr.Button("🎲 Generate Sample").click(
            create_sample,
            inputs=[platforms, languages, domain, labels, mistral_models, ollama_models, temperature],
            outputs=[
                gr.Textbox(label="Generated Text", lines=6, show_copy_button=True),
                gr.Textbox(label="Label"),
                gr.Textbox(label="Status")
            ]
        )

        demo.load(update_models, platforms, [mistral_models, ollama_models])

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    demo = build_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=args.share, debug=args.debug)