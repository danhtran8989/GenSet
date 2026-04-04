import sys
from pathlib import Path
import argparse
import os
from typing import List, Tuple
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


def get_api_key_label(platform: str, api_idx: int) -> str:
    return f"{platform}_{api_idx + 1}"


# ====================== SAMPLE ======================
def create_sample(
    platforms: List[str],
    languages: List[str],
    domain: str,
    labels: str,
    mistral_models: List[str],
    ollama_models: List[str],
    temperature: float,
):
    if not platforms or not languages:
        return "", "", "⚠️ Please select at least one platform and one language"

    models_dict = {}
    if mistral_models: models_dict["mistral"] = mistral_models
    if ollama_models:  models_dict["ollama"] = ollama_models

    platform = platforms[0]
    language = languages[0]
    model_list = models_dict.get(platform, [])
    model = model_list[0] if model_list else None

    try:
        generator = DatasetGenerator()
        example = generator.generate_example(
            platform=platform,
            language=language,
            labels=parse_labels(labels),
            domain=domain,
            model=model,
            temperature=temperature,
        )
        return example.get("text", ""), example.get("label", ""), f"✓ Sample from {platform}/{model} ({language})"
    except Exception as e:
        return "", "", f"✗ Error: {str(e)}"


# ====================== WORKER ======================
def create_dataset_worker(
    platform: str,
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
    worker_id: int,
    total_workers: int,
    api_key_idx: int,
):
    temp_files_info = []
    try:
        generator = DatasetGenerator()
        norm_lang = normalize_weights(lang_weights)
        norm_model = normalize_weights(model_weights)

        samples_per_worker = num_samples // total_workers
        if worker_id == 0:
            samples_per_worker += num_samples % total_workers

        combos = []
        for l_idx, lang in enumerate(languages):
            for m_idx, model in enumerate(models):
                weight = norm_lang[l_idx] * norm_model[m_idx]
                combos.append((lang, model, weight))

        total_weight = sum(w for _, _, w in combos) or 1.0

        for lang, model, weight in combos:
            if weight <= 0: continue
            combo_samples = max(1, int(samples_per_worker * (weight / total_weight)))

            safe_model = model.replace("/", "_").replace(":", "_").replace(".", "_")
            temp_file = output_file.replace(".tsv", f"_{platform}_{lang}_{safe_model}_w{worker_id}.tsv")

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
            temp_files_info.append((result_path, platform, model, api_key_idx))

        return f"✓ Worker {worker_id} ({platform}, API key {api_key_idx+1}): {samples_per_worker} samples", temp_files_info

    except Exception as e:
        return f"✗ Worker {worker_id} error: {str(e)}", temp_files_info


# ====================== MERGE ======================
def merge_tsv_files(temp_files_info: List[Tuple], output_file: str) -> Tuple[bool, str]:
    try:
        all_dfs = []
        for temp_file, platform, model_name, api_idx in temp_files_info:
            if os.path.exists(temp_file):
                df = pd.read_csv(temp_file, sep='\t')
                df['created_by'] = get_api_key_label(platform, api_idx)
                df['created_by_model'] = f"{platform}/{model_name}"
                all_dfs.append(df)

        if not all_dfs:
            return False, "No files to merge"

        merged_df = pd.concat(all_dfs, ignore_index=True)
        Config.ensure_output_dir(output_file)
        merged_df.to_csv(output_file, sep='\t', index=False)

        removed = sum(1 for tf, _, _, _ in temp_files_info if os.path.exists(tf) and os.remove(tf) is None)

        return True, (f"✓ Merged {len(all_dfs)} files → {len(merged_df)} rows\n"
                      f"✓ Columns: created_by (mistral_1, ollama_2...), created_by_model\n"
                      f"✓ Removed {removed} temp files")
    except Exception as e:
        return False, f"✗ Merge error: {str(e)}"


# ====================== CREATE DATASET ======================
def create_dataset(
    platforms, languages, lang_weight_val,
    mistral_models, mistral_weight_val,
    ollama_models, ollama_weight_val,
    domain, labels, num_samples, temperature,
    output_file, balance_labels, num_workers
):
    if not platforms or not languages:
        return "⚠️ Please select at least one platform and one language"

    # Build weights lists
    lang_weights = [float(lang_weight_val)] * len(languages)
    mistral_weights = [float(mistral_weight_val)] * len(mistral_models) if mistral_models else []
    ollama_weights = [float(ollama_weight_val)] * len(ollama_models) if ollama_models else []

    platform_config = {}
    if "mistral" in platforms and mistral_models:
        platform_config["mistral"] = (mistral_models, mistral_weights)
    if "ollama" in platforms and ollama_models:
        platform_config["ollama"] = (ollama_models, ollama_weights)

    if not platform_config:
        return "⚠️ Please select at least one model"

    results = []
    all_temp_info = []
    total_workers = max(1, num_workers * len(platforms))
    worker_id = 0

    try:
        with ThreadPoolExecutor(max_workers=total_workers) as executor:
            futures = []
            for platform in platforms:
                if platform not in platform_config: continue
                models, model_w = platform_config[platform]
                for _ in range(num_workers):
                    api_key_idx = worker_id % max(
                        len(Config.MISTRAL_API_KEYS) if platform == "mistral" else len(Config.OLLAMA_API_KEYS), 1
                    )
                    future = executor.submit(
                        create_dataset_worker,
                        platform, languages, lang_weights, models, model_w,
                        domain, labels, num_samples, temperature, output_file,
                        balance_labels, worker_id, total_workers, api_key_idx
                    )
                    futures.append(future)
                    worker_id += 1

            for future in as_completed(futures):
                status, temp_info = future.result()
                results.append(status)
                all_temp_info.extend(temp_info)

    except Exception as e:
        return f"✗ Fatal error: {str(e)}"

    _, merge_msg = merge_tsv_files(all_temp_info, output_file)
    results.append(merge_msg)
    return "✅ Dataset creation completed!\n\n" + "\n".join(results) + f"\n\nOutput: {output_file}"


def get_download_file(output_file: str):
    try:
        file_path = Config.normalize_output_path(output_file)
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return file_path
        if os.path.exists(output_file) and os.path.isfile(output_file):
            return output_file
        return None
    except:
        return None


# ====================== INTERFACE ======================
def build_interface() -> gr.Blocks:
    with gr.Blocks(title="GenSet Weighted Dataset Generator") as demo:
        gr.Markdown("# GenSet Dataset Generator")
        gr.Markdown("Weighted distribution across Platforms • Languages • Models • API Keys")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Platforms")
                platforms = gr.CheckboxGroup(["mistral", "ollama"], value=["mistral"], label="Platforms")

                gr.Markdown("### Languages & Weight")
                languages = gr.CheckboxGroup(LANGUAGE_CHOICES, value=["english"], label="Languages")
                lang_weight = gr.Slider(0, 100, value=50, step=1, label="Language Weight (%)")

                gr.Markdown("### Mistral Models")
                mistral_models = gr.CheckboxGroup(label="Select Mistral Models", choices=[], value=[])
                mistral_weight = gr.Slider(0, 100, value=50, step=1, label="Mistral Models Weight (%)")

                gr.Markdown("### Ollama Models")
                ollama_models = gr.CheckboxGroup(label="Select Ollama Models", choices=[], value=[])
                ollama_weight = gr.Slider(0, 100, value=50, step=1, label="Ollama Models Weight (%)")

            with gr.Column(scale=1):
                gr.Markdown("### Generation Settings")
                domain = gr.Textbox(value="general customer reviews", label="Domain")
                labels = gr.Textbox(value="positive,negative", label="Labels")
                temperature = gr.Slider(0.0, 1.0, 0.7, step=0.05, label="Temperature")

                gr.Markdown("### Size & Workers")
                num_samples = gr.Slider(1, 5000, value=200, step=10, label="Total Samples")
                num_workers = gr.Slider(1, 20, value=4, step=1, label="Number of Workers")
                balance_labels = gr.Checkbox(False, label="Balance Labels")

            with gr.Column(scale=1):
                gr.Markdown("### Output")
                output_file = gr.Textbox(
                    value=str(Config.normalize_output_path(Config.DEFAULT_OUTPUT_FILE)),
                    label="Output File"
                )
                create_btn = gr.Button("📊 Create Weighted Dataset", variant="primary", size="lg")
                status_box = gr.Textbox(label="Status", lines=15, show_copy_button=True)

                gr.Markdown("### Download")
                dl_btn = gr.Button("⬇️ Download Dataset")
                dl_file = gr.File(label="Download File")
                dl_status = gr.Textbox(label="Download Status")

        # Update model lists when platform changes
        def update_models(selected_platforms):
            all_m = get_all_models()
            m_list = all_m.get("mistral", [])
            o_list = all_m.get("ollama", [])
            return (
                gr.update(choices=m_list, value=m_list[:2] if m_list else []),
                gr.update(choices=o_list, value=o_list[:1] if o_list else [])
            )

        platforms.change(
            update_models,
            inputs=platforms,
            outputs=[mistral_models, ollama_models]
        )

        # Sample button (for quick testing)
        gr.Button("🎲 Generate Sample", variant="secondary").click(
            create_sample,
            inputs=[platforms, languages, domain, labels, mistral_models, ollama_models, temperature],
            outputs=[
                gr.Textbox(label="Generated Text", lines=6, show_copy_button=True),
                gr.Textbox(label="Label", show_copy_button=True),
                gr.Textbox(label="Status")
            ]
        )

        # Main Create Dataset button
        create_btn.click(
            fn=create_dataset,
            inputs=[
                platforms, languages, lang_weight,
                mistral_models, mistral_weight,
                ollama_models, ollama_weight,
                domain, labels, num_samples, temperature,
                output_file, balance_labels, num_workers
            ],
            outputs=status_box
        )

        # Download
        dl_btn.click(
            get_download_file, inputs=[output_file], outputs=[dl_file]
        ).then(
            lambda f: "✓ File ready!" if f else "✗ File not found. Create dataset first.",
            inputs=[dl_file], outputs=[dl_status]
        )

        # Load default models
        demo.load(
            update_models,
            inputs=platforms,
            outputs=[mistral_models, ollama_models]
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GenSet Gradio Interface")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    demo = build_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=args.share, debug=args.debug)