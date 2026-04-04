import sys
from pathlib import Path
import argparse
import os
from typing import List, Dict, Tuple
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


# ====================== SAMPLE GENERATION ======================
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
    if mistral_models:
        models_dict["mistral"] = mistral_models
    if ollama_models:
        models_dict["ollama"] = ollama_models

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
        return (
            example.get("text", ""),
            example.get("label", ""),
            f"✓ Sample generated using {platform}/{model} ({language})"
        )
    except Exception as e:
        return "", "", f"✗ Error: {str(e)}"


# ====================== DATASET WORKER ======================
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

        # Total samples for this worker
        samples_per_worker = num_samples // total_workers
        if worker_id == 0:
            samples_per_worker += num_samples % total_workers

        # Build combinations with weights
        combos = []
        for l_idx, lang in enumerate(languages):
            for m_idx, model in enumerate(models):
                weight = norm_lang[l_idx] * norm_model[m_idx]
                combos.append((lang, model, weight, l_idx, m_idx))

        total_weight = sum(w for _, _, w, _, _ in combos)

        for lang, model, weight, _, _ in combos:
            if weight <= 0:
                continue
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

        status = f"✓ Worker {worker_id} ({platform}, API key {api_key_idx+1}): {samples_per_worker} samples"
        return status, temp_files_info

    except Exception as e:
        return f"✗ Worker {worker_id} ({platform}) Error: {str(e)}", temp_files_info


# ====================== MERGE ======================
def merge_tsv_files(temp_files_info: List[Tuple[str, str, str, int]], output_file: str) -> Tuple[bool, str]:
    try:
        all_dfs = []
        for temp_file, platform, model_name, api_idx in temp_files_info:
            if os.path.exists(temp_file):
                df = pd.read_csv(temp_file, sep='\t')
                df['created_by'] = get_api_key_label(platform, api_idx)
                df['created_by_model'] = f"{platform}/{model_name}"
                all_dfs.append(df)

        if not all_dfs:
            return False, "No valid files to merge"

        merged_df = pd.concat(all_dfs, ignore_index=True)

        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        merged_df.to_csv(output_file, sep='\t', index=False)

        # Cleanup temp files
        removed = 0
        for tf, _, _, _ in temp_files_info:
            try:
                if os.path.exists(tf):
                    os.remove(tf)
                    removed += 1
            except:
                pass

        return True, (f"✓ Successfully merged {len(all_dfs)} files → {len(merged_df)} total rows\n"
                      f"✓ Added columns: created_by (e.g. mistral_1, ollama_2), created_by_model\n"
                      f"✓ Removed {removed} temporary files")

    except Exception as e:
        return False, f"✗ Merge error: {str(e)}"


# ====================== MAIN DATASET CREATION ======================
def create_dataset(
    platforms: List[str],
    languages: List[str],
    lang_weights: List[float],
    mistral_models: List[str],
    mistral_weights: List[float],
    ollama_models: List[str],
    ollama_weights: List[float],
    domain: str,
    labels: str,
    num_samples: int,
    temperature: float,
    output_file: str,
    balance_labels: bool,
    num_workers: int,
):
    if not platforms or not languages:
        return "⚠️ Please select at least one platform and one language"

    # Build platform configs
    platform_config = {}
    if "mistral" in platforms and mistral_models:
        platform_config["mistral"] = (mistral_models, mistral_weights)
    if "ollama" in platforms and ollama_models:
        platform_config["ollama"] = (ollama_models, ollama_weights)

    if not platform_config:
        return "⚠️ Please select at least one model for the chosen platform(s)"

    results = []
    all_temp_info = []

    total_workers = max(1, num_workers * len(platforms))
    worker_id = 0

    try:
        with ThreadPoolExecutor(max_workers=total_workers) as executor:
            futures = []

            for platform in platforms:
                if platform not in platform_config:
                    continue
                models, model_weights = platform_config[platform]

                for _ in range(num_workers):
                    # Rotate API key index per worker
                    api_key_idx = worker_id % max(
                        len(Config.MISTRAL_API_KEYS) if platform == "mistral" else len(Config.OLLAMA_API_KEYS),
                        1
                    )

                    future = executor.submit(
                        create_dataset_worker,
                        platform=platform,
                        languages=languages,
                        lang_weights=lang_weights,
                        models=models,
                        model_weights=model_weights,
                        domain=domain,
                        labels=labels,
                        num_samples=num_samples,
                        temperature=temperature,
                        output_file=output_file,
                        balance_labels=balance_labels,
                        worker_id=worker_id,
                        total_workers=total_workers,
                        api_key_idx=api_key_idx,
                    )
                    futures.append(future)
                    worker_id += 1

            for future in as_completed(futures):
                status, temp_info = future.result()
                results.append(status)
                all_temp_info.extend(temp_info)

    except Exception as e:
        return f"✗ Fatal error during generation: {str(e)}"

    merge_success, merge_msg = merge_tsv_files(all_temp_info, output_file)
    results.append(merge_msg)

    summary = "\n".join(results)
    return f"✅ Dataset creation completed!\n\n{summary}\n\nFinal output: {output_file}"


def get_download_file(output_file: str):
    try:
        file_path = Config.normalize_output_path(output_file)
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return file_path
        if os.path.exists(output_file) and os.path.isfile(output_file):
            return output_file
        return None
    except Exception:
        return None


# ====================== BUILD GRADIO INTERFACE ======================
def build_interface() -> gr.Blocks:
    with gr.Blocks(title="GenSet Weighted Dataset Generator") as demo:
        gr.Markdown("# GenSet Dataset Generator")
        gr.Markdown("**Weighted distribution** across Platforms • Languages • Models • API Keys")

        with gr.Row(equal_height=True):
            # Left Column - Configuration
            with gr.Column(scale=1):
                gr.Markdown("### 1. Platforms")
                platforms = gr.CheckboxGroup(
                    choices=["mistral", "ollama"],
                    value=["mistral"],
                    label="Select Platforms"
                )

                gr.Markdown("### 2. Languages & Weights")
                languages = gr.CheckboxGroup(
                    choices=LANGUAGE_CHOICES,
                    value=["english"],
                    label="Select Languages"
                )
                lang_weights = gr.Slider(0, 100, value=50, step=1, label="Language Weight (%)", info="Applies to all selected languages equally for now")

                gr.Markdown("### 3. Models")
                mistral_models_check = gr.CheckboxGroup(
                    label="Mistral Models (multi-select)",
                    choices=[],
                    value=[]
                )
                mistral_model_weights = gr.Slider(0, 100, value=50, step=1, label="Mistral Model Weight (%)")

                ollama_models_check = gr.CheckboxGroup(
                    label="Ollama Models (multi-select)",
                    choices=[],
                    value=[]
                )
                ollama_model_weights = gr.Slider(0, 100, value=50, step=1, label="Ollama Model Weight (%)")

            # Middle Column - Generation Settings
            with gr.Column(scale=1):
                gr.Markdown("### Generation Settings")
                domain = gr.Textbox(value="general customer reviews", label="Domain")
                labels = gr.Textbox(value="positive,negative", label="Labels (comma separated)")
                temperature = gr.Slider(0.0, 1.0, value=0.7, step=0.05, label="Temperature")

                gr.Markdown("### Dataset Size")
                num_samples = gr.Slider(1, 5000, value=200, step=10, label="Total Number of Samples")
                num_workers = gr.Slider(1, 20, value=4, step=1, label="Number of Workers (API Key Rotation)")
                balance_labels = gr.Checkbox(value=False, label="Balance Labels")

            # Right Column - Output
            with gr.Column(scale=1):
                gr.Markdown("### Output")
                output_file = gr.Textbox(
                    value=str(Config.normalize_output_path(Config.DEFAULT_OUTPUT_FILE)),
                    label="Output File (.tsv)"
                )

                create_btn = gr.Button("📊 Create Weighted Dataset", variant="primary", size="lg")
                status_box = gr.Textbox(label="Creation Status", lines=14, show_copy_button=True)

                gr.Markdown("### Download")
                download_btn = gr.Button("⬇️ Download Dataset", variant="secondary")
                download_file = gr.File(label="Download File")
                download_status = gr.Textbox(label="Download Status")

        # Update model choices when platforms change
        def update_model_choices(selected_platforms):
            all_models = get_all_models()
            mistral_list = all_models.get("mistral", [])
            ollama_list = all_models.get("ollama", [])

            return (
                gr.update(choices=mistral_list, value=mistral_list[:2] if mistral_list else []),
                gr.update(choices=ollama_list, value=ollama_list[:1] if ollama_list else []),
            )

        platforms.change(
            update_model_choices,
            inputs=[platforms],
            outputs=[mistral_models_check, ollama_models_check]
        )

        # Sample Button
        gr.Button("🎲 Generate Sample", variant="secondary").click(
            create_sample,
            inputs=[platforms, languages, domain, labels, mistral_models_check, ollama_models_check, temperature],
            outputs=[gr.Textbox(label="Generated Text", lines=6, show_copy_button=True),
                     gr.Textbox(label="Label", show_copy_button=True),
                     gr.Textbox(label="Status")]
        )

        # Create Dataset Button
        create_btn.click(
            create_dataset,
            inputs=[
                platforms,
                languages,
                [lang_weights.value] * len(languages) if languages else [50],  # simplified
                mistral_models_check,
                [mistral_model_weights.value] * len(mistral_models_check) if mistral_models_check else [50],
                ollama_models_check,
                [ollama_model_weights.value] * len(ollama_models_check) if ollama_models_check else [50],
                domain,
                labels,
                num_samples,
                temperature,
                output_file,
                balance_labels,
                num_workers
            ],
            outputs=[status_box]
        )

        # Download
        download_btn.click(
            get_download_file,
            inputs=[output_file],
            outputs=[download_file]
        ).then(
            lambda f: "✓ File ready for download!" if f else "✗ File not found. Create dataset first.",
            inputs=[download_file],
            outputs=[download_status]
        )

        # Initial load
        demo.load(
            update_model_choices,
            inputs=[platforms],
            outputs=[mistral_models_check, ollama_models_check]
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GenSet Gradio Web Interface")
    parser.add_argument("--share", action="store_true", help="Enable public sharing link")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    demo = build_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=args.share,
        debug=args.debug
    )