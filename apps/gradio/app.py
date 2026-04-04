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
from GenSet import Config, DatasetGenerator, LANGUAGE_CHOICES, LANGUAGE_DISPLAY, get_all_models

def parse_labels(label_string: str) -> list:
    labels = [label.strip() for label in label_string.split(",") if label.strip()]
    return labels or ["positive", "negative"]


def get_models_for_platforms(platforms: List[str]):
    """Return model lists for selected platforms"""
    all_models = get_all_models()
    return {
        "mistral": all_models.get("mistral", []),
        "ollama": all_models.get("ollama", [])
    }


def get_selected_models(mistral_selected: List[str], ollama_selected: List[str]) -> Dict[str, List[str]]:
    models_dict = {}
    if mistral_selected:
        models_dict["mistral"] = mistral_selected
    if ollama_selected:
        models_dict["ollama"] = ollama_selected
    return models_dict


def get_api_key_label(platform: str, model_idx: int) -> str:
    """Return ollama_1, mistral_2, etc."""
    return f"{platform}_{model_idx + 1}"


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
        return "", "", "⚠️ Select at least one platform and language"

    models_dict = get_selected_models(mistral_models, ollama_models)
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
        return example["text"], example["label"], f"✓ Sample from {platform}/{model} ({language})"
    except Exception as e:
        return "", "", f"✗ Error: {str(e)}"


def create_dataset_worker(
    platform: str,
    languages: List[str],
    domain: str,
    labels: str,
    num_samples: int,           # Total samples for the whole dataset
    selected_models: List[str],
    temperature: float,
    output_file: str,
    balance_labels: bool,
    worker_id: int,
    total_workers: int,
):
    temp_files = []
    try:
        generator = DatasetGenerator()

        # Total combinations for this platform: languages × models
        total_combos = len(languages) * len(selected_models)
        if total_combos == 0:
            return f"Worker {worker_id} skipped (no models/languages)", []

        # How many samples this worker should generate
        samples_per_worker = num_samples // total_workers
        if worker_id == 0:
            samples_per_worker += num_samples % total_workers

        # Distribute as evenly as possible across combos
        samples_per_combo = samples_per_worker // total_combos
        remainder = samples_per_worker % total_combos

        combo_idx = 0
        for language in languages:
            for model_idx, model in enumerate(selected_models):
                combo_samples = samples_per_combo + (1 if combo_idx < remainder else 0)
                combo_idx += 1

                if combo_samples <= 0:
                    continue

                # Create unique temp file name
                safe_model = model.replace("/", "_").replace(":", "_")
                temp_file = output_file.replace(".tsv", f"_{platform}_{language}_{safe_model}_w{worker_id}.tsv")

                result_path = generator.create_dataset(
                    num_samples=combo_samples,
                    platform=platform,
                    language=language,
                    labels=parse_labels(labels),
                    domain=domain,
                    model=model,
                    temperature=temperature,
                    output_file=temp_file,
                    delay=0.8,
                    multilingual=False,
                    balance_labels=balance_labels,
                )
                temp_files.append((result_path, platform, model, model_idx))  # store extra info for merging

        status = f"✓ Worker {worker_id} ({platform}): {samples_per_worker} samples across {len(selected_models)} model(s)"
        return status, temp_files

    except Exception as e:
        return f"✗ Worker {worker_id} ({platform}): Error - {str(e)}", temp_files


def merge_tsv_files(
    temp_files_info: List[Tuple[str, str, str, int]],   # (filepath, platform, model, model_idx)
    output_file: str,
    models_dict: Dict,
    platforms: List[str],
) -> Tuple[bool, str]:
    try:
        all_dfs = []

        for temp_file, platform, model_name, model_idx in temp_files_info:
            if not os.path.exists(temp_file):
                continue

            df = pd.read_csv(temp_file, sep='\t')

            # New column: platform_key like "mistral_1", "ollama_2"
            key_label = get_api_key_label(platform, model_idx)

            df['created_by'] = key_label
            df['created_by_model'] = f"{platform}/{model_name}"

            all_dfs.append(df)

        if not all_dfs:
            return False, "No files to merge"

        merged_df = pd.concat(all_dfs, ignore_index=True)

        # Ensure output directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        merged_df.to_csv(output_file, sep='\t', index=False)

        # Cleanup temp files
        removed = 0
        for temp_file, _, _, _ in temp_files_info:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    removed += 1
            except:
                pass

        final_rows = len(merged_df)
        msg = (f"✓ Merged {len(all_dfs)} files → {final_rows} total rows\n"
               f"✓ Added columns: 'created_by' (e.g. mistral_1, ollama_2) and 'created_by_model'\n"
               f"✓ Removed {removed} temporary files")
        return True, msg

    except Exception as e:
        return False, f"Merge error: {str(e)}"


def create_dataset(
    platforms: List[str],
    languages: List[str],
    domain: str,
    labels: str,
    num_samples: int,
    mistral_models: List[str],
    ollama_models: List[str],
    temperature: float,
    output_file: str,
    balance_labels: bool,
    num_workers: int,
):
    if not platforms or not languages:
        return "⚠️ Please select at least one platform and one language"

    models_dict = get_selected_models(mistral_models, ollama_models)
    if not any(models_dict.values()):
        return "⚠️ Please select at least one model"

    if num_workers < 1:
        num_workers = 1

    results = []
    all_temp_files_info = []   # list of (path, platform, model, model_idx)

    try:
        with ThreadPoolExecutor(max_workers=num_workers * len(platforms)) as executor:
            futures = {}
            worker_id = 0

            for platform in platforms:
                selected_models = models_dict.get(platform, [])
                if not selected_models:
                    continue

                for w in range(num_workers):
                    future = executor.submit(
                        create_dataset_worker,
                        platform=platform,
                        languages=languages,
                        domain=domain,
                        labels=labels,
                        num_samples=num_samples,
                        selected_models=selected_models,
                        temperature=temperature,
                        output_file=output_file,
                        balance_labels=balance_labels,
                        worker_id=worker_id,
                        total_workers=num_workers * len(platforms),
                    )
                    futures[future] = worker_id
                    worker_id += 1

            for future in as_completed(futures):
                status_msg, temp_info = future.result()
                results.append(status_msg)
                all_temp_files_info.extend(temp_info)

    except Exception as e:
        return f"✗ Fatal error: {str(e)}"

    merge_success, merge_msg = merge_tsv_files(all_temp_files_info, output_file, models_dict, platforms)
    results.append(merge_msg)

    summary = "\n".join(results)
    return f"✅ Dataset creation completed!\n\n{summary}\n\nFinal file: {output_file}"


# Keep your existing get_download_file function unchanged
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


# ====================== BUILD INTERFACE ======================
def build_interface() -> gr.Blocks:
    with gr.Blocks(title="GenSet Dataset Generator") as demo:
        gr.Markdown("# GenSet Dataset Generator")
        gr.Markdown("Multi-model support • Nearly equal sample distribution • created_by column (mistral_1, ollama_2, ...)")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Platforms & Languages")
                platforms = gr.CheckboxGroup(["mistral", "ollama"], value=["mistral"], label="Platforms")
                languages = gr.CheckboxGroup(LANGUAGE_CHOICES, value=["english"], label="Languages")

                gr.Markdown("### Models (multi-select)")
                mistral_models_check = gr.CheckboxGroup(label="Mistral Models", choices=[], value=[])
                ollama_models_check = gr.CheckboxGroup(label="Ollama Models", choices=[], value=[])

                def update_models(selected_platforms):
                    all_m = get_models_for_platforms(selected_platforms)
                    m_list = all_m["mistral"]
                    o_list = all_m["ollama"]
                    return (
                        gr.update(choices=m_list, value=m_list[:2] if m_list else []),
                        gr.update(choices=o_list, value=o_list[:1] if o_list else []),
                        gr.update(visible="mistral" in selected_platforms),
                        gr.update(visible="ollama" in selected_platforms)
                    )

                platforms.change(
                    update_models,
                    inputs=[platforms],
                    outputs=[mistral_models_check, ollama_models_check, mistral_models_check, ollama_models_check]
                )

                gr.Markdown("### Generation Settings")
                domain = gr.Textbox(value="general customer reviews", label="Domain")
                labels = gr.Textbox(value="positive,negative", label="Labels")
                temperature = gr.Slider(0.0, 1.0, 0.7, step=0.05, label="Temperature")

            with gr.Column(scale=1):
                gr.Markdown("### Generate Sample")
                sample_output = gr.Textbox(label="Generated Text", lines=6, show_copy_button=True)
                sample_label_out = gr.Textbox(label="Label", show_copy_button=True)
                sample_status = gr.Textbox(label="Status")
                gr.Button("🎲 Generate Sample", variant="primary").click(
                    create_sample,
                    inputs=[platforms, languages, domain, labels, mistral_models_check, ollama_models_check, temperature],
                    outputs=[sample_output, sample_label_out, sample_status]
                )

                gr.Markdown("### Dataset Creation")
                num_samples = gr.Slider(1, 2000, value=100, step=1, label="Total Number of Samples")
                num_workers = gr.Slider(1, 16, value=2, step=1, label="Number of Workers")
                balance_labels = gr.Checkbox(False, label="Balance Labels")

            with gr.Column(scale=1):
                gr.Markdown("### Output")
                output_file = gr.Textbox(value=str(Config.normalize_output_path(Config.DEFAULT_OUTPUT_FILE)), label="Output File")
                create_btn = gr.Button("📊 Create Dataset", variant="primary", size="lg")
                status_box = gr.Textbox(label="Creation Status", lines=14, show_copy_button=True)

                gr.Markdown("### Download")
                dl_btn = gr.Button("⬇️ Download Dataset")
                dl_file = gr.File(label="Download File")
                dl_status = gr.Textbox(label="Download Status")

        # Dataset creation event
        create_btn.click(
            create_dataset,
            inputs=[
                platforms, languages, domain, labels, num_samples,
                mistral_models_check, ollama_models_check,
                temperature, output_file, balance_labels, num_workers
            ],
            outputs=[status_box]
        )

        # Download
        dl_btn.click(
            get_download_file, inputs=[output_file], outputs=[dl_file]
        ).then(
            lambda f: "✓ File ready for download!" if f else "✗ File not found. Create dataset first.",
            inputs=[dl_file], outputs=[dl_status]
        )

        # Initial load
        demo.load(
            update_models,
            inputs=[platforms],
            outputs=[mistral_models_check, ollama_models_check, mistral_models_check, ollama_models_check]
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GenSet Gradio")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    demo = build_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=args.share, debug=args.debug)