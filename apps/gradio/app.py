import sys
from pathlib import Path
import argparse
import os
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import gradio as gr
from GenSet import Config, DatasetGenerator, LANGUAGE_CHOICES, LANGUAGE_DISPLAY, get_all_models

# Lock for thread-safe operations
status_lock = threading.Lock()


def parse_labels(label_string: str) -> list:
    labels = [label.strip() for label in label_string.split(",") if label.strip()]
    return labels or ["positive", "negative"]


def get_models_for_platforms(platforms: List[str]) -> Tuple[Dict, List[str], List[str]]:
    """Return available models for selected platforms + initial selections"""
    all_models = get_all_models()
    mistral_models = all_models.get("mistral", [])
    ollama_models = all_models.get("ollama", [])

    selected_mistral = mistral_models[:2] if mistral_models else []   # default first 2
    selected_ollama = ollama_models[:1] if ollama_models else []

    return {
        "mistral": mistral_models,
        "ollama": ollama_models
    }, selected_mistral, selected_ollama

def get_selected_models(mistral_selected: List[str], ollama_selected: List[str]) -> Dict[str, List[str]]:
    """Build models_dict from checkbox selections"""
    models_dict = {}
    if mistral_selected:
        models_dict["mistral"] = mistral_selected
    if ollama_selected:
        models_dict["ollama"] = ollama_selected
    return models_dict

def create_sample(
    platforms: List[str],
    languages: List[str],
    domain: str,
    labels: str,
    mistral_models: List[str],
    ollama_models: List[str],
    temperature: float,
):
    """Generate a sample using the first selected model of the first platform"""
    if not platforms or not languages:
        return "", "", "⚠️ Please select at least one platform and language"

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


def create_sample(
    platforms: List[str],
    languages: List[str],
    domain: str,
    labels: str,
    mistral_models: List[str],
    ollama_models: List[str],
    temperature: float,
):
    """Generate a sample using the first selected model of the first platform"""
    if not platforms or not languages:
        return "", "", "⚠️ Please select at least one platform and language"

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
    num_samples: int,
    selected_models: List[str],      # Changed: now list of models for this platform
    temperature: float,
    output_file: str,
    balance_labels: bool,
    worker_id: int,
    total_workers: int,
):
    temp_files = []
    try:
        generator = DatasetGenerator()

        # Distribute total samples for this worker across languages and models
        total_items = len(languages) * len(selected_models) or 1
        samples_per_worker = num_samples // total_workers
        if worker_id == 0:
            samples_per_worker += num_samples % total_workers

        samples_per_item = max(1, samples_per_worker // total_items)

        item_idx = 0
        for language in languages:
            for model in selected_models:
                item_samples = samples_per_item
                if item_idx < (samples_per_worker % total_items):
                    item_samples += 1
                item_idx += 1

                if item_samples <= 0:
                    continue

                temp_file = output_file.replace(".tsv", f"_{platform}_{language}_{model.replace('/', '_')}_w{worker_id}.tsv")

                result_path = generator.create_dataset(
                    num_samples=item_samples,
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
                temp_files.append(result_path)

        status = f"✓ Worker {worker_id} ({platform}): {samples_per_worker} samples using {len(selected_models)} model(s)"
        return status, temp_files

    except Exception as e:
        return f"✗ Worker {worker_id} ({platform}): Error - {str(e)}", temp_files


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
    if not models_dict:
        return "⚠️ Please select at least one model"

    if num_workers < 1:
        num_workers = 1

    results = []
    all_temp_files = []

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
                status_msg, temp_files = future.result()
                results.append(status_msg)
                all_temp_files.extend(temp_files)

    except Exception as e:
        return f"✗ Fatal error: {str(e)}"

    merge_success, merge_msg = merge_tsv_files(all_temp_files, output_file, models_dict, platforms)
    results.append(merge_msg)

    summary = "\n".join(results)
    return f"✅ Dataset creation completed!\n\n{summary}\n\nOutput: {output_file}"

def merge_tsv_files(
    temp_files: List[str],
    output_file: str,
    models_dict: Dict,
    platforms: List[str],
) -> Tuple[bool, str]:
    """
    Merge multiple TSV files into one and add the 'created_by_model' column.
    
    Args:
        temp_files: List of temporary TSV file paths
        output_file: Final merged output file path
        models_dict: Dictionary of available models per platform
        platforms: List of platforms used
        
    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        all_dfs = []
        
        for temp_file in temp_files:
            if os.path.exists(temp_file) and os.path.isfile(temp_file):
                try:
                    # Read the TSV file
                    df = pd.read_csv(temp_file, sep='\t')
                    
                    # Extract model info from filename
                    filename = os.path.basename(temp_file)
                    # Format: dataset_{platform}_{language}_w{worker_id}.tsv
                    parts = filename.replace('.tsv', '').split('_')
                    
                    model_name = None
                    for platform in platforms:
                        if platform in filename:
                            model_list = models_dict.get(platform, [])
                            if model_list:
                                model_name = f"{platform}/{model_list[0]}"
                            break
                    
                    # Add created_by_model column
                    if model_name:
                        df['created_by_model'] = model_name
                    else:
                        df['created_by_model'] = 'unknown'
                    
                    all_dfs.append(df)
                except Exception as e:
                    return (False, f"Error reading {temp_file}: {str(e)}")
        
        if not all_dfs:
            return (False, "No valid TSV files to merge")
        
        # Merge all dataframes
        merged_df = pd.concat(all_dfs, ignore_index=True)
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Save merged dataset
        merged_df.to_csv(output_file, sep='\t', index=False)
        
        # Remove temporary files
        removed_count = 0
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    removed_count += 1
            except Exception as e:
                print(f"Warning: Could not remove {temp_file}: {e}")
        
        rows_merged = len(merged_df)
        message = f"✓ Merged {len(all_dfs)} files ({rows_merged} rows) into {os.path.basename(output_file)}\n✓ Removed {removed_count} temporary files\n✓ Added 'created_by_model' column"
        return (True, message)
        
    except Exception as e:
        return (False, f"Error merging files: {str(e)}")


def get_download_file(output_file: str) -> str:
    """
    Prepare file for download.
    
    Args:
        output_file: Path to the output file
        
    Returns:
        Path to the file if it exists, otherwise error message
    """
    try:
        # Normalize the path
        file_path = Config.normalize_output_path(output_file)
        
        # Check if file exists
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return file_path
        else:
            # Try without normalization
            if os.path.exists(output_file) and os.path.isfile(output_file):
                return output_file
            else:
                return None
    except Exception as e:
        print(f"Error preparing file for download: {e}")
        return None


# ====================== BUILD INTERFACE ======================

def build_interface() -> gr.Blocks:
    with gr.Blocks(title="GenSet Dataset Generator") as demo:
        gr.Markdown("# GenSet Dataset Generator")
        gr.Markdown("Generate synthetic datasets with **multiple models per platform**")

        with gr.Row():
            # Left: Configuration
            with gr.Column(scale=1):
                gr.Markdown("### Platforms & Languages")
                platforms = gr.CheckboxGroup(
                    choices=["mistral", "ollama"],
                    value=["mistral"],
                    label="Platforms",
                    info="Select platforms"
                )
                languages = gr.CheckboxGroup(
                    choices=LANGUAGE_CHOICES,
                    value=["english"],
                    label="Languages",
                    info="Select languages"
                )

                gr.Markdown("### Models per Platform")

                mistral_models_check = gr.CheckboxGroup(
                    label="Mistral Models",
                    info="Select one or more Mistral models",
                    choices=[],
                    value=[]
                )
                ollama_models_check = gr.CheckboxGroup(
                    label="Ollama Models",
                    info="Select one or more Ollama models",
                    choices=[],
                    value=[]
                )

                # Update model checkboxes when platforms change
                def update_model_checkboxes(selected_platforms):
                    all_models = get_all_models()
                    mistral_list = all_models.get("mistral", [])
                    ollama_list = all_models.get("ollama", [])

                    mistral_visible = "mistral" in selected_platforms
                    ollama_visible = "ollama" in selected_platforms

                    return (
                        gr.update(choices=mistral_list, value=mistral_list[:2] if mistral_list else []),
                        gr.update(choices=ollama_list, value=ollama_list[:1] if ollama_list else []),
                        gr.update(visible=mistral_visible),
                        gr.update(visible=ollama_visible)
                    )

                platforms.change(
                    fn=update_model_checkboxes,
                    inputs=[platforms],
                    outputs=[mistral_models_check, ollama_models_check, mistral_models_check, ollama_models_check]
                )

                gr.Markdown("### Generation Settings")
                domain = gr.Textbox(value="general customer reviews", label="Domain")
                labels = gr.Textbox(value="positive,negative", label="Labels")
                temperature = gr.Slider(0.0, 1.0, value=0.7, step=0.05, label="Temperature")

            # Middle: Sample + Dataset controls
            with gr.Column(scale=1):
                gr.Markdown("### Generate Sample")
                with gr.Group():
                    sample_output = gr.Textbox(label="Generated Text", lines=6, show_copy_button=True)
                    sample_label = gr.Textbox(label="Label", show_copy_button=True)
                    sample_status = gr.Textbox(label="Status")
                    generate_sample_btn = gr.Button("🎲 Generate Sample", variant="primary")

                gr.Markdown("### Dataset Creation")
                num_samples = gr.Slider(1, 1000, value=50, step=1, label="Number of Samples (Total)")
                num_workers = gr.Slider(1, 16, value=2, step=1, label="Number of Workers")
                balance_labels = gr.Checkbox(value=False, label="Balance Labels")

            # Right: Output
            with gr.Column(scale=1):
                gr.Markdown("### Output")
                output_file = gr.Textbox(
                    value=str(Config.normalize_output_path(Config.DEFAULT_OUTPUT_FILE)),
                    label="Output File (.tsv)"
                )

                create_dataset_btn = gr.Button("📊 Create Dataset", variant="primary", size="lg")
                dataset_status = gr.Textbox(label="Status", lines=12, show_copy_button=True)

                gr.Markdown("### Download")
                download_btn = gr.Button("⬇️ Download Dataset", variant="secondary")
                download_file = gr.File(label="Download File")
                download_status = gr.Textbox(label="Download Status")

        # Event Handlers
        generate_sample_btn.click(
            fn=create_sample,
            inputs=[platforms, languages, domain, labels, mistral_models_check, ollama_models_check, temperature],
            outputs=[sample_output, sample_label, sample_status]
        )

        create_dataset_btn.click(
            fn=create_dataset,
            inputs=[
                platforms, languages, domain, labels, num_samples,
                mistral_models_check, ollama_models_check,
                temperature, output_file, balance_labels, num_workers
            ],
            outputs=[dataset_status]
        )

        download_btn.click(
            fn=get_download_file,
            inputs=[output_file],
            outputs=[download_file]
        ).then(
            lambda f: "✓ File ready!" if f else "✗ File not found. Create dataset first.",
            inputs=[download_file],
            outputs=[download_status]
        )

        # Initial load
        demo.load(
            fn=update_model_checkboxes,
            inputs=[platforms],
            outputs=[mistral_models_check, ollama_models_check, mistral_models_check, ollama_models_check]
        )

    return demo

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GenSet Gradio Web Interface")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    demo = build_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=args.share, debug=args.debug)
