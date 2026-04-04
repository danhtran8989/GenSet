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


def get_models_for_platforms(platforms: List[str]) -> Dict[str, List[str]]:
    """Get available models for selected platforms."""
    all_models = get_all_models()
    models = {}
    for platform in platforms:
        if platform in all_models:
            models[platform] = all_models[platform]
    return models


def create_sample(
    platforms: List[str],
    languages: List[str],
    domain: str,
    labels: str,
    models_dict: Dict,
    temperature: float,
):
    """Generate a sample from the first platform and language."""
    if not platforms or not languages:
        return "", "", "⚠️ Please select at least one platform and language"
    
    platform = platforms[0]
    language = languages[0]
    model = models_dict.get(platform, [""])[0] if models_dict.get(platform) else None
    
    try:
        generator = DatasetGenerator()
        example = generator.generate_example(
            platform=platform,
            language=language,
            labels=parse_labels(labels),
            domain=domain,
            model=model or None,
            temperature=temperature,
        )
        return example["text"], example["label"], f"✓ Sample generated from {platform} ({language})"
    except Exception as e:
        return "", "", f"✗ Error: {str(e)}"


def create_dataset_worker(
    platform: str,
    languages: List[str],
    domain: str,
    labels: str,
    num_samples: int,           # This is now TOTAL samples across everything
    models_dict: Dict,
    temperature: float,
    output_file: str,
    balance_labels: bool,
    worker_id: int,
    total_workers: int,
    total_combinations: int,    # NEW: number of (platform, language) pairs
) -> Tuple[str, List[str]]:
    """
    Worker function to create dataset portion in parallel.
    Now respects total num_samples across all platforms and languages.
    """
    temp_files = []
    try:
        generator = DatasetGenerator()
        model = models_dict.get(platform, [""])[0] if models_dict.get(platform) else None

        # Calculate how many samples this worker should generate in total
        samples_per_worker = num_samples // total_workers
        if worker_id == 0:
            samples_per_worker += num_samples % total_workers

        # Distribute samples across languages for this platform
        samples_per_language = samples_per_worker // len(languages)
        remainder = samples_per_worker % len(languages)

        for lang_idx, language in enumerate(languages):
            lang_samples = samples_per_language + (1 if lang_idx < remainder else 0)

            # Further distribute across total combinations if needed for fairness
            # But for simplicity, we do per-platform language distribution first

            if lang_samples <= 0:
                continue

            temp_file = output_file.replace(".tsv", f"_{platform}_{language}_w{worker_id}.tsv")
            
            result_path = generator.create_dataset(
                num_samples=lang_samples,                    # <-- Now per language
                platform=platform,
                language=language,
                labels=parse_labels(labels),
                domain=domain,
                model=model or None,
                temperature=temperature,
                output_file=temp_file,
                delay=0.8,
                multilingual=False,      # We handle multi-language at app level
                balance_labels=balance_labels,
            )
            temp_files.append(result_path)

        status = f"✓ Worker {worker_id} ({platform}): Generated {samples_per_worker} samples across {len(languages)} languages"
        return (status, temp_files)

    except Exception as e:
        return (f"✗ Worker {worker_id} ({platform}): Error - {str(e)}", temp_files)

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


def create_dataset(
    platforms: List[str],
    languages: List[str],
    domain: str,
    labels: str,
    num_samples: int,           # TRUE total number of rows in final dataset
    models_dict: Dict,
    temperature: float,
    output_file: str,
    balance_labels: bool,
    num_workers: int,
):
    """Create dataset with exact total number of samples across all combinations."""
    if not platforms or not languages:
        return "⚠️ Please select at least one platform and one language"

    if num_workers < 1:
        num_workers = 1

    total_combinations = len(platforms) * len(languages)
    if total_combinations == 0:
        return "⚠️ Invalid platform/language combination"

    results = []
    all_temp_files = []

    try:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {}
            worker_id = 0

            for platform in platforms:
                for w in range(num_workers):
                    future = executor.submit(
                        create_dataset_worker,
                        platform=platform,
                        languages=languages,
                        domain=domain,
                        labels=labels,
                        num_samples=num_samples,          # Pass total
                        models_dict=models_dict,
                        temperature=temperature,
                        output_file=output_file,
                        balance_labels=balance_labels,
                        worker_id=worker_id,
                        total_workers=num_workers * len(platforms),  # Total actual workers
                        total_combinations=total_combinations,       # NEW
                    )
                    futures[future] = worker_id
                    worker_id += 1

            for future in as_completed(futures):
                try:
                    status_msg, temp_files = future.result()
                    results.append(status_msg)
                    all_temp_files.extend(temp_files)
                except Exception as e:
                    results.append(f"✗ Worker error: {str(e)}")

    except Exception as e:
        return f"✗ Fatal error: {str(e)}"

    # Merge
    merge_success, merge_msg = merge_tsv_files(all_temp_files, output_file, models_dict, platforms)
    results.append(merge_msg)

    summary = "\n".join(results)
    status_icon = "✓" if merge_success else "✗"
    
    final_count = 0
    try:
        if os.path.exists(output_file):
            final_count = len(pd.read_csv(output_file, sep='\t'))
    except:
        pass

    return (f"{status_icon} Dataset creation completed!\n"
            f"Total rows: {final_count} (target: {num_samples})\n\n"
            f"{summary}\n\nOutput: {output_file}")

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


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="GenSet Dataset Generator") as demo:
        gr.Markdown("# GenSet Dataset Generator")
        gr.Markdown("Generate synthetic datasets using multiple platforms, languages, and parallel workers")
        
        with gr.Row():
            # Left column: Configuration
            with gr.Column(scale=1):
                gr.Markdown("### Platform & Language")
                platforms = gr.CheckboxGroup(
                    choices=["mistral", "ollama"],
                    value=["mistral"],
                    label="Platforms",
                    info="Select one or more platforms"
                )
                languages = gr.CheckboxGroup(
                    choices=LANGUAGE_CHOICES,
                    value=["english"],
                    label="Languages",
                    info="Select one or more languages"
                )
                
                gr.Markdown("### Models (per Platform)")
                models_display = gr.JSON(label="Available Models")
                
                # Update models display when platforms change
                platforms.change(
                    fn=get_models_for_platforms,
                    inputs=[platforms],
                    outputs=[models_display]
                )
                
                gr.Markdown("### Generation Settings")
                domain = gr.Textbox(
                    value="general customer reviews",
                    label="Domain",
                    info="Domain for generation (e.g., product reviews, social media)"
                )
                labels = gr.Textbox(
                    value="positive,negative",
                    label="Labels",
                    info="Comma-separated classification labels"
                )
                temperature = gr.Slider(
                    0.0, 1.0,
                    value=0.7,
                    step=0.05,
                    label="Temperature",
                    info="Controls randomness in generation (0=deterministic, 1=random)"
                )
            
            # Middle column: Sample generation
            with gr.Column(scale=1):
                gr.Markdown("### Generate Sample")
                with gr.Group():
                    sample_output = gr.Textbox(
                        label="Generated Text",
                        lines=6,
                        interactive=False,
                        show_copy_button=True
                    )
                    sample_label = gr.Textbox(
                        label="Predicted Label",
                        interactive=False,
                        show_copy_button=True
                    )
                    sample_status = gr.Textbox(
                        label="Status",
                        interactive=False
                    )
                    generate_sample_btn = gr.Button("🎲 Generate Sample", variant="primary", size="lg")
                
                gr.Markdown("### Dataset Creation")
                num_samples = gr.Slider(
                    1, 500,
                    value=10,
                    step=1,
                    label="Number of Samples",
                    info="Samples per platform-language combination"
                )
                num_workers = gr.Slider(
                    1, 16,
                    value=2,
                    step=1,
                    label="Number of Workers",
                    info="Parallel workers for faster generation"
                )
                balance_labels = gr.Checkbox(
                    value=False,
                    label="⚖️ Balance Labels",
                    info="Ensure equal distribution across labels"
                )
                
            # Right column: Dataset creation
            with gr.Column(scale=1):
                gr.Markdown("### Output")
                output_file = gr.Textbox(
                    value=str(Config.normalize_output_path(Config.DEFAULT_OUTPUT_FILE)),
                    label="Output File",
                    info="Path for the generated dataset (TSV format)"
                )
                
                with gr.Group():
                    create_dataset_btn = gr.Button("📊 Create Dataset", variant="primary", size="lg")
                    dataset_status = gr.Textbox(
                        label="Creation Status",
                        interactive=False,
                        lines=10,
                        show_copy_button=True
                    )
                
                gr.Markdown("### Download")
                with gr.Group():
                    download_btn = gr.Button("⬇️ Download Dataset", variant="secondary", size="lg")
                    download_file = gr.File(
                        label="Download File",
                        interactive=False
                    )
                    download_status = gr.Textbox(
                        label="Download Status",
                        interactive=False
                    )
        
        # Event handlers
        generate_sample_btn.click(
            fn=create_sample,
            inputs=[platforms, languages, domain, labels, models_display, temperature],
            outputs=[sample_output, sample_label, sample_status],
        )
        
        create_dataset_btn.click(
            fn=create_dataset,
            inputs=[
                platforms,
                languages,
                domain,
                labels,
                num_samples,
                models_display,
                temperature,
                output_file,
                balance_labels,
                num_workers
            ],
            outputs=[dataset_status],
        )
        
        download_btn.click(
            fn=get_download_file,
            inputs=[output_file],
            outputs=[download_file],
        ).then(
            fn=lambda f: "✓ File ready for download!" if f else "✗ File not found. Please create a dataset first.",
            inputs=[download_file],
            outputs=[download_status]
        )
        
        # Initialize models on load
        demo.load(
            fn=get_models_for_platforms,
            inputs=[platforms],
            outputs=[models_display]
        )
    
    return demo


def main(share: bool = False, debug: bool = False) -> None:
    demo = build_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=share, debug=debug)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GenSet Gradio Web Interface"
    )
    
    parser.add_argument(
        "--share", 
        action="store_true",
        help="Enable public sharing link (Gradio share)"
    )

    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug mode with verbose logging"
    )

    args = parser.parse_args()
    main(share=args.share, debug=args.debug)
