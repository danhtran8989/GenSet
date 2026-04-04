import sys
from pathlib import Path
import argparse
import os
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

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
    num_samples: int,
    models_dict: Dict,
    temperature: float,
    output_file: str,
    balance_labels: bool,
    worker_id: int,
    total_workers: int,
) -> str:
    """Worker function to create dataset portion in parallel."""
    try:
        generator = DatasetGenerator()
        model = models_dict.get(platform, [""])[0] if models_dict.get(platform) else None
        
        # Adjust samples per worker
        samples_per_worker = num_samples // total_workers
        if worker_id == 0:
            samples_per_worker += num_samples % total_workers  # Add remainder to first worker
        
        # For each language, generate samples
        for language in languages:
            result_path = generator.create_dataset(
                num_samples=samples_per_worker,
                platform=platform,
                language=language,
                labels=parse_labels(labels),
                domain=domain,
                model=model or None,
                temperature=temperature,
                output_file=output_file.replace(".tsv", f"_{platform}_{language}_w{worker_id}.tsv"),
                delay=0.8,
                multilingual=False,
                balance_labels=balance_labels,
            )
        
        return f"✓ Worker {worker_id} ({platform}): Created {samples_per_worker} samples"
    except Exception as e:
        return f"✗ Worker {worker_id} ({platform}): Error - {str(e)}"


def create_dataset(
    platforms: List[str],
    languages: List[str],
    domain: str,
    labels: str,
    num_samples: int,
    models_dict: Dict,
    temperature: float,
    output_file: str,
    balance_labels: bool,
    num_workers: int,
):
    """Create dataset using multiple workers and platforms."""
    if not platforms or not languages:
        return "⚠️ Please select at least one platform and language"
    
    if num_workers < 1:
        num_workers = 1
    
    results = []
    try:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {}
            worker_id = 0
            
            # Submit tasks for each platform with workers
            for platform in platforms:
                for w in range(num_workers):
                    future = executor.submit(
                        create_dataset_worker,
                        platform=platform,
                        languages=languages,
                        domain=domain,
                        labels=labels,
                        num_samples=num_samples,
                        models_dict=models_dict,
                        temperature=temperature,
                        output_file=output_file,
                        balance_labels=balance_labels,
                        worker_id=worker_id,
                        total_workers=num_workers,
                    )
                    futures[future] = worker_id
                    worker_id += 1
            
            # Collect results as they complete
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append(f"✗ Worker error: {str(e)}")
    
    except Exception as e:
        return f"✗ Fatal error: {str(e)}"
    
    summary = "\n".join(results)
    return f"Dataset creation completed!\n\n{summary}\nOutput: {output_file}"


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
