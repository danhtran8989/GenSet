import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import gradio as gr
from GenSet import Config, DatasetGenerator, LANGUAGE_CHOICES, LANGUAGE_DISPLAY


def parse_labels(label_string: str) -> list:
    labels = [label.strip() for label in label_string.split(",") if label.strip()]
    return labels or ["positive", "negative"]


def create_sample(platform: str, language: str, domain: str, labels: str, model: str, temperature: float):
    generator = DatasetGenerator()
    example = generator.generate_example(
        platform=platform,
        language=language,
        labels=parse_labels(labels),
        domain=domain,
        model=model or None,
        temperature=temperature,
    )
    return example["text"], example["label"], "Sample generated successfully."


def create_dataset(
    platform: str,
    language: str,
    domain: str,
    labels: str,
    num_samples: int,
    model: str,
    temperature: float,
    output_file: str,
    multilingual: bool,
    balance_labels: bool,
):
    generator = DatasetGenerator()
    result_path = generator.create_dataset(
        num_samples=num_samples,
        platform=platform,
        language=language,
        labels=parse_labels(labels),
        domain=domain,
        model=model or None,
        temperature=temperature,
        output_file=output_file,
        delay=0.8,
        multilingual=multilingual,
        balance_labels=balance_labels,
    )
    return f"Dataset created: {result_path}"


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="GenSet Dataset Generator") as demo:
        gr.Markdown("# GenSet Dataset Generator")
        with gr.Row():
            with gr.Column():
                platform = gr.Radio(["mistral", "ollama"], value="mistral", label="Platform")
                language = gr.Radio(LANGUAGE_CHOICES, value="english", label="Language")
                domain = gr.Textbox(value="general customer reviews", label="Domain")
                labels = gr.Textbox(value="positive,negative", label="Labels")
                model = gr.Textbox(value="", label="Model (optional)")
                temperature = gr.Slider(0.0, 1.0, value=0.7, step=0.05, label="Temperature")
            with gr.Column():
                num_samples = gr.Slider(1, 200, value=10, step=1, label="Number of samples")
                output_file = gr.Textbox(value=str(Config.normalize_output_path(Config.DEFAULT_OUTPUT_FILE)), label="Output file")
                multilingual = gr.Checkbox(value=False, label="Multilingual (English + Vietnamese)")
                balance_labels = gr.Checkbox(value=False, label="Balance labels (equal distribution)")
                sample_output = gr.Textbox(label="Generated text", lines=8, interactive=False)
                sample_label = gr.Textbox(label="Predicted label", interactive=False)
                status = gr.Textbox(label="Status", interactive=False)

        with gr.Row():
            generate_sample_btn = gr.Button("Generate Sample")
            create_dataset_btn = gr.Button("Create Dataset")

        generate_sample_btn.click(
            fn=create_sample,
            inputs=[platform, language, domain, labels, model, temperature],
            outputs=[sample_output, sample_label, status],
        )
        create_dataset_btn.click(
            fn=create_dataset,
            inputs=[platform, language, domain, labels, num_samples, model, temperature, output_file, multilingual, balance_labels],
            outputs=[status],
        )

    return demo


def main() -> None:
    demo = build_interface()
    demo.launch(share=False)


if __name__ == "__main__":
    main()
