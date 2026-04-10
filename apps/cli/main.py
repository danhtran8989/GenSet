import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TaskProgressColumn
from tqdm import tqdm

# Add src to Python path
ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from GenSet.config import Config
from GenSet import DatasetGenerator
from GenSet.prompts import LANGUAGE_CHOICES, LANGUAGE_DISPLAY

console = Console()


def load_labels_from_file(labels_path: str) -> List[str]:
    """Load labels from a text file (one label per line)."""
    file_path = Path(labels_path)
    
    if not file_path.exists():
        console.print(f"[yellow]⚠️  Labels file '{file_path}' not found. Using defaults.[/yellow]")
        return ["positive", "negative"]
    
    try:
        labels = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                label = line.strip()
                if label and not label.startswith("#"):
                    labels.append(label)
        
        if not labels:
            console.print(f"[yellow]⚠️  No valid labels in {file_path}. Using defaults.[/yellow]")
            return ["positive", "negative"]

        console.print(f"[green]✅ Loaded {len(labels)} labels from {file_path}[/green]")
        return labels

    except Exception as e:
        console.print(f"[red]❌ Error reading labels file: {e}[/red]")
        return ["positive", "negative"]


def load_domain_from_file(domain_path: str) -> str:
    """Load domain description from file (supports multi-line)."""
    file_path = Path(domain_path)
    
    if not file_path.exists():
        console.print(f"[yellow]⚠️  Domain file '{file_path}' not found. Using default.[/yellow]")
        return "general customer reviews"
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            domain = f.read().strip()
        
        if domain:
            console.print(f"[green]✅ Loaded domain description ({len(domain)} chars)[/green]")
            return domain
        else:
            console.print(f"[yellow]⚠️  Domain file is empty. Using default.[/yellow]")
            return "general customer reviews"

    except Exception as e:
        console.print(f"[red]❌ Error reading domain file: {e}[/red]")
        return "general customer reviews"


def parse_labels(labels_str: str) -> List[str]:
    """Parse comma-separated labels from CLI."""
    if not labels_str:
        return ["positive", "negative"]
    parsed = [label.strip() for label in labels_str.split(",") if label.strip()]
    return parsed or ["positive", "negative"]


def prompt_interactive() -> argparse.Namespace:
    """Interactive mode."""
    console.print("\n[bold cyan]=== Interactive Dataset Generation Mode ===[/bold cyan]")

    language = input(f"→ Language (english/vietnamese) [default: english]: ").strip().lower()
    language = language if language in LANGUAGE_CHOICES else "english"

    platform = input(f"→ Platform (mistral/ollama/openai/gemini) [default: ollama]: ").strip().lower()
    platform = platform if platform in ["mistral", "ollama", "openai", "gemini"] else "ollama"

    domain = input(f"→ Domain [default: general customer reviews]: ").strip() or "general customer reviews"
    labels_input = input(f"→ Labels (comma separated) [default: positive,negative]: ").strip()
    labels = parse_labels(labels_input)

    num_samples = input(f"→ Number of samples [default: 100]: ").strip() or "100"
    output_file = input(f"→ Output file [default: {Config.get_default_output_file()}]: ").strip() or None
    model = input(f"→ Model (Enter for default): ").strip() or None
    temperature = input(f"→ Temperature [default: 0.7]: ").strip() or "0.7"
    multilingual = input(f"→ Multilingual (EN+VI)? (y/n) [default: n]: ").strip().lower() == "y"
    balance_labels = input(f"→ Balance labels? (y/n) [default: n]: ").strip().lower() == "y"

    try:
        num_samples = int(num_samples)
        temperature = float(temperature)
    except ValueError:
        num_samples, temperature = 100, 0.7

    return argparse.Namespace(
        platform=platform,
        language=language,
        domain=domain,
        labels=labels,
        num_samples=num_samples,
        output=output_file or Config.get_default_output_file(),
        model=model,
        temperature=temperature,
        delay=0.8,
        multilingual=multilingual,
        balance_labels=balance_labels,
        labels_file=None,
        domain_file=None,
    )


def print_args(args: argparse.Namespace) -> None:
    """Display configuration summary."""
    console.print("\n[bold magenta]" + "═" * 80 + "[/bold magenta]")
    console.print("[bold magenta]📋 DATASET GENERATOR CONFIGURATION[/bold magenta]")
    console.print("[bold magenta]" + "═" * 80 + "[/bold magenta]")

    console.print(f"{'Platform':<18}: [cyan]{args.platform.upper()}[/cyan]")
    console.print(f"{'Language':<18}: [cyan]{args.language.capitalize()}[/cyan]")
    console.print(f"{'Domain':<18}: [cyan]{args.domain[:50]}{'...' if len(args.domain) > 50 else ''}[/cyan]")
    console.print(f"{'Labels':<18}: [cyan]{', '.join(args.labels)}[/cyan]")
    console.print(f"{'Samples':<18}: [cyan]{args.num_samples}[/cyan]")
    console.print(f"{'Output':<18}: [cyan]{args.output}[/cyan]")
    console.print(f"{'Model':<18}: [cyan]{args.model or 'Default'}[/cyan]")
    console.print(f"{'Temperature':<18}: [cyan]{args.temperature}[/cyan]")
    console.print(f"{'Multilingual':<18}: [cyan]{'Yes' if args.multilingual else 'No'}[/cyan]")
    console.print(f"{'Balance Labels':<18}: [cyan]{'Yes' if args.balance_labels else 'No'}[/cyan]")
    console.print("[bold magenta]" + "═" * 80 + "[/bold magenta]\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GenSet - Multilingual Dataset Generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--platform", choices=["mistral", "ollama", "openai", "gemini"], default="ollama")
    parser.add_argument("--language", choices=LANGUAGE_CHOICES)
    parser.add_argument("--domain", help="Domain description")
    parser.add_argument("--labels", help="Comma-separated labels")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--output", default=None, help="Output file path")
    parser.add_argument("--model")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--delay", type=float, default=0.8)
    parser.add_argument("--multilingual", action="store_true")
    parser.add_argument("--balance-labels", action="store_true")
    parser.add_argument("--interactive", action="store_true")

    # File inputs
    parser.add_argument("--labels-file", default="data/labels.txt")
    parser.add_argument("--domain-file", default="data/descriptions.txt")

    args = parser.parse_args()

    if args.interactive:
        return prompt_interactive()

    # Load from files if not provided via CLI
    if args.labels:
        args.labels = parse_labels(args.labels)
    else:
        args.labels = load_labels_from_file(args.labels_file)

    if not args.domain:
        args.domain = load_domain_from_file(args.domain_file)

    # Final defaults
    args.language = args.language or "english"
    args.output = args.output or Config.get_default_output_file()

    return args


def main() -> None:
    console.print("[bold green]🚀 Starting GenSet Dataset Generator[/bold green]\n")

    args = parse_args()
    print_args(args)

    try:
        generator = DatasetGenerator()

        console.print("[bold cyan]Generating samples...[/bold cyan]")

        # Use tqdm for clean progress bar
        with tqdm(total=args.num_samples, desc="Generating", unit="sample", 
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]") as pbar:

            # Override create_dataset to support progress bar
            # We'll call it in a loop for better control with tqdm
            writer = None
            success = 0

            # Create writer manually to integrate with tqdm
            from GenSet.dataset_writer import DatasetWriter
            writer = DatasetWriter(args.output, multilingual=args.multilingual)

            sample_queue = [None] * args.num_samples
            if args.balance_labels and args.labels:
                # Simple balanced queue
                samples_per_label = args.num_samples // len(args.labels)
                remainder = args.num_samples % len(args.labels)
                sample_queue = []
                for i, label in enumerate(args.labels):
                    count = samples_per_label + (1 if i < remainder else 0)
                    sample_queue.extend([label] * count)
                # Shuffle
                import random
                random.shuffle(sample_queue)
                sample_queue = sample_queue[:args.num_samples]

            for i, target_label in enumerate(sample_queue):
                try:
                    if args.multilingual:
                        example = generator.generate_multilingual_example(
                            platform=args.platform,
                            labels=args.labels,
                            domain=args.domain,
                            min_sentences=1,
                            max_sentences=3,
                            model=args.model,
                            temperature=args.temperature,
                            forced_label=target_label,
                        )
                    else:
                        example = generator.generate_example(
                            platform=args.platform,
                            language=args.language,
                            labels=args.labels,
                            domain=args.domain,
                            min_sentences=1,
                            max_sentences=3,
                            model=args.model,
                            temperature=args.temperature,
                            forced_label=target_label,
                        )

                    writer.write(example)
                    success += 1

                except Exception as e:
                    console.print(f"\n[red]Error at sample {i+1}: {e}[/red]")

                pbar.update(1)
                time.sleep(args.delay)

            # Final summary
            console.print(f"\n[bold green]✅ Done! {success}/{args.num_samples} samples generated[/bold green]")
            console.print(f"📁 Saved to: [cyan]{writer.output_file}[/cyan]")

    except Exception as e:
        console.print(f"\n[bold red]❌ Generation failed: {e}[/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    main()

    # python apps/cli/main.py \
    #     --labels-file "./data/labels.txt"  --domain-file ./data/descriptions.txt \
    #     --language vietnamese --num-samples 2 \
    #     --output "output/dataset.tsv" --model "gemma3:4b-cloud" \
    #     --platform ollama --balance-labels