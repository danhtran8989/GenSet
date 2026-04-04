import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from GenSet import Config, DatasetGenerator, LANGUAGE_CHOICES, LANGUAGE_DISPLAY
from rich.console import Console

console = Console()


def parse_labels(labels: str) -> list:
    parsed = [label.strip() for label in labels.split(",") if label.strip()]
    return parsed or ["positive", "negative"]


def choose_language() -> str:
    console.print("\nChoose language:")
    for lang in LANGUAGE_CHOICES:
        console.print(f"  - {LANGUAGE_DISPLAY[lang]}")
    language = input("Language (english/vietnamese, default english): ").strip().lower()
    return language if language in LANGUAGE_CHOICES else "english"


def choose_platform() -> str:
    platform = input("Platform (mistral/ollama, default mistral): ").strip().lower()
    return platform if platform in ["mistral", "ollama"] else "mistral"


def prompt_interactive() -> argparse.Namespace:
    language = choose_language()
    platform = choose_platform()
    domain = input("Domain (default: general customer reviews): ").strip() or "general customer reviews"
    labels = parse_labels(input("Labels (comma separated, default: positive,negative): ").strip() or "")
    num_samples = input("Number of samples to generate (default 100): ").strip() or "100"
    output_file = input("Output file (default: dataset.tsv): ").strip() or Config.DEFAULT_OUTPUT_FILE
    model = input("Model (press Enter for default): ").strip() or None
    temperature = input("Temperature (0.0-1.0, default 0.7): ").strip() or "0.7"
    multilingual = input("Generate multilingual (English + Vietnamese)? (y/n, default n): ").strip().lower() == "y"
    balance_labels = input("Balance labels (equal distribution)? (y/n, default n): ").strip().lower() == "y"

    try:
        num_samples = int(num_samples)
    except ValueError:
        num_samples = 100

    try:
        temperature = float(temperature)
    except ValueError:
        temperature = 0.7

    return argparse.Namespace(
        platform=platform,
        language=language,
        domain=domain,
        labels=labels,
        num_samples=num_samples,
        output=output_file,
        model=model,
        temperature=temperature,
        delay=0.8,
        multilingual=multilingual,
        balance_labels=balance_labels,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GenSet dataset generator CLI")
    parser.add_argument("--platform", choices=["mistral", "ollama"], help="API platform to use")
    parser.add_argument("--language", choices=LANGUAGE_CHOICES, help="Output language")
    parser.add_argument("--domain", help="Domain for generated examples")
    parser.add_argument("--labels", help="Comma-separated labels")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples to generate")
    parser.add_argument("--output", default=Config.DEFAULT_OUTPUT_FILE, help="Output dataset file path")
    parser.add_argument("--model", help="Model name to use")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation")
    parser.add_argument("--delay", type=float, default=0.8, help="Delay between API calls")
    parser.add_argument("--multilingual", action="store_true", help="Generate both English and Vietnamese text")
    parser.add_argument("--balance-labels", action="store_true", help="Balance labels with equal distribution")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    args = parser.parse_args()

    if args.interactive or (args.platform is None and args.language is None and args.domain is None and args.labels is None and args.model is None):
        return prompt_interactive()

    if args.labels:
        args.labels = parse_labels(args.labels)
    else:
        args.labels = ["positive", "negative"]

    args.language = args.language or "english"
    args.platform = args.platform or "mistral"
    args.domain = args.domain or "general customer reviews"
    return args


def main() -> None:
    Config.print_config()
    args = parse_args()

    generator = DatasetGenerator()
    generator.create_dataset(
        num_samples=args.num_samples,
        platform=args.platform,
        language=args.language,
        labels=args.labels,
        domain=args.domain,
        model=args.model,
        temperature=args.temperature,
        output_file=args.output,
        delay=args.delay,
        multilingual=args.multilingual,
        balance_labels=args.balance_labels,
    )


if __name__ == "__main__":
    main()
