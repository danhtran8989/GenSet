# GenSet - Intelligent Dataset Generator

GenSet is a flexible Python toolkit for generating synthetic datasets using language models (Mistral or Ollama). It provides both CLI and web UI interfaces to generate classification datasets in multiple languages with customizable domains and labels.

## Features

- **Multi-platform support**: Mistral AI and Ollama
- **Multi-language**: English and Vietnamese prompts
- **Multilingual output**: Generate both English and Vietnamese text in a single dataset
- **Label balancing**: Automatic equal distribution across all labels
- **Interactive CLI**: Full terminal interface with argument support
- **Gradio Web UI**: User-friendly web interface
- **Customizable domains**: Define generation domains (e.g., product reviews, social media)
- **Flexible labels**: Support any classification labels
- **Robust JSON parsing**: Multiple fallback strategies for parsing LLM responses
- **Automatic output management**: Creates output directories and manages TSV/CSV files

## Installation

### Prerequisites

- Python 3.8+
- pip or conda

### Setup

1. Clone or download the repository:
```bash
cd GenSet-new
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure API keys in `.env`:
```env
# Mistral AI Keys (numbered incrementally)
MISTRAL_API_KEY_1=your_key_here
MISTRAL_API_KEY_2=another_key_here

# Ollama API Keys (if using remote Ollama)
OLLAMA_API_KEY_1=your_key_here
```

## Project Structure

```
GenSet-new/
├── src/
│   └── GenSet/
│       ├── __init__.py              # Package exports
│       ├── config.py                # Configuration and API management
│       ├── dataset_generator.py    # Core generation logic
│       └── prompts.py              # Language-specific prompts
├── apps/
│   ├── cli/
│   │   ├── __init__.py
│   │   ├── __main__.py
│   │   └── main.py                 # CLI interface
│   └── gradio/
│       ├── __init__.py
│       ├── __main__.py
│       └── app.py                  # Web UI interface
├── output/                          # Generated datasets directory
├── main.py                         # Root entrypoint (wrapper)
├── config.py                       # Root compatibility wrapper
├── prompts.py                      # Root compatibility wrapper
├── dataset_generator.py            # Root compatibility wrapper
├── requirements.txt
├── samples.csv                     # Example dataset
├── samples.tsv                     # Example dataset
└── README.md
```

## Usage

### CLI Interface

#### Interactive Mode

Run the CLI with no arguments for interactive prompts:

```bash
python main.py
```

Or explicitly:

```bash
python -m apps.cli.main
```

You will be prompted for:
- Language (English/Vietnamese)
- Platform (Mistral/Ollama)
- Domain (e.g., "product feedback")
- Labels (comma-separated, e.g., "positive,negative,neutral")
- Number of samples
- Output file path
- Model name (optional)
- Temperature (0.0-1.0)
- Multilingual mode (English + Vietnamese)
- Balance labels (equal distribution)

#### Argument Mode

Specify all parameters via command-line arguments:

```bash
python apps/cli/main.py \
  --platform mistral \
  --language english \
  --domain "restaurant reviews" \
  --labels "positive,negative,neutral" \
  --num-samples 50 \
  --output output/reviews.tsv \
  --temperature 0.7 \
  --multilingual \
  --balance-labels
```

##### Available Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--platform` | mistral\|ollama | mistral | API platform to use |
| `--language` | english\|vietnamese | english | Output language |
| `--domain` | string | general customer reviews | Domain for examples |
| `--labels` | string | positive,negative | Comma-separated labels |
| `--num-samples` | int | 100 | Number of samples to generate |
| `--output` | path | dataset.tsv | Output file path |
| `--model` | string | (default for platform) | Model name |
| `--temperature` | float | 0.7 | Generation temperature (0.0-1.0) |
| `--delay` | float | 0.8 | Delay between API calls (seconds) |
| `--multilingual` | flag | false | Generate English + Vietnamese text |
| `--balance-labels` | flag | false | Balance labels with equal distribution |
| `--interactive` | flag | false | Force interactive mode |

### Web UI (Gradio)

Launch the web interface:

```bash
python -m apps.gradio.app
```

Or:

```bash
python apps/gradio/app.py
```

The interface opens at `http://localhost:7860` with:
- Platform and language selection
- Domain and labels configuration
- Sample generation preview
- Full dataset generation
- Output file path specification
- Multilingual mode toggle
- Label balancing toggle

## API Reference

### DatasetGenerator

The core class for generating datasets.

#### Initialization

```python
from src.GenSet import DatasetGenerator

generator = DatasetGenerator()
```

#### Methods

##### `generate_example()`

Generate a single example.

```python
example = generator.generate_example(
    platform="mistral",              # or "ollama"
    language="english",              # or "vietnamese"
    labels=["positive", "negative"],
    domain="product feedback",
    min_sentences=1,
    max_sentences=3,
    model=None,                      # Uses default if None
    temperature=0.7
)

print(example)
# Output: {"text": "Great product!", "label": "positive"}
```

##### `generate_multilingual_example()`

Generate both English and Vietnamese text for the same label.

```python
example = generator.generate_multilingual_example(
    platform="mistral",
    labels=["positive", "negative"],
    domain="restaurant reviews",
    min_sentences=1,
    max_sentences=3,
    model=None,
    temperature=0.7
)

print(example)
# Output: {
#     "english_text": "Great food and service!",
#     "vietnamese_text": "Thức ăn và dịch vụ tuyệt vời!",
#     "label": "positive"
# }
```

##### `create_dataset()`

Generate a complete dataset.

```python
generator.create_dataset(
    num_samples=300,
    platform="mistral",
    language="english",
    labels=["positive", "negative", "neutral"],
    domain="product feedback",
    min_sentences=1,
    max_sentences=3,
    model=None,
    temperature=0.7,
    output_file="output/dataset.tsv",
    delay=0.8,
    multilingual=True,       # Generate English + Vietnamese columns
    balance_labels=True,     # Ensure equal distribution (100 per label)
)
```

**Parameters**:
- `num_samples`: Total samples to generate
- `platform`: "mistral" or "ollama"
- `language`: Primary language (used when multilingual=False)
- `labels`: List of classification labels
- `domain`: Domain for generation context
- `min_sentences`, `max_sentences`: Text length range
- `model`: Specific model to use (None = default)
- `temperature`: Generation temperature (0.0-1.0)
- `output_file`: Path to save dataset
- `delay`: Seconds between API calls
- `multilingual`: Boolean - Generate English + Vietnamese text
- `balance_labels`: Boolean - Ensure equal label distribution

**Returns**: Full path to the created dataset file.

##### Output Behavior with Flags

| multilingual | balance_labels | Output Format | Distribution |
|-------------|----------------|--------------|--------------|
| False | False | Single language | Random labels |
| False | True | Single language | Equal per label |
| True | False | Bilingual | Random labels |
| True | True | Bilingual | Equal per label |

### Config

Configuration and API key management.

```python
from src.GenSet import Config

# Print current configuration
Config.print_config()

# Normalize and create output paths
output_path = Config.normalize_output_path("output/dataset.tsv")
Config.ensure_output_dir(output_path)
```

**Class Attributes**:
- `MISTRAL_API_KEYS`: List of Mistral API keys from `.env`
- `OLLAMA_API_KEYS`: List of Ollama API keys from `.env`
- `DEFAULT_MISTRAL_MODEL`: "mistral-large-latest"
- `DEFAULT_OLLAMA_MODEL`: "llama3.2"
- `DEFAULT_OUTPUT_DIR`: "output"
- `DEFAULT_OUTPUT_FILE`: "dataset.tsv"

### Prompts

Language-specific prompt management.

```python
from src.GenSet import get_system_prompt, get_user_prompt, LANGUAGE_CHOICES, LANGUAGE_DISPLAY

# Get system prompt
system_prompt = get_system_prompt("english")

# Get user prompt with parameters
user_prompt = get_user_prompt(
    language="vietnamese",
    labels=["tích cực", "tiêu cực"],
    domain="đánh giá nhà hàng",
    min_sentences=2,
    max_sentences=4
)

# List available languages
print(LANGUAGE_CHOICES)        # ["english", "vietnamese"]
print(LANGUAGE_DISPLAY)        # Display names with emojis
```

## Examples

### Example 1: Generate Product Reviews (English)

```bash
python apps/cli/main.py \
  --platform mistral \
  --language english \
  --domain "product feedback" \
  --labels "positive,negative,neutral" \
  --num-samples 100 \
  --output output/product_reviews.tsv
```

### Example 2: Multilingual Dataset with Balanced Labels

Generate balanced datasets with both English and Vietnamese text:

```bash
python apps/cli/main.py \
  --platform mistral \
  --domain "restaurant reviews" \
  --labels "positive,negative,neutral" \
  --num-samples 300 \
  --output output/multilingual_reviews.tsv \
  --multilingual \
  --balance-labels
```

This generates 100 samples per label (300 total) with columns: `English_Text`, `Vietnamese_Text`, `Label`

### Example 3: Generate Vietnamese Social Media Posts

```bash
python apps/cli/main.py \
  --platform ollama \
  --language vietnamese \
  --domain "mạng xã hội" \
  --labels "tích cực,tiêu cực" \
  --num-samples 50 \
  --model llama3.2 \
  --output output/vietnamese_posts.tsv
```

### Example 4: Balanced Labels Only (Single Language)

Generate dataset with equal distribution of labels:

```bash
python apps/cli/main.py \
  --platform mistral \
  --language english \
  --domain "product feedback" \
  --labels "positive,negative,neutral" \
  --num-samples 300 \
  --balance-labels \
  --output output/balanced_reviews.tsv
```

This creates 100 samples for each of the 3 labels.

### Example 5: Programmatic Usage
```python
from src.GenSet import DatasetGenerator, Config

# Initialize
generator = DatasetGenerator()
Config.print_config()

# Generate dataset
output_path = generator.create_dataset(
    num_samples=200,
    platform="mistral",
    language="english",
    labels=["sentiment_positive", "sentiment_negative"],
    domain="email feedback",
    temperature=0.8,
    output_file="output/emails.tsv"
)

print(f"Dataset saved to: {output_path}")
```

### Example 5: Interactive Mode

```bash
python main.py
```

Then follow the terminal prompts to configure your dataset generation.

### Example 6: Multilingual & Balanced Programmatic Usage

```python
from src.GenSet import DatasetGenerator, Config

# Initialize
generator = DatasetGenerator()
Config.print_config()

# Generate multilingual balanced dataset
output_path = generator.create_dataset(
    num_samples=300,
    platform="mistral",
    labels=["positive", "negative", "neutral"],
    domain="email feedback",
    temperature=0.8,
    output_file="output/multilingual_emails.tsv",
    multilingual=True,      # Generate English + Vietnamese
    balance_labels=True,    # 100 samples per label
)

print(f"Dataset saved to: {output_path}")
```

## Output Format

### Single Language Format
Generated datasets are stored in TSV (Tab-Separated Values) format with two columns:

```
text                              label
Great product quality!            positive
The service was terrible.         negative
It's okay, nothing special.       neutral
```

### Multilingual Format (with `--multilingual` flag)
When using multilingual mode, datasets include English and Vietnamese columns:

```
English_Text                      Vietnamese_Text                   Label
Great product quality!            Chất lượng sản phẩm tuyệt vời!    positive
The service was terrible.         Dịch vụ thực sự tệ.               negative
It's okay, nothing special.       Được thôi, không gì đặc biệt.    neutral
```

Specify `.csv` extension for CSV format instead:
```bash
python apps/cli/main.py --output output/dataset.csv
```

## Supported Models

### Mistral

- `mistral-large-latest` (default)
- `mistral-medium-latest`
- `mistral-small-latest`

### Ollama

- `llama3.2` (default)
- `mistral-nemo`
- `qwen2.5`
- `neural-chat`
- Any locally available Ollama model

## Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
# Mistral API Keys (can have multiple, numbered incrementally)
MISTRAL_API_KEY_1=sk-xxxxx
MISTRAL_API_KEY_2=sk-yyyyy
MISTRAL_API_KEY_3=sk-zzzzz

# Ollama Configuration (optional for remote instances)
OLLAMA_API_KEY_1=token_xxxxx

# Note: OLLAMA_BASE_URL defaults to https://ollama.com
# For local instances, modify src/GenSet/config.py
```

### Supported Domains

**English**:
- general customer reviews
- product feedback
- social media
- news articles
- emails
- tweets
- restaurant reviews
- movie reviews

**Vietnamese**:
- đánh giá khách hàng chung
- phản hồi sản phẩm
- mạng xã hội
- bài báo tin tức
- emails
- tweets
- đánh giá nhà hàng
- đánh giá phim ảnh

## Troubleshooting

### API Key Issues

**Error**: "No Mistral API keys loaded in .env!"

**Solution**: Ensure `.env` file has `MISTRAL_API_KEY_1` set in the root directory.

### JSON Parsing Errors

**Issue**: Some samples show "unknown" label

**Solution**: This occurs when the LLM doesn't return valid JSON. Increases in `--temperature` or different models can help. Samples are still generated but labeled as "unknown".

### Output Directory Issues

**Error**: Permission denied when creating output files

**Solution**: Ensure the `output/` directory exists and is writable:
```bash
mkdir output
chmod 755 output  # On Unix/Linux/macOS
```

### Rate Limiting

**Issue**: API errors after many requests

**Solution**: Increase the `--delay` parameter (default 0.8 seconds):
```bash
python apps/cli/main.py --delay 2.0 --num-samples 100
```

## Dependencies

- `python-dotenv` - Environment configuration
- `requests` - HTTP client for API calls
- `tqdm` - Progress bars
- `rich` - Terminal formatting
- `pandas` - Data processing
- `gradio` - Web UI framework

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the examples
3. Check API documentation for your selected platform (Mistral/Ollama)

## Changelog

### v1.0.0 (2026-04-04)

- Initial release with refactored package structure
- CLI interface with argument and interactive modes
- Gradio web UI
- Multi-language support (English, Vietnamese)
- Dual-platform support (Mistral AI, Ollama)
- Robust JSON parsing with fallback strategies
- Output management with directory creation
