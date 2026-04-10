# src/GenSet/dataset_writer.py
import csv
import os
from typing import List, Dict

from .config import Config   # ← New import


class DatasetWriter:
    def __init__(self, output_file: str = None, multilingual: bool = False):
        self.output_file = self._normalize_output_path(output_file)
        self.multilingual = multilingual
        self.delimiter = "\t" if self.output_file.lower().endswith(".tsv") else ","

        self._init_file()

    def _normalize_output_path(self, output_file: str = None) -> str:
        """Normalize output path using new Config"""
        output_file = output_file or Config.get_default_output_file()

        # If no directory is specified, put it in default output dir
        if not os.path.dirname(output_file):
            output_file = os.path.join(Config.get_output_dir(), output_file)

        return os.path.abspath(output_file)

    def _init_file(self):
        """Create file with headers if it doesn't exist"""
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)   # ← Fixed here

        if not os.path.exists(self.output_file):
            headers = ["English_Text", "Vietnamese_Text", "Label"] if self.multilingual else ["text", "label"]
            with open(self.output_file, "w", newline="", encoding="utf-8") as f:
                csv.writer(f, delimiter=self.delimiter).writerow(headers)

    def write(self, example: Dict):
        """Append a single example to the dataset file"""
        with open(self.output_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=self.delimiter)
            if self.multilingual:
                writer.writerow([
                    example.get("english_text", ""),
                    example.get("vietnamese_text", ""),
                    example.get("label", "unknown")
                ])
            else:
                writer.writerow([
                    example.get("text", ""),
                    example.get("label", "unknown")
                ])