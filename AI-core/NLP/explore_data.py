"""Loads the dataset and prints basic info. 
Run from repo root with venv activated:
ex: python3 Ad-block-with-XR-glasses/AI-core/NLP/explore_data.py
"""

from pathlib import Path
from datasets import load_dataset

NLP_DIR = Path(__file__).resolve().parent
TRAIN_FILE = NLP_DIR / "dataset" / "test_large.jsonl"

print(TRAIN_FILE)
print("exists:", TRAIN_FILE.exists())

dataset = load_dataset("json", data_files=str(TRAIN_FILE), split="train")

print(f"Number of samples: {len(dataset)}")
print(f"Columns: {dataset.column_names}")
print()
print("First 3 samples:")
for i in range(3):
    print(dataset[i])

from collections import Counter

label_counts = Counter(dataset["label"])
total = len(dataset)

print()
print("Class distribution:")
for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
    percentage = (count / total) * 100
    bar = "#" * int(percentage // 2)  # Scale bar length
    print(f"  {label:20s} {count:5d} ({percentage:5.2f}%) {bar}")


