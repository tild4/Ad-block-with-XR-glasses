"""
Evaluates the fine-tuned text classifier on AI-core/NLP/dataset/test_original.jsonl
and prints accuracy plus a per-class classification report.
"""

from pathlib import Path

import numpy as np
from datasets import load_dataset
from label_schema import LABELS, LABEL_IDS, label2id
from sklearn.metrics import accuracy_score, classification_report
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

NLP_DIR = Path(__file__).resolve().parent
TEST_FILE = NLP_DIR / "dataset" / "test_original_augmented.jsonl"
SAVED_MODEL_DIR = NLP_DIR / "saved_model"
RESULTS_DIR = NLP_DIR / "results"


def encode_label(example):
    """Convert the string label to its integer id."""
    example["label"] = label2id[example["label"]]
    return example


def tokenize(batch, tokenizer):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=64,
    )


def main():
    if not SAVED_MODEL_DIR.exists():
        raise FileNotFoundError(f"Could not find saved model directory: {SAVED_MODEL_DIR}")

    print(f"Loading test dataset from {TEST_FILE}")
    dataset = load_dataset("json", data_files=str(TEST_FILE), split="train")
    dataset = dataset.map(encode_label)
    print(f"  {len(dataset)} examples")

    print(f"Loading tokenizer and model from {SAVED_MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(SAVED_MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(SAVED_MODEL_DIR)

    dataset = dataset.map(
        lambda batch: tokenize(batch, tokenizer),
        batched=True,
    )

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(RESULTS_DIR / "eval"),
            per_device_eval_batch_size=64,
            report_to="none",
        ),
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    print("Running evaluation...")
    predictions = trainer.predict(dataset)
    predicted_ids = np.argmax(predictions.predictions, axis=1)
    true_ids = np.array(dataset["label"])

    accuracy = accuracy_score(true_ids, predicted_ids)
    print(f"Accuracy: {accuracy:.4f}")
    print()
    print("Classification report:")
    print(
        classification_report(
            true_ids,
            predicted_ids,
            labels=LABEL_IDS,
            target_names=LABELS,
            digits=4,
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()
