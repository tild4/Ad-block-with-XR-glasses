"""
Evaluates the fine-tuned text classifier on AI-core/NLP/dataset/test_original.jsonl
and prints accuracy plus a per-class classification report.
"""

from pathlib import Path

import numpy as np
from datasets import load_dataset
from label_schema import LABELS, LABEL_IDS, label2id
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
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

    # Confusion matrix saved as plot
    cm = confusion_matrix(true_ids, predicted_ids, labels=LABEL_IDS)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plots_dir = NLP_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plots_dir / "confusion_matrix.png", dpi=120)
    plt.close()
    print(f"Saved confusion matrix to {plots_dir / 'confusion_matrix.png'}")

    # Safety check: how often is samhällsnyttig misclassified as reklam?
    samh_id = label2id["samhällsnyttig"]
    reklam_id = label2id["reklam"]
    samh_as_reklam = cm[samh_id][reklam_id]
    samh_total = cm[samh_id].sum()
    print(f"Safety: {samh_as_reklam}/{samh_total} samhällsnyttig classified as reklam "
          f"({samh_as_reklam/samh_total*100:.1f}%)")


if __name__ == "__main__":
    main()
