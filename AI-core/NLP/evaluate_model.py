"""
Evaluates the fine-tuned text classifier and prints accuracy, a per-class
classification report, confusion matrix, and a safety check.

=== Provenance-split evaluation ===
Each example in the test set has a "source" field ("real" or "synthetic"),
added by create_dataset_split.py. This script reports metrics both for the
full test set AND separately for real vs synthetic examples, so we can see
how well the model generalises to real-world text vs synthetic training
distribution.

=== Test files ===
  test_clean.jsonl          — Clean text (for Model A baseline evaluation)
  test_augmented_v2.jsonl   — Clean + OCR-noisy copies (for Model B production eval)

Change TEST_FILE below to switch between them.
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

# "clean" → no OCR noise, "augmented" → OCR noise
MODE = "augmented"

# "mixed" → real + synthetic, "real_only" → ablation baseline
DATASET = "mixed"

_TEST_FILES = {
    ("mixed", "augmented"): "test_augmented_v2.jsonl",
    ("mixed", "clean"): "test_clean.jsonl",
    ("real_only", "augmented"): "test_real_only_augmented.jsonl",
    ("real_only", "clean"): "test_real_only.jsonl",
}

TEST_FILE = NLP_DIR / "dataset" / _TEST_FILES[(DATASET, MODE)]
RUN_TAG = f"{DATASET}_{MODE}"
SAVED_MODEL_DIR = NLP_DIR / "saved_model" / RUN_TAG
RESULTS_DIR = NLP_DIR / "results" / RUN_TAG


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


def evaluate_subset(true_ids, predicted_ids, subset_name):
    """Print classification report and safety check for a subset of examples."""
    if len(true_ids) == 0:
        print(f"  No {subset_name} examples found, skipping.")
        return

    accuracy = accuracy_score(true_ids, predicted_ids)
    print(f"\n{'=' * 50}")
    print(f"  {subset_name} ({len(true_ids)} examples) — Accuracy: {accuracy:.4f}")
    print(f"{'=' * 50}")
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

    # Safety check for this subset
    samh_id = label2id["socially beneficial"]
    reklam_id = label2id["ad"]
    cm = confusion_matrix(true_ids, predicted_ids, labels=LABEL_IDS)
    samh_total = cm[samh_id].sum()
    if samh_total > 0:
        samh_as_reklam = cm[samh_id][reklam_id]
        print(f"  Safety: {samh_as_reklam}/{samh_total} socially beneficial → reklam "
              f"({samh_as_reklam / samh_total * 100:.1f}%)")


def main():
    if not SAVED_MODEL_DIR.exists():
        raise FileNotFoundError(f"Could not find saved model directory: {SAVED_MODEL_DIR}")

    print(f"Loading test dataset from {TEST_FILE}")
    dataset = load_dataset("json", data_files=str(TEST_FILE), split="train")

    # Save source tags before encoding labels (source column may be dropped by map)
    sources = dataset["source"] if "source" in dataset.column_names else None

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

    # ── Full test set metrics ─────────────────────────────────────────
    accuracy = accuracy_score(true_ids, predicted_ids)
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    print()
    print("Classification report (all examples):")
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

    # ── Confusion matrix (full set) ──────────────────────────────────
    cm = confusion_matrix(true_ids, predicted_ids, labels=LABEL_IDS)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plots_dir = NLP_DIR / "plots" / MODE
    plots_dir.mkdir(parents=True, exist_ok=True)
    cm_path = plots_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=120)
    plt.close()
    print(f"Saved confusion matrix to {cm_path}")

    # ── Safety check (full set) ───────────────────────────────────────
    samh_id = label2id["socially beneficial"]
    reklam_id = label2id["ad"]
    samh_as_reklam = cm[samh_id][reklam_id]
    samh_total = cm[samh_id].sum()
    if samh_total > 0:
        print(f"Safety: {samh_as_reklam}/{samh_total} socially beneficial classified as reklam "
              f"({samh_as_reklam / samh_total * 100:.1f}%)")

    # ── Provenance-split evaluation ───────────────────────────────────
    # Report metrics separately for real vs synthetic test examples.
    # This shows how well the model generalises to real-world text
    # compared to the synthetic distribution it was partially trained on.
    if sources is not None:
        sources_arr = np.array(sources)
        for source_type in ["real", "synthetic"]:
            mask = sources_arr == source_type
            if mask.sum() > 0:
                evaluate_subset(
                    true_ids[mask],
                    predicted_ids[mask],
                    f"{source_type.upper()} examples",
                )


if __name__ == "__main__":
    main()
