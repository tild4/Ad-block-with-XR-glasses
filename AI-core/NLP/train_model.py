"""
Fine-tunes KBLab/electra-small-swedish-cased-discriminator for Swedish
ad-classification (3 classes: "non-ad", "socially beneficial", "ad").

=== Two training modes ===

Model A — Baseline (USE_AUGMENTATION = False):
  Trains on clean text only (train_clean.jsonl).
  Evaluates on clean test data (test_clean.jsonl).
  Shows the model's base performance without OCR-noise robustness.

Model B — Production (USE_AUGMENTATION = True):
  Trains with on-the-fly OCR-noise augmentation: each epoch, every example
  has a probability AUGMENT_PROB of being corrupted by add_noise() from
  augment_ocr_noise.py. This gives the model fresh noise patterns every
  epoch instead of memorising fixed noisy copies.
  Evaluates on test_clean.jsonl (during training) and test_augmented_v2.jsonl
  (during final evaluation with evaluate_model.py).

=== Dataset origin ===
train_clean.jsonl and test_clean.jsonl are created by create_dataset_split.py,
which pools:
  - Real data from test_original.jsonl (372 manually collected examples)
  - Synthetic data from (Claude + GPT)generated_examples.jsonl (1994 examples)
and performs a stratified split with source-tagging ("real"/"synthetic").

Toggle USE_AUGMENTATION and AUGMENT_PROB below to switch between Model A and B.
Adjust hyperparameters (learning_rate, weight_decay, etc.) for tuning runs.
"""

import random
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from label_schema import LABELS, id2label, label2id
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from augment_ocr_noise import CHAR_NOISE_PROB, add_noise

# ── Paths ─────────────────────────────────────────────────────────────
NLP_DIR = Path(__file__).resolve().parent
TRAIN_FILE = NLP_DIR / "dataset" / "train_clean.jsonl"
# Model A: use "test_clean.jsonl"
# Model B: use "test_augmented_v2.jsonl" (clean + OCR-noisy copies)
TEST_FILE = NLP_DIR / "dataset" / "test_augmented_v2.jsonl"
RESULTS_DIR = NLP_DIR / "results"
SAVED_MODEL_DIR = NLP_DIR / "saved_model"
PLOTS_DIR = NLP_DIR / "plots"

# ── Model ─────────────────────────────────────────────────────────────
MODEL_NAME = "KBLab/electra-small-swedish-cased-discriminator"
POSITIVE_LABEL = "ad"
POSITIVE_LABEL_ID = label2id[POSITIVE_LABEL]

# ── Augmentation config ──────────────────────────────────────────────
# Set USE_AUGMENTATION = False for Model A (baseline),
#                        True  for Model B (production with OCR noise).
USE_AUGMENTATION = True
AUGMENT_PROB = 0.5  # probability of applying OCR noise to each example per epoch


# ── AugmentedDataset ─────────────────────────────────────────────────
class AugmentedDataset(torch.utils.data.Dataset):
    """
    Wraps a HuggingFace Dataset to apply random OCR noise on-the-fly.

    Instead of pre-generating a fixed set of noisy copies (which the model
    memorises after the first epoch), this dataset applies add_noise() with
    probability augment_prob each time an example is accessed. This means
    every epoch sees different noise patterns, increasing the effective
    training diversity without inflating the dataset size.

    The tokenizer is called inside __getitem__ so that the noisy text is
    tokenized fresh each time (we can't pre-tokenize because the text changes).
    """

    def __init__(self, hf_dataset, tokenizer, augment_prob=0.5):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.augment_prob = augment_prob

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = item["text"]

        # With probability augment_prob, corrupt the text with OCR noise
        if random.random() < self.augment_prob:
            text = add_noise(text)

        tokens = self.tokenizer(text, truncation=True, max_length=64)
        tokens["labels"] = item["label"]
        return tokens


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


def load_and_prepare(jsonl_path, tokenizer, label_name, do_tokenize=True):
    """Load a jsonl, encode labels to ids, optionally tokenize.

    When using AugmentedDataset, we skip tokenization here because
    the dataset tokenizes on-the-fly after applying noise.
    """
    print(f"Loading {label_name} dataset from {jsonl_path}")
    ds = load_dataset("json", data_files=str(jsonl_path), split="train")
    print(f"  {len(ds)} examples, columns: {ds.column_names}")
    ds = ds.map(encode_label)
    if do_tokenize:
        ds = ds.map(lambda batch: tokenize(batch, tokenizer), batched=True)
    return ds


def plot_loss_curves(log_history, output_path):
    """Read trainer.state.log_history and plot train + eval loss vs epoch."""
    train_points = [(e["epoch"], e["loss"]) for e in log_history if "loss" in e and "eval_loss" not in e]
    eval_points = [(e["epoch"], e["eval_loss"]) for e in log_history if "eval_loss" in e]

    plt.figure(figsize=(8, 5))
    if train_points:
        epochs, losses = zip(*train_points)
        plt.plot(epochs, losses, label="train loss", alpha=0.6)
    if eval_points:
        epochs, losses = zip(*eval_points)
        plt.plot(epochs, losses, label="eval loss", marker="o", linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and evaluation loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()
    print(f"Saved loss curves to {output_path}")


def compute_metrics(eval_pred):
    """Log metrics that reflect ad-detection quality, not just loss."""
    logits, labels = eval_pred
    predicted_ids = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predicted_ids)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels,
        predicted_ids,
        average="macro",
        zero_division=0,
    )
    reklam_precision, reklam_recall, reklam_f1, _ = precision_recall_fscore_support(
        labels,
        predicted_ids,
        labels=[POSITIVE_LABEL_ID],
        average=None,
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "reklam_precision": reklam_precision[0],
        "reklam_recall": reklam_recall[0],
        "reklam_f1": reklam_f1[0],
    }


def main():
    # ── Training configuration ────────────────────────────────────────
    args = TrainingArguments(
        output_dir=str(RESULTS_DIR),
        num_train_epochs=20,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.05,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        report_to="none",
        seed=42,
    )

    # ── Log run configuration ─────────────────────────────────────────
    # Printed at the start of every run so we can compare Minerva logs.
    print("=== Run Configuration ===")
    print(f"  model: {MODEL_NAME}")
    print(f"  use_augmentation: {USE_AUGMENTATION}")
    print(f"  augment_prob: {AUGMENT_PROB}")
    print(f"  char_noise_prob: {CHAR_NOISE_PROB}")
    print(f"  learning_rate: {args.learning_rate}")
    print(f"  weight_decay: {args.weight_decay}")
    print(f"  warmup_ratio: {args.warmup_ratio}")
    print(f"  num_train_epochs: {args.num_train_epochs}")
    print(f"  metric_for_best_model: {args.metric_for_best_model}")
    print(f"  train_file: {TRAIN_FILE}")
    print(f"  test_file: {TEST_FILE}")
    print("=========================")

    # ── Load tokenizer ────────────────────────────────────────────────
    print(f"\nLoading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # ── Prepare datasets ──────────────────────────────────────────────
    if USE_AUGMENTATION:
        # Model B: load without tokenizing — AugmentedDataset handles it
        train_hf = load_and_prepare(TRAIN_FILE, tokenizer, "train", do_tokenize=False)
        train_dataset = AugmentedDataset(train_hf, tokenizer, augment_prob=AUGMENT_PROB)
    else:
        # Model A: standard tokenized dataset
        train_dataset = load_and_prepare(TRAIN_FILE, tokenizer, "train")

    # Eval dataset is always pre-tokenized (no on-the-fly noise for eval)
    eval_dataset = load_and_prepare(TEST_FILE, tokenizer, "eval")

    # ── Load model ────────────────────────────────────────────────────
    print(f"Loading model: {MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        label2id=label2id,
        id2label=id2label,
    )

    # ── Data collator ─────────────────────────────────────────────────
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ── Print class distribution ──────────────────────────────────────
    if USE_AUGMENTATION:
        label_counts = Counter(train_hf["label"])
    else:
        label_counts = Counter(train_dataset["label"])
    total = sum(label_counts.values())
    print("Training class distribution:")
    for lid in sorted(label_counts):
        print(f"  {id2label[lid]}: {label_counts[lid]} ({label_counts[lid]/total*100:.1f}%)")

    # ── Initialize trainer ────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=4)],
    )

    # ── Train ─────────────────────────────────────────────────────────
    print("Starting training...")
    trainer.train()
    print("Training complete.")
    if trainer.state.best_model_checkpoint is not None:
        print(f"Best checkpoint: {trainer.state.best_model_checkpoint}")
        print(f"Best {args.metric_for_best_model}: {trainer.state.best_metric:.4f}")

    # ── Save model ────────────────────────────────────────────────────
    print(f"Saving final model to {SAVED_MODEL_DIR}")
    trainer.save_model(str(SAVED_MODEL_DIR))

    # ── Plot loss curves ──────────────────────────────────────────────
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_loss_curves(trainer.state.log_history, PLOTS_DIR / "loss_curves.png")
    print("Done.")


if __name__ == "__main__":
    main()
