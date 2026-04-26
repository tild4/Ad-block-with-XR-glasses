"""                                                                                                                                                        
Fine-tunes KBLab/electra-small-swedish-cased-discriminator on the ad-classification                                                                          
dataset and saves the result to AI-core/NLP/saved_model/
"""

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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

NLP_DIR = Path(__file__).resolve().parent
TRAIN_FILE = NLP_DIR / "dataset" / "train_augmented.jsonl"
TEST_FILE = NLP_DIR / "dataset" / "test_original_augmented.jsonl"
RESULTS_DIR = NLP_DIR / "results"
SAVED_MODEL_DIR = NLP_DIR / "saved_model"
PLOTS_DIR = NLP_DIR / "plots"

MODEL_NAME = "KBLab/electra-small-swedish-cased-discriminator"
POSITIVE_LABEL = "reklam"
POSITIVE_LABEL_ID = label2id[POSITIVE_LABEL]


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


def load_and_prepare(jsonl_path, tokenizer, label_name):
    """Load a jsonl, encode labels to ids, tokenize. Returns the prepared dataset."""
    print(f"Loading {label_name} dataset from {jsonl_path}")
    ds = load_dataset("json", data_files=str(jsonl_path), split="train")
    print(f"  {len(ds)} examples, columns: {ds.column_names}")
    ds = ds.map(encode_label)
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
    # 1+2+3+4. Load tokenizer once, then prepare both datasets through it.
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = load_and_prepare(TRAIN_FILE, tokenizer, "train")
    eval_dataset = load_and_prepare(TEST_FILE, tokenizer, "eval")


    # 5. Load model with a classification head 
    print(f"Loading model: {MODEL_NAME}")                                                                                                                    
    model = AutoModelForSequenceClassification.from_pretrained(                                                                                              
        MODEL_NAME,                                                                                                                                          
        num_labels=len(LABELS),                                                                                                                              
        label2id=label2id,                                                                                                                                   
        id2label=id2label,                                                                                                                                 
    ) 

    # 6. Training configuration
    args = TrainingArguments(
        output_dir=str(RESULTS_DIR),
        num_train_epochs=15,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        learning_rate=1e-5,
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

    # 7. Data collator to handle dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 8. Print class distribution so we can verify balance
    label_counts = Counter(train_dataset["label"])
    total = sum(label_counts.values())
    print("Training class distribution:")
    for lid in sorted(label_counts):
        print(f"  {id2label[lid]}: {label_counts[lid]} ({label_counts[lid]/total*100:.1f}%)")

    # 9. Initialize trainer
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

    # 10. Start training
    print("Starting training...")                                                                                                                      
    trainer.train()                                                                                                                                      
    print("Training complete.") 
    if trainer.state.best_model_checkpoint is not None:
        print(f"Best checkpoint: {trainer.state.best_model_checkpoint}")
        print(f"Best {args.metric_for_best_model}: {trainer.state.best_metric:.4f}")

    # 11. Save the fine-tuned model and tokenizer
    print(f"Saving final model to {SAVED_MODEL_DIR}")
    trainer.save_model(str(SAVED_MODEL_DIR))

    # 12. Plot train + eval loss curves so we can spot under-/overfitting
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_loss_curves(trainer.state.log_history, PLOTS_DIR / "loss_curves.png")
    print("Done.")


if __name__ == "__main__":
    main() 



