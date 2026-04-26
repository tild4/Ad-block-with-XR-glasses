"""                                                                                                                                                        
Fine-tunes KBLab/electra-small-swedish-cased-discriminator on the ad-classification                                                                          
dataset and saves the result to AI-core/NLP/saved_model/
"""

from pathlib import Path

import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

NLP_DIR = Path(__file__).resolve().parent
TRAIN_FILE = NLP_DIR / "dataset" / "train_augmented.jsonl"
TEST_FILE = NLP_DIR / "dataset" / "test_original.jsonl"
RESULTS_DIR = NLP_DIR / "results"
SAVED_MODEL_DIR = NLP_DIR / "saved_model"
PLOTS_DIR = NLP_DIR / "plots"

MODEL_NAME = "KBLab/electra-small-swedish-cased-discriminator"

# The order defines the output indices of the classifier                                                                                                  
# Should be in sync with test_model.py and the inference code                                                                                      
LABELS = ["inte reklam", "samhällsnyttig", "reklam"]                                                                                              
label2id = {label: i for i, label in enumerate(LABELS)}                                                                                                      
id2label = {i: label for i, label in enumerate(LABELS)}


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
        num_train_epochs=20,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
    )

    # 7. Data collator to handle dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 8. Initialize the automatic Trainer class
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 9. Start training
    print("Starting training...")                                                                                                                      
    trainer.train()                                                                                                                                      
    print("Training complete.") 

    # 10. Save the fine-tuned model and tokenizer
    print(f"Saving final model to {SAVED_MODEL_DIR}")
    trainer.save_model(str(SAVED_MODEL_DIR))

    # 11. Plot train + eval loss curves so we can spot under-/overfitting
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_loss_curves(trainer.state.log_history, PLOTS_DIR / "loss_curves.png")
    print("Done.")


if __name__ == "__main__":
    main() 





