"""                                                                                                                                                        
Fine-tunes KBLab/electra-small-swedish-cased-discriminator on the ad-classification                                                                          
dataset and saves the result to AI-core/NLP/saved_model/
"""

from pathlib import Path

from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,                                                                                                                      
    AutoTokenizer,
    DataCollatorWithPadding,                                                                                                                                 
    Trainer,                                                                                                                                               
    TrainingArguments,                                                                                                                                     
)

NLP_DIR = Path(__file__).resolve().parent                                                                                                                    
TRAIN_FILE = NLP_DIR / "dataset" / "train_large.jsonl"                                                                                                       
RESULTS_DIR = NLP_DIR / "results"                                                                                         
SAVED_MODEL_DIR = NLP_DIR / "saved_model" 

MODEL_NAME = "KBLab/electra-small-swedish-cased-discriminator"

# The order defines the output indices of the classifier                                                                                                  
# Should be in sync with test_model.py and the inference code                                                                                      
LABELS = ["inte reklam", "reklam", "skadlig", "samhällsnyttig"]                                                                                              
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


def main():                                    
    # 1. Load dataset from jsonl                                                                                                                             
    print(f"Loading dataset from {TRAIN_FILE}")                                                                                                              
    dataset = load_dataset("json", data_files=str(TRAIN_FILE), split="train")                                                                                
    print(f"  {len(dataset)} examples, columns: {dataset.column_names}")                                                                                     
                                                                                                                                                            
    # 2. Convert string labels to integer ids                                                                                                                
    dataset = dataset.map(encode_label)                                                                                                                    
    print(f"  After label encoding: {dataset[0]}")   

    # 3. Load tokenizer                                                                                                                                      
    print(f"Loading tokenizer: {MODEL_NAME}")                                                                                                              
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)                                                                                                    
                                                                                                                                                            
    # 4. Tokenize the entire dataset                                                                                                                         
    dataset = dataset.map(                                                                                                                                   
        lambda batch: tokenize(batch, tokenizer),                                                                                                          
        batched=True,                                                                                                                                        
    )
    print(f"  After tokenization: columns = {dataset.column_names}")                                                                                         
    print(f"  Example 0 input_ids length: {len(dataset[0]['input_ids'])}")                                                                                                                                                                                                                                                    



if __name__ == "__main__":
    main() 





