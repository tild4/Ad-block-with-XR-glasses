"""
Loads KBLab/electra-small-swedish-cased-discriminator as a sequence                                                                                          
classification model with 4 labels and inspects its architecture.                                                                                            
                                                                                                                                                            
Run from the repo root with the ad-block-nlp venv active:                                                                                                    
    python AI-core/NLP/explore_model.py                                                                                                                      
""" 

from transformers import AutoModelForSequenceClassification

MODEL_NAME = "KBLab/electra-small-swedish-cased-discriminator"
LABELS = ["inte reklam", "reklam", "skadlig", "samhällsnyttig"]

label2id = {label: i for i, label in enumerate(LABELS)}
id2label = {i: label for i, label in enumerate(LABELS)}

print(f"Loading model: {MODEL_NAME}")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABELS),
    label2id=label2id,
    id2label=id2label,
)
print("Done.")

















