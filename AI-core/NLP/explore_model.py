"""
Loads KBLab/electra-small-swedish-cased-discriminator as a sequence                                                                                          
classification model with the shared label schema and inspects its architecture.                                                                                            
                                                                                                                                                            
Run from the repo root with the ad-block-nlp venv active:                                                                                                    
    python AI-core/NLP/explore_model.py                                                                                                                      
""" 

from label_schema import LABELS, id2label, label2id
from transformers import AutoModelForSequenceClassification

MODEL_NAME = "KBLab/electra-small-swedish-cased-discriminator"

print(f"Loading model: {MODEL_NAME}")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABELS),
    label2id=label2id,
    id2label=id2label,
)
print("Done.")
















