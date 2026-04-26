"""
Quick test: classify a single sentence via the ONNX model.

Usage:
    python3 AI-core/NLP/testPredict.py "Halva priset på alla laptops"
"""

import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from label_schema import LABELS
from transformers import AutoTokenizer

# --- Paths ---
NLP_DIR = Path(__file__).resolve().parent
MODELS_DIR = NLP_DIR / "models"


def main():
    # Load tokenizer and ONNX model from the export directory
    tokenizer = AutoTokenizer.from_pretrained(MODELS_DIR)
    session = ort.InferenceSession(str(MODELS_DIR / "model.onnx"))

    # Use command-line argument as input, or a default sentence
    text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Halva priset på alla laptops"

    # Tokenize (same settings as training: truncation + max_length=64)
    encoded = tokenizer(text, return_tensors="np", truncation=True, max_length=64)

    # Build input dict matching the ONNX model's expected input names
    inputs = {i.name: encoded[i.name] for i in session.get_inputs() if i.name in encoded}

    # Run inference
    logits = session.run(None, inputs)[0]

    # Convert logits to probabilities via softmax
    probs = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
    pred = int(np.argmax(probs))

    # Print results
    print(f"Text:       {text}")
    print(f"Prediction: {LABELS[pred]} ({probs[0][pred]:.1%})")
    for i, label in enumerate(LABELS):
        print(f"  {label:20s} {probs[0][i]:.1%}")


if __name__ == "__main__":
    main()
