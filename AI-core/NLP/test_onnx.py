"""
Smoke-test for the ONNX export.

Runs the same inputs through:
  1. The PyTorch checkpoint in saved_model/
  2. The ONNX model in models/model.onnx (via onnxruntime)

…and compares logits + predicted classes. If the numerical difference is
small (< 1e-4), the ONNX export is faithful and safe to ship to Unity.
"""

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from label_schema import LABELS
from transformers import AutoModelForSequenceClassification, AutoTokenizer

NLP_DIR = Path(__file__).resolve().parent
SAVED_MODEL_DIR = NLP_DIR / "saved_model"
MODELS_DIR = NLP_DIR / "models"
ONNX_MODEL_PATH = MODELS_DIR / "model.onnx"

# A small set of test sentences — one per class plus a noisy OCR-style one.
TEST_SENTENCES = [
    "Imorgon öppnar biblioteket klockan nio",            # non-ad
    "Vaccinationen mot influensa är gratis i höst",      # socially beneficial
    "Halva priset på alla laptops denna vecka!",         # ad
    "B0ka b0rd nu h0s Café S01 till halva priset",       # noisy / mixed
]


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax along the last axis."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def main():
    if not SAVED_MODEL_DIR.exists():
        raise FileNotFoundError(f"Missing PyTorch checkpoint: {SAVED_MODEL_DIR}")
    if not ONNX_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing ONNX model: {ONNX_MODEL_PATH}")

    # --- Load tokenizer (shared) ---
    # We load from MODELS_DIR because that's the tokenizer that ships with
    # the ONNX file. It should be identical to the one in saved_model/, but
    # using the ONNX-side copy mirrors what Unity will do.
    print(f"Loading tokenizer from {MODELS_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(MODELS_DIR)

    # --- Load PyTorch model ---
    print(f"Loading PyTorch model from {SAVED_MODEL_DIR}")
    pt_model = AutoModelForSequenceClassification.from_pretrained(SAVED_MODEL_DIR)
    pt_model.eval()  # disable dropout etc.

    # --- Load ONNX session ---
    print(f"Loading ONNX session from {ONNX_MODEL_PATH}")
    ort_session = ort.InferenceSession(
        str(ONNX_MODEL_PATH),
        providers=["CPUExecutionProvider"],
    )
    print(f"  ONNX inputs:  {[i.name for i in ort_session.get_inputs()]}")
    print(f"  ONNX outputs: {[o.name for o in ort_session.get_outputs()]}")
    print()

    # --- Run both models on each test sentence ---
    max_diff_overall = 0.0
    mismatches = 0

    for i, text in enumerate(TEST_SENTENCES):
        encoded = tokenizer(
            text,
            return_tensors="np",  # numpy for ONNX; we convert for PyTorch below
            truncation=True,
            max_length=64,
        )

        # PyTorch forward pass (no grad, eval mode)
        with torch.no_grad():
            pt_inputs = {k: torch.from_numpy(v) for k, v in encoded.items()}
            pt_logits = pt_model(**pt_inputs).logits.numpy()

        # ONNX forward pass.
        # ONNX models exported by optimum expect input names that match
        # the model's signature: input_ids, attention_mask, (token_type_ids).
        ort_inputs = {
            inp.name: encoded[inp.name]
            for inp in ort_session.get_inputs()
            if inp.name in encoded
        }
        ort_logits = ort_session.run(None, ort_inputs)[0]

        # --- Compare ---
        diff = np.abs(pt_logits - ort_logits).max()
        max_diff_overall = max(max_diff_overall, diff)

        pt_pred = int(np.argmax(pt_logits, axis=-1)[0])
        ort_pred = int(np.argmax(ort_logits, axis=-1)[0])
        match = "OK" if pt_pred == ort_pred else "MISMATCH"
        if pt_pred != ort_pred:
            mismatches += 1

        pt_probs = softmax(pt_logits)[0]
        ort_probs = softmax(ort_logits)[0]

        print(f"[{i}] {text!r}")
        print(f"    PyTorch pred:  {LABELS[pt_pred]} ({pt_probs[pt_pred]:.4f})")
        print(f"    ONNX    pred:  {LABELS[ort_pred]} ({ort_probs[ort_pred]:.4f})")
        print(f"    max |logit diff|: {diff:.2e}   [{match}]")
        print()

    # --- Summary ---
    print("=" * 50)
    print(f"Sentences tested:        {len(TEST_SENTENCES)}")
    print(f"Class mismatches:        {mismatches}")
    print(f"Max |logit diff| seen:   {max_diff_overall:.2e}")

    if mismatches == 0 and max_diff_overall < 1e-4:
        print("Result: PASS — ONNX export is faithful.")
    elif mismatches == 0 and max_diff_overall < 1e-3:
        print("Result: WARNING — small numerical drift, but predictions match.")
    else:
        print("Result: FAIL — investigate before shipping to Unity.")


if __name__ == "__main__":
    main()
