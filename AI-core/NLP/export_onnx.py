"""
Exports the fine-tuned Hugging Face checkpoint in AI-core/NLP/saved_model/
to ONNX format under AI-core/NLP/models/ for later Unity Sentis import.
"""

from pathlib import Path

from optimum.exporters.onnx import main_export

NLP_DIR = Path(__file__).resolve().parent

# "clean" → no OCR noise, "augmented" → OCR noise
MODE = "augmented"

# "mixed" → real + synthetic, "real_only" → ablation baseline
DATASET = "mixed"

RUN_TAG = f"{DATASET}_{MODE}"
SAVED_MODEL_DIR = NLP_DIR / "saved_model" / RUN_TAG
MODELS_DIR = NLP_DIR / "models" / RUN_TAG

# Unity Sentis docs recommend ONNX opset 15 for best compatibility.
ONNX_OPSET = 15


def main():
    if not SAVED_MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Could not find saved model directory: {SAVED_MODEL_DIR}"
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Exporting ONNX model from {SAVED_MODEL_DIR}")
    print(f"Output directory: {MODELS_DIR}")
    print(f"Using opset: {ONNX_OPSET}")

    main_export(
        model_name_or_path=str(SAVED_MODEL_DIR),
        output=MODELS_DIR,
        task="text-classification",
        opset=ONNX_OPSET,
        framework="pt",
        device="cpu",
        local_files_only=True,
    )

    print("ONNX export complete.")
    print(f"Model file: {MODELS_DIR / 'model.onnx'}")


if __name__ == "__main__":
    main()
