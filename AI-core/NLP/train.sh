#!/bin/bash

#SBATCH --job-name=ad-block-nlp-train
#SBATCH --output=slurm-%j.out
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:L4:1

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/Ad-block-with-XR-glasses}"
NLP_DIR="$PROJECT_DIR/AI-core/NLP"

if [ -n "${VENV_ACTIVATE:-}" ]; then
    ACTIVATE_SCRIPT="$VENV_ACTIVATE"
else
    for candidate in \
        "$HOME/venvs/ad-block-nlp/bin/activate" \
        "$PROJECT_DIR/.venv/bin/activate" \
        "$PROJECT_DIR/venv/bin/activate" \
        "$HOME/.virtualenvs/ad-block-nlp/bin/activate"
    do
        if [ -f "$candidate" ]; then
            ACTIVATE_SCRIPT="$candidate"
            break
        fi
    done
fi

if [ -z "${ACTIVATE_SCRIPT:-}" ] || [ ! -f "${ACTIVATE_SCRIPT:-}" ]; then
    echo "Could not find a virtual environment activate script."
    echo "Set VENV_ACTIVATE to the full path before submitting, for example:"
    echo "  sbatch --export=ALL,VENV_ACTIVATE=\$HOME/path/to/bin/activate AI-core/NLP/train.sh"
    exit 1
fi

source "$ACTIVATE_SCRIPT"
cd "$NLP_DIR"

echo "Running on host: $(hostname)"
echo "Using Python: $(which python)"
echo "Working directory: $(pwd)"
echo "Virtual environment: $ACTIVATE_SCRIPT"
nvidia-smi

python -u train_model.py
