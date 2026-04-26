"""Shared label schema for the current 3-class NLP pipeline."""

LABELS = ["inte reklam", "samhällsnyttig", "reklam"]
label2id = {label: i for i, label in enumerate(LABELS)}
id2label = {i: label for i, label in enumerate(LABELS)}
LABEL_IDS = list(range(len(LABELS)))
