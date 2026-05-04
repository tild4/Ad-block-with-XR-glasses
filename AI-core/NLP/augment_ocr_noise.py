"""
Generates OCR-noisy variants of clean training data.
For each clean example, N noisy copies are created with the same label.
The output file contains both the clean originals AND the noisy variants, shuffled together.
"""

import json
import random
import sys


NOISE_COPIES = 1  # how many noisy variants per clean example
CHAR_NOISE_PROB = 0.15  # probability of noise per character

# Common OCR character confusions (what the camera might misread)
SWAP_MAP = {
    "ö": "o", "ä": "a", "å": "a",
    "Ö": "O", "Ä": "A", "Å": "A",
    "l": "1", "1": "l",
    "O": "0", "0": "O",
    "é": "e", "ü": "u",
    "rn": "m", 
}


def add_noise(text):
    result = []
    i = 0
    while i < len(text):
        # Skip spaces from char-level noise (handled separately below)
        if text[i] == " ":
            result.append(" ")
            i += 1
            continue

        # Roll the dice (should this character get noise?)
        if random.random() > CHAR_NOISE_PROB:
            result.append(text[i])
            i += 1
            continue

        # Pick a random noise type
        noise_type = random.choice(["swap", "drop", "double", "case"])

        if noise_type == "swap":
            # Check 2-char combos first (like "rn" → "m")
            bigram = text[i:i+2]
            if bigram in SWAP_MAP:
                result.append(SWAP_MAP[bigram])
                i += 2
                continue
            # Then single char swaps (like "ö" → "o")
            if text[i] in SWAP_MAP:
                result.append(SWAP_MAP[text[i]])
            else:
                result.append(text[i])  # no swap available, keep it

        elif noise_type == "drop":
            pass  # just skip the character

        elif noise_type == "double":
            # Repeat the character (like "Hemköp" → "Hemmköp")
            result.append(text[i])
            result.append(text[i])

        elif noise_type == "case":
            # Flip upper/lower case
            if text[i].isupper():
                result.append(text[i].lower())
            else:
                result.append(text[i].upper())

        i += 1

    # Space noise: randomly remove or insert spaces
    text_out = "".join(result)
    words = text_out.split(" ")
    final_words = []
    for word in words:
        if len(word) > 3 and random.random() < 0.1:
            # Insert a space in the middle of a word (like "Hemköp" → "Hem köp")
            split = random.randint(1, len(word) - 1)
            final_words.append(word[:split])
            final_words.append(word[split:])
        elif random.random() < 0.1 and final_words:
            # Merge with previous word (like "Just nu" → "Justnu")
            final_words[-1] = final_words[-1] + word
        else:
            final_words.append(word)

    return " ".join(final_words)


def main():
    if len(sys.argv) != 3:
        print("Usage: python augment_ocr_noise.py <input.jsonl> <output.jsonl>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    # Read clean examples
    examples = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line.strip()))

    # Start with the clean originals (model needs to see correct text too)
    all_examples = list(examples)

    # Add noisy variants
    for ex in examples:
        for _ in range(NOISE_COPIES):
            noisy_text = add_noise(ex["text"])
            all_examples.append({"text": noisy_text, "label": ex["label"]})

    # Shuffle so clean and noisy are mixed together
    random.seed(42)
    random.shuffle(all_examples)

    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    clean = len(examples)
    noisy = len(all_examples) - clean
    print(f"Clean:  {clean}")
    print(f"Noisy:  {noisy} ({NOISE_COPIES} per clean example)")
    print(f"Total:  {len(all_examples)}")


if __name__ == "__main__":
    main()
