
import json
import re
from pathlib import Path

 
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "data/processed/chunks.json"
OUTPUT_FILE = BASE_DIR / "data/processed/preprocessed_chunks.json"


def preprocess_text(text):

    text = text.lower()

    tokens = re.findall(r"[a-z0-9]+", text)

    normalized_text = " ".join(tokens)

    return tokens, normalized_text


def main():

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    preprocessed_chunks = []
    vocabulary = set()
    total_tokens = 0

    for chunk in chunks:
        tokens, normalized_text = preprocess_text(
            chunk["text"]
        )

        processed_chunk = {
            **chunk,
            "normalized_text": normalized_text,
            "tokens": tokens
        }

        preprocessed_chunks.append(processed_chunk)

        vocabulary.update(tokens)

        total_tokens += len(tokens)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(
            preprocessed_chunks,
            file,
            indent=2,
            ensure_ascii=False
        )

    print("\n===== PREPROCESSING RESULTS =====")
    print(f"Chunks processed: {len(preprocessed_chunks)}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Unique words: {len(vocabulary):,}")
    print(f"Output file: {OUTPUT_FILE}")

    first_chunk = preprocessed_chunks[0]

    print("\n===== PREPROCESSING EXAMPLE =====")

    print("\nOriginal text:")
    print(first_chunk["text"][:300])

    print("\nNormalized text:")
    print(first_chunk["normalized_text"][:300])

    print("\nFirst 20 tokens:")
    print(first_chunk["tokens"][:20])


if __name__ == "__main__":
    main()
