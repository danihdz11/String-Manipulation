
import json
from pathlib import Path

CHUNK_SIZE = 180
OVERLAP = 30

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

DOCUMENTS = {
    "decision_trees": "Decision Trees",
    "clustering": "Clustering",
    "cross_validation": "Cross-validation",
    "model_evaluation": "Model Evaluation"
}


def create_chunks(text, document_id, document_name):
    words = text.split()
    chunks = []

    step = CHUNK_SIZE - OVERLAP

    for start in range(0, len(words), step):
        end = min(start + CHUNK_SIZE, len(words))
        chunk_words = words[start:end]

        chunk = {
            "id": f"{document_id}_{len(chunks) + 1}",
            "document": document_name,
            "text": " ".join(chunk_words),
            "word_count": len(chunk_words),
            "start_word": start,
            "end_word": end
        }

        chunks.append(chunk)

        if end == len(words):
            break

    return chunks


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    all_chunks = []
    total_words = 0
    total_characters = 0

    for document_id, document_name in DOCUMENTS.items():

        file_path = RAW_DIR / f"{document_id}.txt"

        text = file_path.read_text(encoding="utf-8")

        word_count = len(text.split())
        character_count = len(text)

        chunks = create_chunks(
            text,
            document_id,
            document_name
        )

        all_chunks.extend(chunks)

        total_words += word_count
        total_characters += character_count

        print(f"\nDocument: {document_name}")
        print(f"Words: {word_count:,}")
        print(f"Characters: {character_count:,}")
        print(f"Chunks generated: {len(chunks)}")
        print("-" * 40)

    output_path = PROCESSED_DIR / "chunks.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            all_chunks,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\n===== FINAL SUMMARY =====")
    print(f"Total documents: {len(DOCUMENTS)}")
    print(f"Total words: {total_words:,}")
    print(f"Total characters: {total_characters:,}")
    print(f"Total chunks: {len(all_chunks):,}")
    print(f"Output file: {output_path}")

    if all_chunks:
        print("\n===== FIRST CHUNK EXAMPLE =====")
        print(f"ID: {all_chunks[0]['id']}")
        print(f"Document: {all_chunks[0]['document']}")
        print(f"Words: {all_chunks[0]['word_count']}")
        print(f"Text preview: {all_chunks[0]['text'][:300]}...")


if __name__ == "__main__":
    main()
