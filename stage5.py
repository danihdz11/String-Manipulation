
import csv
import json
import re
import sys
import time
from pathlib import Path

from stage4 import (
    build_index,
    combine_rankings,
    correct_query,
    jaccard_similarity,
    search_jaccard,
    search_levenshtein,
    tokenize_question,
)

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "data/processed/preprocessed_chunks.json"
OUTPUT_JSON = BASE_DIR / "data/processed/stage5_results.json"
OUTPUT_CSV = BASE_DIR / "data/processed/stage5_results.csv"

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"   # 0.5B ignoraba el contexto; 1.5B sigue mejor las instrucciones
MIN_COVERAGE = 0.5      # fraction of query terms that must appear in the selected context
MAX_CHUNKS = 4          # chunks taken greedily by score
HARD_MAX_CHUNKS = 6     # absolute limit after the coverage step
MAX_CONTEXT_WORDS = 350
OVERLAP_WORDS = 30      # same overlap used in Stage 1
REDUNDANCY_THRESHOLD = 0.5



QUESTIONS = [
    ("clean", "What is the difference between K-Means and DBSCAN?"),
    ("clean", "How do decision trees work for classification?"),
    ("clean", "What is the difference between precision and recall?"),
    ("clean", "What is cross-validation used for?"),
    ("clean", "What is overfitting?"),
    ("clean", "How does DBSCAN find clusters?"),
    ("clean", "What is the F1 score?"),
    ("clean", "What are the advantages of decision trees?"),
    ("clean", "How does k-fold cross-validation split the data?"),
    ("clean", "What is the silhouette coefficient?"),
    ("clean", "What is a confusion matrix?"),
    ("clean", "How can decision trees be pruned?"),
    ("typo", "What is the difference between K-Meens and DBSCN?"),
    ("typo", "How do decison trees work for classificaton?"),
    ("typo", "What is precison and recal?"),
    ("typo", "What is crosvalidation used for?"),
    ("typo", "What is the siluette coeficient?"),
    ("offtopic", "How do I train a transformer language model?"),
    ("offtopic", "What is the capital of Germany?"),
    ("offtopic", "Which neural network architecture is best for images?"),
]




def retrieve_hybrid(question, vocabulary_frequency, prepared_chunks, chunks):
    """Hybrid retrieval exactly as in Stage 4. Returns (ranking, corrected_terms, corrections)."""
    query_terms = tokenize_question(question)
    if not query_terms:
        return [], [], []
    corrections = correct_query(query_terms, vocabulary_frequency)
    lev_results = search_levenshtein(prepared_chunks, corrections)
    corrected_terms = list(dict.fromkeys(c["corrected"] for c in corrections))
    jac_results = search_jaccard(prepared_chunks, corrected_terms)
    ranking = combine_rankings(lev_results, jac_results, chunks)
    return ranking, corrected_terms, corrections





def split_id(chunk_id):
    prefix, _, number = chunk_id.rpartition("_")
    return prefix, int(number) if number.isdigit() else None


def select_context(ranking, corrected_terms, chunk_tokens):
    """Pick chunks for the LLM.

    1. Greedy by score, up to MAX_CHUNKS, skipping redundant chunks.
    2. Coverage step: if a query term appears in no selected chunk, add the
       best-ranked chunk that contains it (up to HARD_MAX_CHUNKS).
    3. Consecutive chunks overlap by 30 words, so the repeated words are trimmed.
    Everything respects MAX_CONTEXT_WORDS.
    """
    selected = []   # dicts: id, words, wordset
    total_words = 0

    def try_add(result):
        nonlocal total_words
        prefix, number = split_id(result["id"])
        words = result["text"].split()

        # Trim overlap with already selected neighbours.
        for s in selected:
            s_prefix, s_number = split_id(s["id"])
            if s_prefix == prefix and number is not None and s_number is not None:
                if s_number == number - 1:
                    words = words[OVERLAP_WORDS:]
                elif s_number == number + 1:
                    words = words[:-OVERLAP_WORDS]

        wordset = {w.lower() for w in words}
        for s in selected:
            if jaccard_similarity(wordset, s["wordset"]) > REDUNDANCY_THRESHOLD:
                return False
        if total_words + len(words) > MAX_CONTEXT_WORDS:
            return False

        selected.append({"id": result["id"], "document": result["document"],
                         "score": result["score"], "words": words, "wordset": wordset})
        total_words += len(words)
        return True

    # Step 1: greedy by score.
    for result in ranking:
        if len(selected) >= MAX_CHUNKS:
            break
        try_add(result)

    # Step 2: make sure every meaningful query term is covered.
    for term in corrected_terms:
        if len(term) <= 3 or len(selected) >= HARD_MAX_CHUNKS:
            continue
        if any(term in chunk_tokens[s["id"]] for s in selected):
            continue
        chosen = {s["id"] for s in selected}
        for result in ranking:
            if result["id"] not in chosen and term in chunk_tokens[result["id"]]:
                if try_add(result):
                    break

    context = "\n\n".join(f"[{s['id']}] " + " ".join(s["words"]) for s in selected)
    return selected, context


def context_coverage(corrected_terms, selected, chunk_tokens):
    """Fraction of query terms that appear in at least one selected chunk."""
    if not corrected_terms or not selected:
        return 0.0
    covered = sum(
        1 for term in corrected_terms
        if any(term in chunk_tokens[s["id"]] for s in selected)
    )
    return covered / len(corrected_terms)





def load_model():
    from transformers import pipeline  # imported here so retrieval can be tested without it
    return pipeline("text-generation", model=MODEL_NAME)


def generate_answer(generator, question, context):
    messages = [
        {"role": "system",
         "content": "You answer questions about machine learning using ONLY the provided context. "
                    "If the context does not contain the answer, say exactly: "
                    "\"The provided context does not contain this information.\" Answer in one to three complete sentences, reusing the wording of the context."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    output = generator(messages, max_new_tokens=200, do_sample=False)
    return output[0]["generated_text"][-1]["content"].strip()


def main():
    with INPUT_FILE.open("r", encoding="utf-8") as f:
        chunks = json.load(f)

    # Offline setup (not part of query time).
    vocabulary_frequency, prepared_chunks = build_index(chunks)
    chunk_tokens = {c["id"]: set(c["tokens"]) for c in chunks}
    generator = load_model()

    # Optional: python stage5.py "your question here"
    if len(sys.argv) > 1:
        questions = [("custom", " ".join(sys.argv[1:]))]
    else:
        questions = QUESTIONS

    results = []
    for number, (category, question) in enumerate(questions, start=1):
        t0 = time.perf_counter()
        ranking, corrected_terms, corrections = retrieve_hybrid(
            question, vocabulary_frequency, prepared_chunks, chunks)
        selected, context = select_context(ranking, corrected_terms, chunk_tokens)
        retrieval_seconds = time.perf_counter() - t0

        coverage = context_coverage(corrected_terms, selected, chunk_tokens)
        if not selected:
            answer = "(no chunks retrieved)"
            generation_seconds = 0.0
        elif coverage < MIN_COVERAGE:
            # Retrieval is too weak: do not let the model improvise from its own knowledge.
            answer = "The provided context does not contain this information. (rejected: low coverage)"
            generation_seconds = 0.0
        else:
            t0 = time.perf_counter()
            answer = generate_answer(generator, question, context)
            generation_seconds = time.perf_counter() - t0

        row = {
            "n": number,
            "category": category,
            "question": question,
            "corrections": [c for c in corrections if c["original"] != c["corrected"]],
            "chunks_with_matches": len(ranking),
            "selected_chunks": [{"id": s["id"], "score": s["score"]} for s in selected],
            "context_words": sum(len(s["words"]) for s in selected),
            "coverage": round(coverage, 2),
            "context": context,
            "answer": answer,
            "retrieval_seconds": round(retrieval_seconds, 6),
            "generation_seconds": round(generation_seconds, 3),
            "supported_by_context": "",   # fill in by hand: yes / partial / no
        }
        results.append(row)

        print(f"\n[{number}] ({category}) {question}")
        print(f"  Chunks: {[s['id'] for s in selected]}  | retrieval: {retrieval_seconds*1000:.1f} ms")
        print(f"  Answer: {answer}")

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n", "category", "question", "selected_chunks",
                         "retrieval_s", "generation_s", "answer", "supported_by_context"])
        for r in results:
            writer.writerow([r["n"], r["category"], r["question"],
                             ";".join(c["id"] for c in r["selected_chunks"]),
                             r["retrieval_seconds"], r["generation_seconds"],
                             r["answer"], r["supported_by_context"]])

    print(f"\nSaved: {OUTPUT_JSON}\nSaved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()