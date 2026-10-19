import json
import re
import time
from collections import Counter
from pathlib import Path

from stage3 import STOPWORDS

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "data/processed/preprocessed_chunks.json"
OUTPUT_FILE = BASE_DIR / "data/processed/approximate_search_results_question4.json"
TOP_K = 5
WINDOW_MULTIPLIER = 3


def tokenize_question(question):
    tokens = re.findall(r"[a-z0-9]+", question.lower())
    return list(dict.fromkeys(token for token in tokens if token not in STOPWORDS))


def levenshtein_distance(a, b):
    if len(a) < len(b):
        a, b = b, a

    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            substitution = previous[j - 1] + (char_a != char_b)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


def build_index(chunks):
    vocabulary_frequency = Counter()
    prepared_chunks = []

    for chunk in chunks:
        terms = Counter(chunk["tokens"])
        vocabulary_frequency.update(terms)
        content_tokens = [t for t in chunk["tokens"] if t not in STOPWORDS]
        prepared_chunks.append({
            "chunk": chunk,
            "terms": terms,
            "content_tokens": content_tokens,
        })

    return vocabulary_frequency, prepared_chunks


def correct_query(query_terms, vocabulary_frequency):
    vocabulary = vocabulary_frequency.keys()
    corrections = []

    for term in query_terms:
        if term in vocabulary_frequency:
            corrections.append({"original": term, "corrected": term, "distance": 0})
            continue

        max_distance = 0 if len(term) <= 3 else (1 if len(term) <= 7 else 2)
        candidates = []

        if max_distance:
            for candidate in vocabulary:
                if abs(len(candidate) - len(term)) > max_distance:
                    continue
                distance = levenshtein_distance(term, candidate)
                if distance <= max_distance:
                    candidates.append((distance, -vocabulary_frequency[candidate], candidate))

        if candidates:
            distance, _, best = min(candidates)
            corrections.append({"original": term, "corrected": best, "distance": distance})
        else:
            corrections.append({"original": term, "corrected": term, "distance": None})

    return corrections


def search_levenshtein(prepared_chunks, corrections):
    results = []
    count = len(corrections)
    if not count:
        return results

    for item in prepared_chunks:
        matches = {}
        weighted_coverage = 0.0
        frequency_bonus = 0.0

        for correction in corrections:
            term = correction["corrected"]
            frequency = item["terms"].get(term, 0)
            if not frequency:
                continue

            distance = correction["distance"]
            reliability = 0.0 if distance is None else 1 - distance / (len(correction["original"]) + 1)
            weighted_coverage += reliability
            frequency_bonus += min(frequency, 3) / 3
            matches[correction["original"]] = {"matched_as": term, "occurrences": frequency}

        if matches:
            score = 100 * (0.90 * weighted_coverage / count + 0.10 * frequency_bonus / count)
            results.append({
                "id": item["chunk"]["id"],
                "document": item["chunk"]["document"],
                "score": round(score, 4),
                "matches": matches,
                "text": item["chunk"]["text"],
            })

    return sorted(results, key=lambda r: (-r["score"], -len(r["matches"]), r["id"]))


def word_ngrams(words):
    grams = set(words)
    grams.update(zip(words, words[1:]))
    return grams


def jaccard_similarity(set_a, set_b):
    union = set_a | set_b
    return len(set_a & set_b) / len(union) if union else 0.0


def max_window_jaccard(content_tokens, query_grams, query_length):
    if not content_tokens or not query_grams:
        return 0.0

    window_size = min(len(content_tokens), max(4, min(12, WINDOW_MULTIPLIER * query_length)))
    best = 0.0

    for start in range(len(content_tokens) - window_size + 1):
        window = content_tokens[start:start + window_size]
        score = jaccard_similarity(query_grams, word_ngrams(window))
        if score > best:
            best = score
            if best == 1.0:
                break

    return best


def search_jaccard(prepared_chunks, query_terms):
    results = []
    query_grams = word_ngrams(query_terms)

    for item in prepared_chunks:
        similarity = max_window_jaccard(item["content_tokens"], query_grams, len(query_terms))
        if similarity <= 0:
            continue

        results.append({
            "id": item["chunk"]["id"],
            "document": item["chunk"]["document"],
            "score": round(100 * similarity, 4),
            "similarity": round(similarity, 6),
            "text": item["chunk"]["text"],
        })

    return sorted(results, key=lambda r: (-r["score"], r["id"]))


def combine_rankings(levenshtein_results, jaccard_results, chunks):
    lev_scores = {item["id"]: item["score"] for item in levenshtein_results}
    jac_scores = {item["id"]: item["score"] for item in jaccard_results}
    combined = []

    for chunk in chunks:
        chunk_id = chunk["id"]
        lev = lev_scores.get(chunk_id, 0.0)
        jac = jac_scores.get(chunk_id, 0.0)
        score = round(0.65 * lev + 0.35 * jac, 4)
        if score > 0:
            combined.append({
                "id": chunk_id,
                "document": chunk["document"],
                "score": score,
                "levenshtein_score": lev,
                "jaccard_score": jac,
                "text": chunk["text"],
            })

    return sorted(combined, key=lambda r: (-r["score"], r["id"]))


def print_ranking(title, results, seconds, total_chunks):
    print(f"\n===== {title} =====")
    print(f"Search time: {seconds:.6f} seconds")
    print(f"Chunks examined: {total_chunks}")
    print(f"Chunks with matches: {len(results)}")

    for position, chunk in enumerate(results[:TOP_K], start=1):
        print(f"\n{position}. ID: {chunk['id']}")
        print(f"   Document: {chunk['document']}")
        print(f"   Score: {chunk['score']:.4f}")
        if "matches" in chunk:
            print(f"   Matches: {chunk['matches']}")
        if "levenshtein_score" in chunk:
            print(f"   Levenshtein: {chunk['levenshtein_score']:.4f}; Jaccard: {chunk['jaccard_score']:.4f}")
        print(f"   Text: {chunk['text'][:450]}...")


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"File not found: {INPUT_FILE}. Run Stage 2 first.")

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    vocabulary_frequency, prepared_chunks = build_index(chunks)
    print(f"Loaded {len(chunks)} chunks and {len(vocabulary_frequency)} unique terms.")

    question = input("Question in English: ").strip()
    if not question:
        question = "What is the difference between K-Means and DBSCAN?"

    query_terms = tokenize_question(question)
    if not query_terms:
        print("No meaningful search terms found.")
        return

    t0 = time.perf_counter()
    corrections = correct_query(query_terms, vocabulary_frequency)
    levenshtein_results = search_levenshtein(prepared_chunks, corrections)
    lev_elapsed = time.perf_counter() - t0

    t0 = time.perf_counter()
    jaccard_results = search_jaccard(prepared_chunks, query_terms)
    jac_elapsed = time.perf_counter() - t0

    t0 = time.perf_counter()
    corrected_terms = list(dict.fromkeys(item["corrected"] for item in corrections))
    corrected_jaccard_results = search_jaccard(prepared_chunks, corrected_terms)
    hybrid_results = combine_rankings(levenshtein_results, corrected_jaccard_results, chunks)
    hybrid_extra_time = time.perf_counter() - t0
    hybrid_elapsed = lev_elapsed + hybrid_extra_time

    print(f"\nQuestion: {question}")
    print(f"Original query terms: {query_terms}")
    print(f"Corrections: {corrections}")
    print_ranking("LEVENSHTEIN RESULTS", levenshtein_results, lev_elapsed, len(chunks))
    print_ranking("JACCARD RESULTS", jaccard_results, jac_elapsed, len(chunks))
    print_ranking("HYBRID RESULTS", hybrid_results, hybrid_elapsed, len(chunks))

    report = {
        "question": question,
        "query_terms": query_terms,
        "corrections": corrections,
        "levenshtein": {
            "elapsed_seconds": lev_elapsed,
            "chunks_examined": len(chunks),
            "chunks_with_matches": len(levenshtein_results),
            "top_chunks": levenshtein_results[:TOP_K],
        },
        "jaccard": {
            "elapsed_seconds": jac_elapsed,
            "chunks_examined": len(chunks),
            "chunks_with_matches": len(jaccard_results),
            "top_chunks": jaccard_results[:TOP_K],
        },
        "hybrid": {
            "elapsed_seconds": hybrid_elapsed,
            "chunks_examined": len(chunks),
            "chunks_with_matches": len(hybrid_results),
            "top_chunks": hybrid_results[:TOP_K],
        },
    }
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
