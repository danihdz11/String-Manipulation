import json
import re
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "data/processed/preprocessed_chunks.json"
OUTPUT_FILE = BASE_DIR / "data/processed/exact_search_results.json"
TOP_K = 5

# Common question words that do not help identify the topic.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "between", "by", "can",
    "compare", "describe", "did", "difference", "do", "does", "explain",
    "for", "from", "how", "in", "into", "is", "it", "me", "of", "on",
    "or", "please", "the", "their", "there", "these", "this", "to",
    "us", "was", "were", "what", "when", "where", "which", "who",
    "why", "with", "would", "you", "your",
}



def extract_patterns(question):
    """
    Extract meaningful search patterns while keeping
    hyphenated terms as a single concept.
    """
    raw_terms = re.findall(
        r"[a-z0-9]+(?:-[a-z0-9]+)*",
        question.lower()
    )

    normalized_terms = [
        term.replace("-", " ")
        for term in raw_terms
    ]

    important_terms = [
        term
        for term in normalized_terms
        if term not in STOPWORDS
        and len(term.replace(" ", "")) > 1
    ]

    patterns = []
    seen = set()

    for term in important_terms:
        if term not in seen:
            patterns.append(term)
            seen.add(term)

    # Add expressions consisting of adjacent words.
    # Hyphenated terms are already preserved above.
    for i in range(len(normalized_terms) - 1):
        first = normalized_terms[i]
        second = normalized_terms[i + 1]

        if (
            first not in STOPWORDS
            and second not in STOPWORDS
            and len(first) > 1
            and len(second) > 1
            and " " not in first
            and " " not in second
        ):
            phrase = first + " " + second

            if phrase not in seen:
                patterns.append(phrase)
                seen.add(phrase)

    return patterns


def compute_lps(pattern):
    """Longest proper prefix that is also a suffix (KMP preprocessing)."""
    lps = [0] * len(pattern)
    prefix_length = 0
    i = 1

    while i < len(pattern):
        if pattern[i] == pattern[prefix_length]:
            prefix_length += 1
            lps[i] = prefix_length
            i += 1
        elif prefix_length != 0:
            prefix_length = lps[prefix_length - 1]
        else:
            i += 1

    return lps


def kmp_count(text, pattern):
    """Count all exact occurrences of pattern in text using KMP."""
    if not pattern:
        return 0

    lps = compute_lps(pattern)
    i = 0
    j = 0
    matches = 0

    while i < len(text):
        if text[i] == pattern[j]:
            i += 1
            j += 1
            if j == len(pattern):
                matches += 1
                j = lps[j - 1]
        elif j > 0:
            j = lps[j - 1]
        else:
            i += 1

    return matches


def z_function(s):
    """Compute the Z-array in O(len(s))."""
    z = [0] * len(s)
    left = 0
    right = 0

    for i in range(1, len(s)):
        if i <= right:
            z[i] = min(right - i + 1, z[i - left])

        while i + z[i] < len(s) and s[z[i]] == s[i + z[i]]:
            z[i] += 1

        if i + z[i] - 1 > right:
            left = i
            right = i + z[i] - 1

    return z


def z_count(text, pattern):
    """Count all exact occurrences using Z-function."""
    if not pattern:
        return 0

    # '\0' cannot occur in our normalized documents or patterns.
    combined = pattern + "\0" + text
    z = z_function(combined)
    offset = len(pattern) + 1
    return sum(z[i] >= len(pattern) for i in range(offset, len(combined)))


def retrieve_exact(chunks, patterns, algorithm, top_k=TOP_K):
    """Rank chunks using the same scoring strategy for both algorithms."""
    if algorithm == "kmp":
        count_matches = kmp_count
    elif algorithm == "z":
        count_matches = z_count
    else:
        raise ValueError("Algorithm must be 'kmp' or 'z'.")

    results = []
    start_time = time.perf_counter()

    for chunk in chunks:
        # Padding with spaces prevents substring matches inside other words.
        text = " " + chunk["normalized_text"].strip() + " "
        matches = {}

        for pattern in patterns:
            occurrences = count_matches(text, " " + pattern + " ")
            if occurrences:
                matches[pattern] = occurrences

        if not matches:
            continue

        # Base score based on matched patterns and frequency
        base_score = sum(
            (15 if " " in pattern else 10) + min(count, 3)
            for pattern, count in matches.items()
        )

        # Reward chunks containing multiple query patterns
        coverage_bonus = 20 * max(0, len(matches) - 1)

        score = base_score + coverage_bonus

        results.append({
            "id": chunk["id"],
            "document": chunk["document"],
            "score": score,
            "matches": matches,
            "text": chunk["text"],
        })

    results.sort(key=lambda item: (-item["score"], -len(item["matches"]), item["id"]))
    elapsed = time.perf_counter() - start_time

    return {
        "algorithm": algorithm,
        "elapsed_seconds": elapsed,
        "chunks_examined": len(chunks),
        "chunks_with_matches": len(results),
        "top_chunks": results[:top_k],
        "all_results": results,
    }


def print_results(result):
    print(f"\n===== {result['algorithm'].upper()} RESULTS =====")
    print(f"Search time: {result['elapsed_seconds']:.6f} seconds")
    print(f"Chunks examined: {result['chunks_examined']}")
    print(f"Chunks with matches: {result['chunks_with_matches']}")

    for position, chunk in enumerate(result["top_chunks"], start=1):
        print(f"\n{position}. ID: {chunk['id']}")
        print(f"   Document: {chunk['document']}")
        print(f"   Score: {chunk['score']}")
        print(f"   Matched patterns (occurrences): {chunk['matches']}")
        print(f"   Text: {chunk['text'][:600]}...")


def result_signature(result):
    return [
        (item["id"], item["score"], item["matches"])
        for item in result["all_results"]
    ]


def main():
    with INPUT_FILE.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    question = input("Question in English: ").strip()
    if not question:
        question = "What is the difference between K-Means and DBSCAN?"

    patterns = extract_patterns(question)
    print(f"\nQuestion: {question}")
    print(f"Search patterns: {patterns}")

    if not patterns:
        print("No meaningful search patterns were identified.")
        return

    kmp_result = retrieve_exact(chunks, patterns, "kmp")
    z_result = retrieve_exact(chunks, patterns, "z")

    print_results(kmp_result)
    print_results(z_result)

    identical = result_signature(kmp_result) == result_signature(z_result)
    print("\n===== COMPARISON =====")
    print(f"Identical ranked results: {identical}")

    # Save only the top results so the output is easy to inspect.
    report = {
        "question": question,
        "patterns": patterns,
        "same_results": identical,
        "kmp": {key: value for key, value in kmp_result.items() if key != "all_results"},
        "z": {key: value for key, value in z_result.items() if key != "all_results"},
    }
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(f"Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
