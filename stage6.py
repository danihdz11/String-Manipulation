
import argparse
import csv
import json
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path

from stage3 import extract_patterns, retrieve_exact
from stage4 import (
    build_index,
    correct_query,
    search_jaccard,
    search_levenshtein,
    tokenize_question,
)
from stage5 import (
    MIN_COVERAGE,
    context_coverage,
    generate_answer,
    load_model,
    retrieve_hybrid,
    select_context,
)

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "data/processed/preprocessed_chunks.json"
OUT_DIR = BASE_DIR / "data/processed/stage6"

REPEATS = 5
SCALES = [0.25, 0.5, 1, 2, 4, 8]
METHODS = ["kmp", "z_function", "levenshtein", "jaccard", "hybrid"]
QUALITY_METHODS = ["kmp", "levenshtein", "jaccard", "hybrid"]   # Z-function gives the same ranking as KMP

DT = ["decision tree", "decision trees"]





BANK = [
    ("C01", "clean", "What is the difference between K-Means and DBSCAN?", [["k means", "kmeans"], ["dbscan"]], True),
    ("C02", "clean", "How do decision trees work for classification?", [DT, ["classification"]], True),
    ("C03", "clean", "What is the difference between precision and recall?", [["precision"], ["recall"]], True),
    ("C04", "clean", "What is cross-validation used for?", [["cross validation"], ["generalization", "generalize", "performance", "estimate"]], True),
    ("C05", "clean", "What is overfitting?", [["overfitting"]], False),
    ("C06", "clean", "How does DBSCAN find clusters?", [["dbscan"], ["density"]], False),
    ("C07", "clean", "What is the F1 score?", [["f1"]], True),
    ("C08", "clean", "What are the advantages of decision trees?", [DT, ["advantages"]], False),
    ("C09", "clean", "How does k-fold cross-validation split the data?", [["k fold", "kfold"], ["folds", "fold"]], True),
    ("C10", "clean", "What is the silhouette coefficient?", [["silhouette"], ["coefficient"]], True),
    ("C11", "clean", "What is a confusion matrix?", [["confusion matrix"]], False),
    ("C12", "clean", "How can decision trees be pruned?", [["pruning", "pruned", "prune"], ["tree", "trees"]], True),
    ("C13", "clean", "What is the Rand index?", [["rand index"]], False),
    ("C14", "clean", "What is the ROC curve?", [["roc"], ["curve"]], False),
    ("C15", "clean", "What is mean squared error?", [["mean squared error"]], False),
    ("C16", "clean", "What is stratified k-fold cross-validation?", [["stratified"], ["fold", "folds"]], False),
    ("C17", "clean", "What is hierarchical clustering?", [["hierarchical"], ["clustering"]], False),
    ("C18", "clean", "What is the Davies-Bouldin index?", [["davies bouldin"]], False),
    ("C19", "clean", "What is Gini impurity?", [["gini"]], False),
    ("C20", "clean", "What is balanced accuracy?", [["balanced accuracy"]], False),

    ("T01", "typo", "What is the difference between K-Meens and DBSCN?", [["k means", "kmeans"], ["dbscan"]], True),
    ("T02", "typo", "How do decison trees work for classificaton?", [DT, ["classification"]], True),
    ("T03", "typo", "What is precison and recal?", [["precision"], ["recall"]], True),
    ("T04", "typo", "What is the siluette coeficient?", [["silhouette"], ["coefficient"]], True),
    ("T05", "typo", "What is a confuson matrix?", [["confusion matrix"]], False),
    ("T06", "typo", "How can decision trees be prunned?", [["pruning", "pruned", "prune"], ["tree", "trees"]], False),
    ("T07", "typo", "What is stratifed k-fold cross-validation?", [["stratified"], ["fold", "folds"]], False),
    ("T08", "typo", "What is hierarchial clustering?", [["hierarchical"], ["clustering"]], False),
    ("T09", "typo", "What is crosvalidation used for?", [["cross validation"], ["generalization", "generalize", "performance", "estimate"]], True),

    ("P01", "paraphrase", "Which algorithm groups points that are packed closely together in dense regions?", [["dbscan"]], True),
    ("P02", "paraphrase", "How do I measure how many of the actual positive samples a classifier finds?", [["recall"]], True),
    ("P03", "paraphrase", "How can I estimate how well my model will work on new unseen data?", [["cross validation"]], True),
    ("P04", "paraphrase", "Which technique removes branches of a tree to reduce its complexity?", [["pruning", "pruned", "prune"], ["tree", "trees"]], True),
    ("P05", "paraphrase", "How can I judge cluster quality when there are no ground truth labels?", [["silhouette", "davies bouldin", "calinski harabasz"]], True),
    ("P06", "paraphrase", "Which metric combines precision and recall into a single number?", [["f1"]], False),
    ("P07", "paraphrase", "Why does a model memorize noise in the training data?", [["overfitting"]], False),
    ("P08", "paraphrase", "Which table counts true positives and false negatives?", [["confusion matrix"]], False),

    ("O01", "offtopic", "How do I train a transformer language model?", None, True),
    ("O02", "offtopic", "What is the capital of Germany?", None, True),
    ("O03", "offtopic", "Which neural network architecture is best for images?", None, False),
]





def padded(text):
    return " " + text.strip() + " "


def satisfies(padded_text, must):
    return all(any((" " + term + " ") in padded_text for term in group) for group in must)


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


def time_call(fn, question, repeats):
    """Run fn(question) several times. Returns (mean_ms, median_ms, last_ranking)."""
    times = []
    ranking = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        ranking = fn(question)
        times.append(time.perf_counter() - t0)
    return statistics.mean(times) * 1000, statistics.median(times) * 1000, ranking


def build_methods(chunks):
    """Build every retrieval method over a collection. Index construction is NOT timed."""
    vocabulary_frequency, prepared_chunks = build_index(chunks)

    def make_exact(algorithm):
        def run(question):
            patterns = extract_patterns(question)
            if not patterns:
                return []
            return retrieve_exact(chunks, patterns, algorithm)["all_results"]
        return run

    def run_levenshtein(question):
        terms = tokenize_question(question)
        if not terms:
            return []
        return search_levenshtein(prepared_chunks, correct_query(terms, vocabulary_frequency))

    def run_jaccard(question):
        terms = tokenize_question(question)
        if not terms:
            return []
        return search_jaccard(prepared_chunks, terms)

    def run_hybrid(question):
        return retrieve_hybrid(question, vocabulary_frequency, prepared_chunks, chunks)[0]

    methods = {
        "kmp": make_exact("kmp"),
        "z_function": make_exact("z"),
        "levenshtein": run_levenshtein,
        "jaccard": run_jaccard,
        "hybrid": run_hybrid,
    }
    return methods, vocabulary_frequency, prepared_chunks


def evaluate(ranking_ids, relevant_ids):
    """hit@3, hit@5, precision@5 and reciprocal rank of one ranking."""
    rr = 0.0
    for position, chunk_id in enumerate(ranking_ids, start=1):
        if chunk_id in relevant_ids:
            rr = 1.0 / position
            break
    return {
        "hit3": int(any(c in relevant_ids for c in ranking_ids[:3])),
        "hit5": int(any(c in relevant_ids for c in ranking_ids[:5])),
        "prec5": sum(c in relevant_ids for c in ranking_ids[:5]) / 5,
        "rr": rr,
    }


def mean_or_blank(values):
    return round(statistics.mean(values), 3) if values else ""







def run_quality_and_timing(chunks, repeats):
    methods, vocabulary_frequency, _ = build_methods(chunks)
    norm_by_id = {c["id"]: padded(c["normalized_text"]) for c in chunks}

    rows = []
    rankings = {}       # (question_id, method) -> ranking list of dicts
    label_rows = []
    z_mismatch = 0

    for number, (qid, category, question, must, _) in enumerate(BANK, start=1):
        relevant_ids = None
        if must is not None:
            relevant_ids = {cid for cid, text in norm_by_id.items() if satisfies(text, must)}
            label_rows.append({
                "id": qid, "question": question,
                "relevant_total": len(relevant_ids),
                "relevant_ids": ";".join(sorted(relevant_ids)[:25]),
            })
            if not relevant_ids:
                print(f"WARNING: no chunk satisfies the relevance labels of {qid}; excluded from quality metrics.")

        print(f"[{number}/{len(BANK)}] {qid} ({category}) {question}")
        kmp_ids = None
        for name in METHODS:
            mean_ms, median_ms, ranking = time_call(methods[name], question, repeats)
            ids = [r["id"] for r in ranking]
            rankings[(qid, name)] = ranking
            if name == "kmp":
                kmp_ids = ids
            if name == "z_function" and ids != kmp_ids:
                z_mismatch += 1

            row = {
                "id": qid, "category": category, "method": name,
                "mean_ms": round(mean_ms, 3), "median_ms": round(median_ms, 3),
                "examined": len(chunks), "retrieved": len(ranking),
                "top1": ids[0] if ids else "",
            }
            if relevant_ids:
                row.update(evaluate(ids, relevant_ids))
            rows.append(row)

    write_csv(OUT_DIR / "relevance_labels.csv", label_rows)
    write_csv(OUT_DIR / "per_question.csv", rows)
    return rows, rankings, vocabulary_frequency, z_mismatch


def summarize(rows, z_mismatch):
    by_method = defaultdict(list)
    by_cat = defaultdict(list)
    for r in rows:
        by_method[r["method"]].append(r)
        by_cat[(r["category"], r["method"])].append(r)

    summary = []
    for name in METHODS:
        group = by_method[name]
        scored = [r for r in group if "hit3" in r]
        summary.append({
            "method": name,
            "mean_ms": round(statistics.mean(r["mean_ms"] for r in group), 2),
            "median_ms": round(statistics.median(r["median_ms"] for r in group), 2),
            "avg_retrieved": round(statistics.mean(r["retrieved"] for r in group), 1),
            "examined": group[0]["examined"],
            "hit@3": mean_or_blank([r["hit3"] for r in scored]),
            "hit@5": mean_or_blank([r["hit5"] for r in scored]),
            "precision@5": mean_or_blank([r["prec5"] for r in scored]),
            "MRR": mean_or_blank([r["rr"] for r in scored]),
            "questions_scored": len(scored),
        })

    category_rows = []
    for (category, name), group in sorted(by_cat.items()):
        scored = [r for r in group if "hit3" in r]
        category_rows.append({
            "category": category, "method": name,
            "mean_ms": round(statistics.mean(r["mean_ms"] for r in group), 2),
            "hit@3": mean_or_blank([r["hit3"] for r in scored]),
            "MRR": mean_or_blank([r["rr"] for r in scored]),
            "avg_retrieved": round(statistics.mean(r["retrieved"] for r in group), 1),
        })

    write_csv(OUT_DIR / "summary_by_method.csv", summary)
    write_csv(OUT_DIR / "summary_by_category.csv", category_rows)

    print("\n===== SUMMARY BY METHOD =====")
    for s in summary:
        print(s)
    print(f"Z-function ranking differed from KMP in {z_mismatch} questions.")
    return summary, category_rows


def overlap_cases(rankings):
    """Top-5 overlap between exact (KMP) and hybrid retrieval, to find similar / different cases."""
    rows = []
    for qid, category, question, must, _ in BANK:
        if must is None:
            continue
        a = [r["id"] for r in rankings[(qid, "kmp")][:5]]
        b = [r["id"] for r in rankings[(qid, "hybrid")][:5]]
        overlap = len(set(a) & set(b)) / 5
        rows.append({
            "id": qid, "category": category, "question": question,
            "top5_overlap": overlap,
            "kmp_top5": ";".join(a), "hybrid_top5": ";".join(b),
            "case": "similar" if overlap >= 0.6 else ("different" if overlap <= 0.2 else "partial"),
        })
    write_csv(OUT_DIR / "exact_vs_hybrid_overlap.csv", rows)



def scale_chunks(chunks, scale):
    target = max(1, int(len(chunks) * scale))
    if target <= len(chunks):
        indexes = sorted(random.Random(0).sample(range(len(chunks)), target))
        return [chunks[i] for i in indexes]
    scaled, copy = [], 0
    while len(scaled) < target:
        for chunk in chunks:
            if len(scaled) >= target:
                break
            scaled.append(chunk if copy == 0 else {**chunk, "id": f"{chunk['id']}__copy{copy}"})
        copy += 1
    return scaled


def scaling_experiment(chunks, repeats):
    questions = [b[2] for b in BANK if b[3] is not None][::5][:8]
    rows = []
    for scale in SCALES:
        scaled = scale_chunks(chunks, scale)
        methods, _, _ = build_methods(scaled)
        print(f"Scaling x{scale}: {len(scaled)} chunks")
        for name in METHODS:
            times = [time_call(methods[name], q, min(repeats, 3))[0] for q in questions]
            rows.append({
                "scale": scale, "chunks": len(scaled), "method": name,
                "mean_ms": round(statistics.mean(times), 2),
            })
    write_csv(OUT_DIR / "scaling.csv", rows)
    return rows





def search_terms(method, question, vocabulary_frequency):
    """Terms used for the context coverage step: exact search has no spelling correction."""
    terms = tokenize_question(question)
    if method == "kmp":
        return terms
    return list(dict.fromkeys(c["corrected"] for c in correct_query(terms, vocabulary_frequency)))


def context_experiment(chunks, rankings, vocabulary_frequency):
    chunk_tokens = {c["id"]: set(c["tokens"]) for c in chunks}
    norm_by_id = {c["id"]: c["normalized_text"] for c in chunks}
    stats = defaultdict(lambda: {"ok": [], "words": []})

    for qid, category, question, must, _ in BANK:
        if must is None:
            continue
        for method in ("kmp", "hybrid"):
            ranking = rankings[(qid, method)]
            terms = search_terms(method, question, vocabulary_frequency)
            selected, _ = select_context(ranking, terms, chunk_tokens)
            variants = {
                "top1": [r["id"] for r in ranking[:1]],
                "top3_raw": [r["id"] for r in ranking[:3]],
                "stage5_selection": [s["id"] for s in selected],
            }
            for variant, ids in variants.items():
                text = padded(" ".join(norm_by_id[i] for i in ids))
                stats[(method, variant)]["ok"].append(int(bool(ids) and satisfies(text, must)))
                stats[(method, variant)]["words"].append(sum(len(norm_by_id[i].split()) for i in ids))

    rows = [{
        "method": method, "context_variant": variant,
        "answer_terms_in_context": round(statistics.mean(v["ok"]), 3),
        "avg_context_words": round(statistics.mean(v["words"]), 1),
        "questions": len(v["ok"]),
    } for (method, variant), v in sorted(stats.items())]
    write_csv(OUT_DIR / "context_selection.csv", rows)
    return rows





def llm_comparison(chunks, rankings, vocabulary_frequency):
    chunk_tokens = {c["id"]: set(c["tokens"]) for c in chunks}
    norm_by_id = {c["id"]: padded(c["normalized_text"]) for c in chunks}
    generator = load_model()
    rows = []

    for qid, category, question, must, use_llm in BANK:
        if not use_llm:
            continue
        for method in ("kmp", "hybrid"):
            terms = search_terms(method, question, vocabulary_frequency)
            selected, context = select_context(rankings[(qid, method)], terms, chunk_tokens)
            coverage = context_coverage(terms, selected, chunk_tokens)
            generation_seconds = 0.0

            if not selected:
                answer = "(no chunks retrieved)"
            elif coverage < MIN_COVERAGE:
                answer = "The provided context does not contain this information. (rejected: low coverage)"
            else:
                t0 = time.perf_counter()
                answer = generate_answer(generator, question, context)
                generation_seconds = time.perf_counter() - t0

            ids = [s["id"] for s in selected]
            has_answer = ""
            if must is not None:
                has_answer = int(bool(ids) and satisfies(padded(" ".join(norm_by_id[i] for i in ids)), must))
            rows.append({
                "id": qid, "category": category, "method": method, "question": question,
                "selected_chunks": ";".join(ids), "coverage": round(coverage, 2),
                "answer_terms_in_context": has_answer,
                "generation_s": round(generation_seconds, 2),
                "answer": answer,
                "supported_by_context": "",   # fill in by hand: yes / partial / no
            })
            print(f"\n[{qid} | {method}] {question}\n  Chunks: {ids}\n  Answer: {answer[:300]}")

    write_csv(OUT_DIR / "llm_comparison.csv", rows)
    with (OUT_DIR / "llm_comparison.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    return rows





def make_plots(summary, category_rows, scaling_rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping plots (pip install matplotlib).")
        return

    # Average search time per method.
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([s["method"] for s in summary], [s["mean_ms"] for s in summary])
    ax.set_ylabel("Mean search time (ms)")
    ax.set_title("Average search time per method")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "plot_time_per_method.png", dpi=150)
    plt.close(fig)

    # Time vs collection size.
    if scaling_rows:
        fig, ax = plt.subplots(figsize=(7, 4))
        for name in METHODS:
            points = [(r["chunks"], r["mean_ms"]) for r in scaling_rows if r["method"] == name]
            ax.plot([p[0] for p in points], [p[1] for p in points], marker="o", label=name)
        ax.set_xlabel("Number of chunks")
        ax.set_ylabel("Mean search time (ms)")
        ax.set_title("Search time vs collection size")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT_DIR / "plot_time_vs_size.png", dpi=150)
        plt.close(fig)

    # hit@3 by category.
    categories = [c for c in ("clean", "typo", "paraphrase") if any(r["category"] == c for r in category_rows)]
    width = 0.2
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, name in enumerate(QUALITY_METHODS):
        values = []
        for category in categories:
            match = [r["hit@3"] for r in category_rows if r["category"] == category and r["method"] == name]
            values.append(match[0] if match and match[0] != "" else 0)
        ax.bar([x + i * width for x in range(len(categories))], values, width, label=name)
    ax.set_xticks([x + 1.5 * width for x in range(len(categories))])
    ax.set_xticklabels(categories)
    ax.set_ylabel("hit@3")
    ax.set_ylim(0, 1.05)
    ax.set_title("Relevant chunk in the top 3, by question type")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "plot_hit3_by_category.png", dpi=150)
    plt.close(fig)
    print(f"Plots saved in {OUT_DIR}")




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-llm", action="store_true", help="skip the language-model comparison")
    parser.add_argument("--no-scaling", action="store_true", help="skip the collection-size experiment")
    parser.add_argument("--repeats", type=int, default=REPEATS, help="timing repetitions per query")
    args = parser.parse_args()

    with INPUT_FILE.open("r", encoding="utf-8") as f:
        chunks = json.load(f)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loaded {len(chunks)} chunks; {len(BANK)} test questions; {args.repeats} repetitions.")

    rows, rankings, vocabulary_frequency, z_mismatch = run_quality_and_timing(chunks, args.repeats)
    summary, category_rows = summarize(rows, z_mismatch)
    overlap_cases(rankings)

    context_rows = context_experiment(chunks, rankings, vocabulary_frequency)
    print("\n===== CONTEXT SELECTION =====")
    for r in context_rows:
        print(r)

    scaling_rows = [] if args.no_scaling else scaling_experiment(chunks, args.repeats)
    make_plots(summary, category_rows, scaling_rows)

    if not args.no_llm:
        llm_comparison(chunks, rankings, vocabulary_frequency)

    print(f"\nDone. Everything is in {OUT_DIR}")


if __name__ == "__main__":
    main()


    