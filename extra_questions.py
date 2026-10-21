import json
import time
from pathlib import Path

from stage4 import build_index
from stage5 import (
    INPUT_FILE,
    MIN_COVERAGE,
    context_coverage,
    generate_answer,
    load_model,
    retrieve_hybrid,
    select_context,
)

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "data/processed"

QUESTIONS = [
    "What is Gini impurity in decision trees?",
    "What is stratified k-fold cross-validation?",
]


def main():
    chunks = json.loads(Path(INPUT_FILE).read_text(encoding="utf-8"))
    vocabulary_frequency, prepared_chunks = build_index(chunks)
    chunk_tokens = {c["id"]: set(c["tokens"]) for c in chunks}
    generator = load_model()

    rows = []
    for number, question in enumerate(QUESTIONS, start=21):
        t0 = time.perf_counter()
        ranking, terms, _ = retrieve_hybrid(question, vocabulary_frequency, prepared_chunks, chunks)
        selected, context = select_context(ranking, terms, chunk_tokens)
        retrieval_seconds = time.perf_counter() - t0

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

        rows.append({
            "n": number,
            "question": question,
            "selected_chunks": [s["id"] for s in selected],
            "retrieval_ms": round(retrieval_seconds * 1000, 1),
            "generation_s": round(generation_seconds, 2),
            "context": context,
            "answer": answer,
        })
        print(f"\n[{number}] {question}\n  Chunks: {rows[-1]['selected_chunks']}\n  Answer: {answer}")

    (OUT_DIR / "stage5_extra_results.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    tsv = []
    for r in rows:
        answer = " ".join(r["answer"].split())
        tsv.append("\t".join([str(r["n"]), r["question"], ", ".join(r["selected_chunks"]),
                              str(r["retrieval_ms"]), f"{r['generation_s']:.2f}", answer]))
    (OUT_DIR / "anexo_a_extra.tsv").write_text("\n".join(tsv), encoding="utf-8")

    lines = []
    for r in rows:
        lines += [f"Pregunta {r['n']}: {r['question']}",
                  f"Fragmentos seleccionados: {', '.join(r['selected_chunks'])}",
                  f"Tiempo de recuperación: {r['retrieval_ms']} ms",
                  "Contexto proporcionado al modelo:", r["context"], ""]
    (OUT_DIR / "anexo_b_extra.txt").write_text("\n".join(lines), encoding="utf-8")

    print(f"\nGuardados en {OUT_DIR}: stage5_extra_results.json, anexo_a_extra.tsv, anexo_b_extra.txt")


if __name__ == "__main__":
    main()