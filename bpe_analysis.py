import json
from pathlib import Path
from statistics import mean

from transformers import AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent
chunks = json.loads((BASE_DIR / "data/processed/chunks.json").read_text(encoding="utf-8"))
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

word_counts = [c["word_count"] for c in chunks]
bpe_counts = []
vocab = set()
for c in chunks:
    ids = tok(c["text"], add_special_tokens=False)["input_ids"]
    bpe_counts.append(len(ids))
    vocab.update(ids)

print(f"Fragmentos: {len(chunks)}")
print(f"Palabras totales: {sum(word_counts):,}")
print(f"Tokens BPE totales: {sum(bpe_counts):,}")
print(f"Tokens BPE únicos: {len(vocab):,}")
print(f"Tokens por palabra: {sum(bpe_counts) / sum(word_counts):.2f}")
print(f"Tokens BPE por fragmento: promedio {mean(bpe_counts):.1f}, mín {min(bpe_counts)}, máx {max(bpe_counts)}")

for word in ["classification", "classificaton", "crosvalidation", "dbscan", "silhouette", "siluette"]:
    print(word, "->", tok.tokenize(word))