
import json
import time
from pathlib import Path


class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.chunk_ids = set()


class Trie:
    def __init__(self):
        self.root = TrieNode()
        self.word_count = 0
        self.node_count = 1

    def insert(self, word, chunk_id):
        node = self.root

        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
                self.node_count += 1

            node = node.children[char]

        if not node.is_end_of_word:
            node.is_end_of_word = True
            self.word_count += 1

        node.chunk_ids.add(chunk_id)

    def search(self, word):
        node = self.root

        for char in word:
            if char not in node.children:
                return []

            node = node.children[char]

        if node.is_end_of_word:
            return sorted(node.chunk_ids)

        return []

    def autocomplete(self, prefix, limit=10):
        if not prefix or limit <= 0:
            return []

        node = self.root

        for char in prefix:
            if char not in node.children:
                return []

            node = node.children[char]

        suggestions = []
        stack = [(node, prefix)]

        while stack and len(suggestions) < limit:
            current, word = stack.pop()

            if current.is_end_of_word:
                suggestions.append(word)

            # Reverse order for alphabetical DFS
            for char in sorted(current.children, reverse=True):
                stack.append(
                    (current.children[char], word + char)
                )

        return suggestions


def build_trie(chunks):
    trie = Trie()

    for chunk in chunks:
        chunk_id = chunk["id"]

        # Avoid inserting the same word repeatedly
        # within the same chunk
        for word in set(chunk["tokens"]):
            trie.insert(word, chunk_id)

    return trie


def main():
    base_dir = Path(__file__).resolve().parent

    input_path = (
        base_dir / "data/processed/preprocessed_chunks.json"
    )

    with open(input_path, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    start_time = time.perf_counter()
    trie = build_trie(chunks)
    construction_time = time.perf_counter() - start_time

    print("\n===== TRIE CONSTRUCTION =====")
    print(f"Chunks indexed: {len(chunks)}")
    print(f"Unique words: {trie.word_count:,}")
    print(f"Trie nodes: {trie.node_count:,}")
    print(f"Construction time: {construction_time:.6f} seconds")

    print("\n===== EXACT WORD SEARCH =====")

    test_words = [
        "classification",
        "regression",
        "clustering",
        "dbscan",
        "nonexistent"
    ]

    for word in test_words:
        start_time = time.perf_counter()
        results = trie.search(word)
        search_time = time.perf_counter() - start_time

        print(f"\nWord: {word}")
        print(f"Chunks found: {len(results)}")
        print(f"First matching chunks: {results[:5]}")
        print(f"Search time: {search_time:.8f} seconds")

    print("\n===== AUTOCOMPLETE =====")

    prefixes = ["clas", "reg", "clust", "cross", "zzzz"]

    for prefix in prefixes:
        suggestions = trie.autocomplete(prefix, limit=10)

        print(f"\nPrefix: {prefix}")
        print(f"Suggestions: {suggestions}")


if __name__ == "__main__":
    main()
