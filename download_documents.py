
import requests
from bs4 import BeautifulSoup
from pathlib import Path

# Official scikit-learn documentation
documents = {
    "decision_trees": "https://scikit-learn.org/stable/modules/tree.html",
    "clustering": "https://scikit-learn.org/stable/modules/clustering.html",
    "cross_validation": "https://scikit-learn.org/stable/modules/cross_validation.html",
    "model_evaluation": "https://scikit-learn.org/stable/modules/model_evaluation.html"
}

output_dir = Path("data/raw")
output_dir.mkdir(parents=True, exist_ok=True)

for name, url in documents.items():

    print(f"Downloading: {name}...")

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, "html.parser")

    article = soup.select_one("article.bd-article")

    if article is None:
        raise RuntimeError(
            f"Could not find article content: {url}"
        )


    for element in article.select(
        "script, style, nav, aside, footer, .headerlink, .copybtn"
    ):
        element.decompose()

    text = article.get_text(separator="\n", strip=True)

    file_path = output_dir / f"{name}.txt"

    file_path.write_text(text, encoding="utf-8")

    print(f"Saved: {file_path}")
    print(f"Words: {len(text.split()):,}")
    print(f"Characters: {len(text):,}")
    print("-" * 40)

print("All documents downloaded!")
