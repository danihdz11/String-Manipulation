import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RESULTS = BASE_DIR / "data/processed/stage5_results.json"
OUTPUT = BASE_DIR / "data/processed/anexo_b_contextos.txt"


def main():
    rows = json.loads(RESULTS.read_text(encoding="utf-8"))
    if len(rows) != 20:
        print(f"OJO: el archivo tiene {len(rows)} filas, no 20.")

    lines = ["Anexo B. Fragmentos seleccionados como contexto para el modelo de lenguaje", ""]
    for row in rows:
        ids = ", ".join(c["id"] for c in row["selected_chunks"]) or "ninguno"
        lines.append(f"Pregunta {row['n']}: {row['question']}")
        lines.append(f"Fragmentos seleccionados: {ids}")
        lines.append(f"Tiempo de recuperación: {row['retrieval_seconds'] * 1000:.1f} ms")
        if not row["selected_chunks"]:
            lines.append("Contexto: no se recuperaron fragmentos.")
        else:
            if row["generation_seconds"] == 0:
                lines.append("Nota: este contexto no se envió al modelo (cobertura menor al 50%).")
            lines.append("Contexto proporcionado al modelo:")
            lines.append(row["context"])
        lines.append("")

    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Guardado en {OUTPUT}")


if __name__ == "__main__":
    main()