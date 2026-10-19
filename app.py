import json
import queue
import re
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

from stage3 import extract_patterns, retrieve_exact
from stage4 import (
    build_index,
    combine_rankings,
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
    select_context,
)
from trie_index import build_trie

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "data/processed/preprocessed_chunks.json"

TOP_K = 5
MAX_SUGGESTIONS = 8
PREVIEW_CHARACTERS = 500


METHODS = {
    "Exacta - KMP": "kmp",
    "Exacta - Z-function": "z",
    "Aproximada - Levenshtein": "levenshtein",
    "Aproximada - Jaccard": "jaccard",
    "Hibrida - Levenshtein + Jaccard": "hybrid",
}

EXAMPLE_QUESTION = "What is the difference between K-Means and DBSCAN?"


class RetrievalApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Recuperacion de informacion + modelo de lenguaje")
        self.geometry("1000x700")
        self.minsize(820, 560)

        self.messages = queue.Queue()
        self.chunks = []
        self.prepared_chunks = []
        self.vocabulary_frequency = {}
        self.chunk_tokens = {}
        self.trie = None
        self.generator = None
        self.busy = False

        self._build_widgets()
        self._set_busy(True, "Cargando fragmentos y construyendo el Trie...")

        threading.Thread(target=self._load_data, daemon=True).start()
        self.after(100, self._process_messages)



    def _build_widgets(self):
        question_frame = ttk.LabelFrame(self, text="Pregunta")
        question_frame.pack(fill="x", padx=10, pady=(10, 5))
        question_frame.columnconfigure(0, weight=1)

        self.question_variable = tk.StringVar(value=EXAMPLE_QUESTION)
        self.question_entry = ttk.Entry(
            question_frame,
            textvariable=self.question_variable,
            font=("Segoe UI", 11),
        )
        self.question_entry.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        self.search_button = ttk.Button(
            question_frame, text="Buscar", command=self.on_search
        )
        self.search_button.grid(row=0, column=1, padx=5, pady=5)

        self.suggestion_list = tk.Listbox(
            question_frame, height=MAX_SUGGESTIONS, font=("Segoe UI", 10)
        )
        self.suggestion_list.grid(row=1, column=0, sticky="ew", padx=5)
        self.suggestion_list.grid_remove()

        hint = ttk.Label(
            question_frame,
            text="Autocompletado con Trie: escribe un prefijo, usa la flecha abajo y "
                 "Enter o doble clic para completar la palabra.",
            foreground="#555555",
        )
        hint.grid(row=2, column=0, columnspan=2, sticky="w", padx=5, pady=(2, 5))

        self.question_entry.bind("<KeyRelease>", self.on_key_release)
        self.question_entry.bind("<Down>", self.on_focus_suggestions)
        self.question_entry.bind("<Return>", lambda event: self.on_search())
        self.question_entry.bind("<Escape>", lambda event: self.hide_suggestions())
        self.suggestion_list.bind("<Return>", self.on_accept_suggestion)
        self.suggestion_list.bind("<Double-Button-1>", self.on_accept_suggestion)
        self.suggestion_list.bind("<Escape>", lambda event: self.hide_suggestions())

        options_frame = ttk.Frame(self)
        options_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(options_frame, text="Metodo de busqueda:").pack(side="left")

        self.method_variable = tk.StringVar(value="Hibrida - Levenshtein + Jaccard")
        method_box = ttk.Combobox(
            options_frame,
            textvariable=self.method_variable,
            values=list(METHODS),
            state="readonly",
            width=32,
        )
        method_box.pack(side="left", padx=8)

        self.use_model_variable = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            options_frame,
            text="Generar respuesta con el modelo de lenguaje",
            variable=self.use_model_variable,
        ).pack(side="left", padx=8)

        results_frame = ttk.LabelFrame(self, text="Fragmentos recuperados")
        results_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.results_text = ScrolledText(
            results_frame, wrap="word", font=("Consolas", 10), state="disabled"
        )
        self.results_text.pack(fill="both", expand=True, padx=5, pady=5)

        answer_frame = ttk.LabelFrame(self, text="Respuesta del modelo")
        answer_frame.pack(fill="both", padx=10, pady=5)

        self.answer_text = ScrolledText(
            answer_frame, wrap="word", height=7, font=("Segoe UI", 10), state="disabled"
        )
        self.answer_text.pack(fill="both", expand=True, padx=5, pady=5)

        self.status_variable = tk.StringVar(value="Iniciando...")
        ttk.Label(
            self, textvariable=self.status_variable, relief="sunken", anchor="w"
        ).pack(fill="x", side="bottom")

    def _set_busy(self, busy, status=None):
        self.busy = busy
        self.search_button.state(["disabled"] if busy else ["!disabled"])
        if status is not None:
            self.status_variable.set(status)

    def _write(self, widget, text):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")



    def current_prefix(self):
        """Word being typed right before the cursor."""
        text = self.question_variable.get()[: self.question_entry.index("insert")]
        match = re.search(r"[A-Za-z0-9]+$", text)
        return match.group(0).lower() if match else ""

    def on_key_release(self, event):
        if event.keysym in ("Down", "Up", "Return", "Escape", "Left", "Right"):
            return

        prefix = self.current_prefix()
        if self.trie is None or len(prefix) < 2:
            self.hide_suggestions()
            return

        suggestions = self.trie.autocomplete(prefix, limit=MAX_SUGGESTIONS)
        suggestions = [word for word in suggestions if word != prefix]

        if not suggestions:
            self.hide_suggestions()
            return

        self.suggestion_list.delete(0, "end")
        for word in sorted(suggestions):
            self.suggestion_list.insert("end", word)

        self.suggestion_list.configure(height=min(len(suggestions), MAX_SUGGESTIONS))
        self.suggestion_list.grid()

    def hide_suggestions(self):
        self.suggestion_list.grid_remove()
        self.question_entry.focus_set()

    def on_focus_suggestions(self, event):
        if self.suggestion_list.winfo_ismapped():
            self.suggestion_list.focus_set()
            self.suggestion_list.selection_clear(0, "end")
            self.suggestion_list.selection_set(0)
            self.suggestion_list.activate(0)
        return "break"

    def on_accept_suggestion(self, event):
        selection = self.suggestion_list.curselection()
        if not selection:
            return

        word = self.suggestion_list.get(selection[0])
        prefix = self.current_prefix()
        cursor = self.question_entry.index("insert")
        text = self.question_variable.get()

        completed = text[: cursor - len(prefix)] + word + " " + text[cursor:]
        self.question_variable.set(completed)
        self.hide_suggestions()
        self.question_entry.icursor(cursor - len(prefix) + len(word) + 1)



    def _load_data(self):
        try:
            with INPUT_FILE.open("r", encoding="utf-8") as file:
                chunks = json.load(file)

            vocabulary_frequency, prepared_chunks = build_index(chunks)
            trie = build_trie(chunks)

            self.messages.put(("loaded", {
                "chunks": chunks,
                "vocabulary_frequency": vocabulary_frequency,
                "prepared_chunks": prepared_chunks,
                "chunk_tokens": {c["id"]: set(c["tokens"]) for c in chunks},
                "trie": trie,
            }))
        except Exception as error:
            self.messages.put(("error", f"No se pudieron cargar los datos: {error}"))


    def retrieve(self, question, method):
        """Return (ranking, context_terms, description, seconds)."""
        start = time.perf_counter()

        if method in ("kmp", "z"):
            patterns = extract_patterns(question)
            ranking = (
                retrieve_exact(self.chunks, patterns, method)["all_results"]
                if patterns else []
            )
            seconds = time.perf_counter() - start
            description = f"Patrones buscados: {patterns}"
            return ranking, tokenize_question(question), description, seconds

        query_terms = tokenize_question(question)
        if not query_terms:
            return [], [], "No se identificaron terminos de busqueda.", 0.0

        corrections = correct_query(query_terms, self.vocabulary_frequency)
        corrected_terms = list(dict.fromkeys(c["corrected"] for c in corrections))

        if method == "jaccard":
            ranking = search_jaccard(self.prepared_chunks, query_terms)
        else:
            levenshtein_ranking = search_levenshtein(self.prepared_chunks, corrections)
            if method == "levenshtein":
                ranking = levenshtein_ranking
            else:
                jaccard_ranking = search_jaccard(self.prepared_chunks, corrected_terms)
                ranking = combine_rankings(
                    levenshtein_ranking, jaccard_ranking, self.chunks
                )

        seconds = time.perf_counter() - start

        applied = [
            f"{c['original']} -> {c['corrected']}"
            for c in corrections
            if c["original"] != c["corrected"]
        ]
        description = f"Terminos: {query_terms}"
        if applied:
            description += f"\nCorrecciones (Levenshtein): {applied}"

        return ranking, corrected_terms, description, seconds

    def format_results(self, question, method_label, description, ranking, seconds):
        lines = [
            f"Pregunta: {question}",
            f"Metodo: {method_label}",
            description,
            f"Fragmentos examinados: {len(self.chunks)}",
            f"Fragmentos con coincidencias: {len(ranking)}",
            f"Tiempo de busqueda: {seconds * 1000:.2f} ms",
            "",
        ]

        if not ranking:
            lines.append("No se encontraron fragmentos relacionados con la pregunta.")
            return "\n".join(lines)

        lines.append(f"===== TOP {min(TOP_K, len(ranking))} FRAGMENTOS =====")

        for position, result in enumerate(ranking[:TOP_K], start=1):
            lines.append("")
            lines.append(f"{position}. {result['id']}  |  {result['document']}")
            lines.append(f"   Puntuacion: {result['score']}")

            if "matches" in result:
                lines.append(f"   Coincidencias: {result['matches']}")
            if "levenshtein_score" in result:
                lines.append(
                    f"   Levenshtein: {result['levenshtein_score']:.2f}"
                    f"  |  Jaccard: {result['jaccard_score']:.2f}"
                )

            lines.append(f"   Texto: {result['text'][:PREVIEW_CHARACTERS]}...")

        return "\n".join(lines)

    def on_search(self):
        if self.busy or not self.chunks:
            return

        question = self.question_variable.get().strip()
        if not question:
            self.status_variable.set("Escribe una pregunta.")
            return

        self.hide_suggestions()
        self._write(self.answer_text, "")
        self._set_busy(True, "Buscando...")

        method = METHODS[self.method_variable.get()]
        want_answer = self.use_model_variable.get()

        threading.Thread(
            target=self._search_worker,
            args=(question, method, self.method_variable.get(), want_answer),
            daemon=True,
        ).start()

    def _search_worker(self, question, method, method_label, want_answer):
        try:
            ranking, context_terms, description, seconds = self.retrieve(question, method)

            self.messages.put((
                "results",
                self.format_results(question, method_label, description, ranking, seconds),
            ))

            if not want_answer:
                self.messages.put((
                    "status",
                    f"Busqueda terminada en {seconds * 1000:.2f} ms "
                    f"({len(ranking)} fragmentos con coincidencias).",
                ))
                self.messages.put(("done", None))
                return

            selected, context = select_context(ranking, context_terms, self.chunk_tokens)
            coverage = context_coverage(context_terms, selected, self.chunk_tokens)

            if not selected:
                self.messages.put(("answer", "(no se recuperaron fragmentos)"))
                self.messages.put(("done", None))
                return

            if coverage < MIN_COVERAGE:
                self.messages.put((
                    "answer",
                    "The provided context does not contain this information. "
                    f"(rechazada: cobertura {coverage:.2f} < {MIN_COVERAGE})",
                ))
                self.messages.put(("done", None))
                return

            if self.generator is None:
                self.messages.put((
                    "status",
                    "Cargando el modelo de lenguaje (puede tardar la primera vez)...",
                ))
                self.generator = load_model()

            self.messages.put(("status", "Generando la respuesta..."))
            start = time.perf_counter()
            answer = generate_answer(self.generator, question, context)
            generation_seconds = time.perf_counter() - start

            context_ids = ", ".join(s["id"] for s in selected)
            self.messages.put((
                "answer",
                f"{answer}\n\n"
                f"Contexto usado: {context_ids}\n"
                f"Cobertura de terminos: {coverage:.2f}",
            ))
            self.messages.put((
                "status",
                f"Busqueda: {seconds * 1000:.2f} ms  |  "
                f"Generacion: {generation_seconds:.2f} s",
            ))
            self.messages.put(("done", None))

        except Exception as error:
            self.messages.put(("error", str(error)))

    def _process_messages(self):
        while True:
            try:
                kind, payload = self.messages.get_nowait()
            except queue.Empty:
                break

            if kind == "loaded":
                self.chunks = payload["chunks"]
                self.vocabulary_frequency = payload["vocabulary_frequency"]
                self.prepared_chunks = payload["prepared_chunks"]
                self.chunk_tokens = payload["chunk_tokens"]
                self.trie = payload["trie"]
                self._set_busy(
                    False,
                    f"Listo: {len(self.chunks)} fragmentos, "
                    f"{self.trie.word_count:,} palabras en el Trie.",
                )
            elif kind == "results":
                self._write(self.results_text, payload)
            elif kind == "answer":
                self._write(self.answer_text, payload)
            elif kind == "status":
                self.status_variable.set(payload)
            elif kind == "done":
                self._set_busy(False)
            elif kind == "error":
                self._write(self.answer_text, f"Error: {payload}")
                self._set_busy(False, "Ocurrio un error.")

        self.after(100, self._process_messages)


if __name__ == "__main__":
    RetrievalApp().mainloop()
