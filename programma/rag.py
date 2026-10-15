"""Punto 3: IA sui tuoi documenti (RAG) con il modello di embedding di LM Studio."""
import json
from pathlib import Path

import numpy as np

from . import config
from .llm import complete, get_client

EXTS = {".txt", ".md", ".py", ".csv", ".json", ".html", ".rst"}


def chunk_text(text: str, size: int = 800, overlap: int = 100) -> list[str]:
    text = text.strip()
    if not text:
        return []
    step = size - overlap
    return [text[i : i + size] for i in range(0, len(text), step)]


def embed(client, texts: list[str], prefix: str, batch: int = 32) -> np.ndarray:
    # nomic-embed richiede i prefissi "search_document:" / "search_query:"
    out = []
    for i in range(0, len(texts), batch):
        part = [prefix + t for t in texts[i : i + batch]]
        r = client.embeddings.create(model=config.EMBED_MODEL, input=part)
        out += [d.embedding for d in r.data]
    return np.array(out, dtype=np.float32)


def ingest(client=None, docs_dir: Path | None = None, index_path: Path | None = None) -> int:
    client = client or get_client()
    docs_dir = Path(docs_dir or config.DOCS_DIR)
    index_path = Path(index_path or config.INDEX_PATH)
    chunks, sources = [], []
    for p in sorted(docs_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in EXTS:
            for c in chunk_text(p.read_text(errors="ignore")):
                chunks.append(c)
                sources.append(str(p.relative_to(docs_dir)))
    if not chunks:
        raise SystemExit(f"Nessun documento trovato in {docs_dir}")
    vecs = embed(client, chunks, "search_document: ")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps({"chunks": chunks, "sources": sources, "vectors": vecs.tolist()}))
    return len(chunks)


class Index:
    def __init__(self, client=None, index_path: Path | None = None):
        self.client = client or get_client()
        data = json.loads(Path(index_path or config.INDEX_PATH).read_text())
        self.chunks, self.sources = data["chunks"], data["sources"]
        v = np.array(data["vectors"], dtype=np.float32)
        self.vectors = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)

    def search(self, query: str, k: int = 4):
        q = embed(self.client, [query], "search_query: ")[0]
        q = q / max(np.linalg.norm(q), 1e-9)
        top = np.argsort(-(self.vectors @ q))[:k]
        return [(self.sources[i], self.chunks[i]) for i in top]


def ask(question: str, client=None, index: Index | None = None, k: int = 4) -> str:
    client = client or get_client()
    index = index or Index(client)
    ctx = "\n\n".join(f"[{s}]\n{c}" for s, c in index.search(question, k))
    msgs = [
        {"role": "system", "content": config.SYSTEM_PROMPT + " Rispondi usando SOLO il contesto "
         "fornito; se non basta, dillo. Cita le fonti tra [parentesi quadre]."},
        {"role": "user", "content": f"Contesto:\n{ctx}\n\nDomanda: {question}"},
    ]
    return complete(client, msgs)
