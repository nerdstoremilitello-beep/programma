"""Tool per l'agente. File limitati alla cartella WORKSPACE (sandbox)."""
import ast
import operator as op
import urllib.request
from pathlib import Path

from . import config

MAX_OUT = 8000


def _safe(path: str) -> Path:
    root = Path(config.WORKSPACE).resolve()
    p = (root / path).resolve()
    if p != root and root not in p.parents:
        raise ValueError("percorso fuori dal workspace")
    return p


def list_dir(path: str = ".") -> str:
    return "\n".join(sorted(x.name + ("/" if x.is_dir() else "") for x in _safe(path).iterdir())) or "(vuota)"


def read_file(path: str) -> str:
    return _safe(path).read_text(errors="ignore")[:MAX_OUT]


def write_file(path: str, content: str) -> str:
    p = _safe(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"scritto {path} ({len(content)} caratteri)"


_OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
        ast.Pow: op.pow, ast.Mod: op.mod, ast.USub: op.neg, ast.FloorDiv: op.floordiv}


def calculator(expression: str) -> str:
    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        raise ValueError("espressione non valida")
    return str(ev(ast.parse(expression, mode="eval").body))


def fetch_url(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        raise ValueError("solo http/https")
    req = urllib.request.Request(url, headers={"User-Agent": "programma/0.1"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return r.read(200_000).decode("utf-8", "ignore")[:MAX_OUT]


def search_docs(query: str) -> str:
    from .rag import Index
    return "\n\n".join(f"[{s}]\n{c}" for s, c in Index().search(query))


def _schema(name, desc, props, required):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": required}}}


S = {"type": "string"}
REGISTRY = {
    "list_dir": (list_dir, _schema("list_dir", "Elenca i file del workspace", {"path": S}, [])),
    "read_file": (read_file, _schema("read_file", "Legge un file del workspace", {"path": S}, ["path"])),
    "write_file": (write_file, _schema("write_file", "Scrive un file nel workspace",
                                       {"path": S, "content": S}, ["path", "content"])),
    "calculator": (calculator, _schema("calculator", "Calcola un'espressione matematica",
                                       {"expression": S}, ["expression"])),
    "fetch_url": (fetch_url, _schema("fetch_url", "Scarica il testo di una pagina web", {"url": S}, ["url"])),
    "search_docs": (search_docs, _schema("search_docs", "Cerca nei documenti indicizzati dell'utente",
                                         {"query": S}, ["query"])),
}
