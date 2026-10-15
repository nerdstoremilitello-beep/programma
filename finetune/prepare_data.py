"""Converte le chat salvate (data/chats/*.jsonl) e/o file di esempi in finetune/train.jsonl.

Formato di ogni riga: {"messages": [{"role": "...", "content": "..."}, ...]}
Esempi manuali: file .jsonl con {"prompt": "...", "response": "..."} (opzionale "system").
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def convert(obj):
    if "messages" in obj:
        return obj
    if "prompt" in obj and "response" in obj:
        msgs = ([{"role": "system", "content": obj["system"]}] if obj.get("system") else [])
        msgs += [{"role": "user", "content": obj["prompt"]},
                 {"role": "assistant", "content": obj["response"]}]
        return {"messages": msgs}
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="*", default=[str(ROOT / "data" / "chats")])
    ap.add_argument("--out", default=str(ROOT / "finetune" / "train.jsonl"))
    a = ap.parse_args()
    rows = []
    for inp in a.inputs:
        p = Path(inp)
        for f in ([p] if p.is_file() else sorted(p.glob("*.jsonl"))):
            for line in f.read_text().splitlines():
                if line.strip() and (row := convert(json.loads(line))):
                    if any(m["role"] == "assistant" for m in row["messages"]):
                        rows.append(row)
    Path(a.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"{len(rows)} esempi scritti in {a.out}")


if __name__ == "__main__":
    main()
