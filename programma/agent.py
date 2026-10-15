"""Punto 2: agente che usa tool (function calling) in un ciclo."""
import json

from . import config
from .llm import get_client
from .tools import REGISTRY


def run_agent(task: str, client=None, model=None, max_steps: int = 8, tools=None, verbose=True) -> str:
    client = client or get_client()
    names = tools or list(REGISTRY)
    specs = [REGISTRY[n][1] for n in names]
    msgs = [
        {"role": "system", "content": config.SYSTEM_PROMPT + " Hai dei tool: usali quando servono "
         "e rispondi quando hai abbastanza informazioni."},
        {"role": "user", "content": task},
    ]
    for _ in range(max_steps):
        r = client.chat.completions.create(model=model or config.CHAT_MODEL, messages=msgs, tools=specs)
        m = r.choices[0].message
        if not m.tool_calls:
            return m.content or ""
        msgs.append({"role": "assistant", "content": m.content or "", "tool_calls": [
            {"id": c.id, "type": "function",
             "function": {"name": c.function.name, "arguments": c.function.arguments}}
            for c in m.tool_calls]})
        for c in m.tool_calls:
            try:
                if c.function.name not in names:
                    raise ValueError("tool sconosciuto")
                out = REGISTRY[c.function.name][0](**json.loads(c.function.arguments or "{}"))
            except Exception as e:  # l'errore torna al modello, che può correggersi
                out = f"ERRORE: {e}"
            if verbose:
                print(f"  [tool] {c.function.name}({c.function.arguments}) -> {str(out)[:100]!r}")
            msgs.append({"role": "tool", "tool_call_id": c.id, "content": str(out)})
    return "Raggiunto il limite di passi senza una risposta finale."
