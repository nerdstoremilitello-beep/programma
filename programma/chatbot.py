"""Punto 1: chatbot con personalità, memoria della conversazione e log salvato."""
import json
import time

from . import config
from .llm import get_client


class Chatbot:
    def __init__(self, client=None, system_prompt=None, model=None, log=True):
        self.client = client or get_client()
        self.model = model or config.CHAT_MODEL
        self.messages = [{"role": "system", "content": system_prompt or config.SYSTEM_PROMPT}]
        self.log_path = None
        if log:
            config.CHATS_DIR.mkdir(parents=True, exist_ok=True)
            self.log_path = config.CHATS_DIR / f"chat-{int(time.time())}.jsonl"

    def say(self, text: str) -> str:
        self.messages.append({"role": "user", "content": text})
        r = self.client.chat.completions.create(model=self.model, messages=self.messages)
        reply = r.choices[0].message.content or ""
        self.messages.append({"role": "assistant", "content": reply})
        self._log()
        return reply

    def _log(self):
        if self.log_path:
            # una riga = una conversazione completa (formato usato dal fine-tuning)
            self.log_path.write_text(json.dumps({"messages": self.messages}, ensure_ascii=False) + "\n")


def repl():
    bot = Chatbot()
    print(f"Chat con {bot.model} (Ctrl+C o 'esci' per uscire)")
    try:
        while (t := input("\ntu> ").strip()) not in ("esci", "exit"):
            if t:
                print("ia>", bot.say(t))
    except (KeyboardInterrupt, EOFError):
        print()
