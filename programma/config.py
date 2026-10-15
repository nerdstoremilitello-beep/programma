"""Configurazione (sovrascrivibile con variabili d'ambiente)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Endpoint OpenAI-compatibile. Default: LM Studio diretto.
# Con OmniRoute: LLM_BASE_URL=http://localhost:20128/v1 e LLM_API_KEY=<chiave della dashboard>
BASE_URL = os.getenv("LLM_BASE_URL") or os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
API_KEY = os.getenv("LLM_API_KEY") or os.getenv("LMSTUDIO_API_KEY", "lm-studio")

CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen/qwen3.8-27b")
SMALL_MODEL = os.getenv("SMALL_MODEL", "google/gemma-4-e4b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-nomic-embed-text-v1.5")

DOCS_DIR = Path(os.getenv("DOCS_DIR", ROOT / "data" / "docs"))
INDEX_PATH = Path(os.getenv("INDEX_PATH", ROOT / "data" / "index.json"))
CHATS_DIR = Path(os.getenv("CHATS_DIR", ROOT / "data" / "chats"))
WORKSPACE = Path(os.getenv("WORKSPACE", ROOT / "workspace"))

SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "Sei un assistente utile, preciso e conciso. Rispondi in italiano "
    "a meno che l'utente non usi un'altra lingua.",
)
