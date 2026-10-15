# Programma

La tua IA costruita sopra i modelli locali di LM Studio (Qwen), con quattro moduli:

| # | Modulo | Cosa fa |
|---|--------|---------|
| 1 | `programma/chatbot.py` | Chat con personalità (`SYSTEM_PROMPT`), memoria e log delle conversazioni |
| 2 | `programma/agent.py` + `tools.py` | Agente con tool: file (sandbox in `workspace/`), calcolatrice, web, ricerca nei documenti |
| 3 | `programma/rag.py` | Risponde sui tuoi documenti in `data/docs/` (embedding nomic di LM Studio) |
| 4 | `finetune/` | Prepara i dati dalle chat e addestra un adattatore LoRA |

## Avvio
```
pip install -r requirements.txt
python -m programma check      # verifica server e modello
python -m programma chat
python -m programma agent "crea workspace/ciao.txt con una poesia"
python -m programma ingest     # poi: python -m programma ask "domanda"
python -m unittest discover -s tests
```

## Collegamento: LM Studio diretto o tramite OmniRoute
Default: LM Studio su `http://localhost:1234/v1`.

Con [OmniRoute](https://github.com/diegosouzapw/OmniRoute) (gateway locale, porta 20128):
1. Avvialo (`npm install -g omniroute && omniroute`), imposta la password admin e aggiungi
   LM Studio come provider OpenAI-compatibile (`http://localhost:1234/v1`) dalla dashboard.
2. Copia la chiave API dalla dashboard e imposta:
```
export LLM_BASE_URL=http://localhost:20128/v1
export LLM_API_KEY=<chiave-omniroute>
export CHAT_MODEL=<id del modello come lo mostra OmniRoute>
```
Altre variabili: `SMALL_MODEL`, `EMBED_MODEL`, `SYSTEM_PROMPT`, `WORKSPACE`.

## Fine-tuning
```
python -m programma chat                      # le chat vengono salvate in data/chats/
python finetune/prepare_data.py               # -> finetune/train.jsonl
pip install -r finetune/requirements.txt
python finetune/train_lora.py --base <id-HF-modello-piccolo> --qlora
```
Si addestra il modello originale Hugging Face (non il GGUF di LM Studio) e serve una GPU NVIDIA.
Per 27B servono almeno ~24 GB di VRAM anche in 4 bit: comincia con un modello 4B-8B.
