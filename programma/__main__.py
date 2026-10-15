import argparse

from . import config


def main():
    ap = argparse.ArgumentParser(prog="programma")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("chat", help="chatbot interattivo")
    a = sub.add_parser("agent", help="agente con tool"); a.add_argument("task")
    i = sub.add_parser("ingest", help="indicizza i documenti di data/docs"); i.add_argument("--dir")
    q = sub.add_parser("ask", help="domanda sui tuoi documenti"); q.add_argument("question")
    sub.add_parser("check", help="verifica la connessione al server")
    args = ap.parse_args()

    if args.cmd == "chat":
        from .chatbot import repl; repl()
    elif args.cmd == "agent":
        from .agent import run_agent; print(run_agent(args.task))
    elif args.cmd == "ingest":
        from .rag import ingest; print(f"{ingest(docs_dir=args.dir)} blocchi indicizzati")
    elif args.cmd == "ask":
        from .rag import ask; print(ask(args.question))
    elif args.cmd == "check":
        from .llm import get_client
        ids = [m.id for m in get_client().models.list().data]
        print(f"Server: {config.BASE_URL}\nModelli: {ids}")
        print("OK" if config.CHAT_MODEL in ids else f"ATTENZIONE: {config.CHAT_MODEL} non trovato")


if __name__ == "__main__":
    main()
