import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from programma import agent, config, rag, tools  # noqa: E402
from programma.chatbot import Chatbot  # noqa: E402


def msg(content="", tool_calls=None):
    return NS(choices=[NS(message=NS(content=content, tool_calls=tool_calls))])


class FakeChat:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []
        self.completions = NS(create=self._create)

    def _create(self, **kw):
        self.calls.append({**kw, "messages": list(kw["messages"])})
        return self.replies.pop(0)


class FakeClient:
    def __init__(self, replies=()):
        self.chat = FakeChat(replies)
        self.embeddings = NS(create=self._emb)

    def _emb(self, model, input):  # embedding finto: conta le parole chiave
        keys = ["gatto", "cane", "pizza"]
        return NS(data=[NS(embedding=[float(t.lower().count(k)) + 0.01 for k in keys]) for t in input])


class T(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        config.WORKSPACE = self.tmp / "ws"; config.WORKSPACE.mkdir()
        config.CHATS_DIR = self.tmp / "chats"

    def test_chatbot_memory_and_log(self):
        c = FakeClient([msg("ciao"), msg("bene")])
        b = Chatbot(client=c)
        self.assertEqual(b.say("hey"), "ciao"); b.say("come va")
        self.assertEqual(len(c.chat.calls[1]["messages"]), 4)  # system+user+assistant+user
        self.assertEqual(len(json.loads(b.log_path.read_text())["messages"]), 5)

    def test_agent_tool_loop(self):
        call = NS(id="1", function=NS(name="calculator", arguments='{"expression": "2+3*4"}'))
        c = FakeClient([msg("", [call]), msg("Fa 14")])
        self.assertEqual(agent.run_agent("calcola", client=c, verbose=False), "Fa 14")
        self.assertEqual(c.chat.calls[1]["messages"][-1]["content"], "14")

    def test_agent_tool_error_returns_to_model(self):
        call = NS(id="1", function=NS(name="read_file", arguments='{"path": "../../etc/passwd"}'))
        c = FakeClient([msg("", [call]), msg("ok")])
        agent.run_agent("x", client=c, verbose=False)
        self.assertIn("ERRORE", c.chat.calls[1]["messages"][-1]["content"])

    def test_sandbox_and_calc(self):
        tools.write_file("a/b.txt", "hi"); self.assertEqual(tools.read_file("a/b.txt"), "hi")
        with self.assertRaises(ValueError): tools.read_file("../x")
        with self.assertRaises(ValueError): tools.calculator("__import__('os')")

    def test_rag(self):
        d = self.tmp / "docs"; d.mkdir()
        (d / "gatti.txt").write_text("il gatto dorme. gatto gatto")
        (d / "cibo.txt").write_text("la pizza è buona. pizza pizza")
        idx = self.tmp / "i.json"; c = FakeClient([msg("risposta [gatti.txt]")])
        self.assertEqual(rag.ingest(c, d, idx), 2)
        top = rag.Index(c, idx).search("gatto", k=1)
        self.assertEqual(top[0][0], "gatti.txt")
        self.assertEqual(rag.ask("gatto?", c, rag.Index(c, idx)), "risposta [gatti.txt]")

    def test_prepare_data(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "finetune"))
        import prepare_data
        self.assertEqual(len(prepare_data.convert({"prompt": "a", "response": "b"})["messages"]), 2)


if __name__ == "__main__":
    unittest.main()
