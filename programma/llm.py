"""Client verso il server locale di LM Studio (API compatibile OpenAI)."""
from openai import OpenAI

from . import config


def get_client() -> OpenAI:
    return OpenAI(base_url=config.BASE_URL, api_key=config.API_KEY)


def complete(client, messages, model=None, **kw) -> str:
    r = client.chat.completions.create(
        model=model or config.CHAT_MODEL, messages=messages, **kw
    )
    return r.choices[0].message.content or ""
