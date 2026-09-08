"""A minimal LLM client, plus a recorder so the demo runs without a key.

Everything here speaks the OpenAI chat-completions API, because every provider
worth using offers an OpenAI-compatible endpoint. Switching model is a base
URL and a key, not a rewrite -- which is the whole reason to keep the wrapper
this thin.

The default is Google's Gemini through AI Studio, which has a free tier: the
point of this tutorial is not to make anyone pay to follow along.

The recorder matters more than it looks. A live demo that depends on every
attendee's API key working is a demo that fails for somebody. Responses
captured from real calls are replayed when no key is set, so the pipeline runs
on any laptop. Recordings are captured from real models and committed; none
of them are written by hand.
"""

import hashlib
import json
import os
import pathlib

RECORDINGS = pathlib.Path(__file__).parent / "recordings.json"

PROVIDERS = {
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key_env": "GEMINI_API_KEY",
        "model": "gemini-2.5-flash",
    },
    "openai": {
        "base_url": None,  # the SDK default
        "key_env": "OPENAI_API_KEY",
        "model": "gpt-4.1-mini",
    },
    "anthropic": {
        "base_url": "https://api.anthropic.com/v1/",
        "key_env": "ANTHROPIC_API_KEY",
        "model": "claude-haiku-4-5-20251001",
    },
}

DEFAULT_PROVIDER = os.environ.get("MINIFEA_PROVIDER", "gemini")


def _fingerprint(model: str, system: str, messages: list) -> str:
    """Key for a recorded response: the model and the exact conversation."""
    digest = hashlib.sha256()
    digest.update(model.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(system.encode("utf-8"))
    for message in messages:
        digest.update(b"\x00")
        digest.update(message["role"].encode("utf-8"))
        digest.update(b"\x00")
        digest.update(message["content"].encode("utf-8"))
    return digest.hexdigest()[:16]


class LLMClient:
    """Chat completions against any OpenAI-compatible endpoint.

    Args:
        provider: key into ``PROVIDERS``. Overridden by ``$MINIFEA_PROVIDER``.
        model: overrides the provider default; also ``$MINIFEA_MODEL``.
        record: append live responses to the recordings file.
        offline: replay recordings and never call out. Selected automatically
            when the provider's key is absent, so following along without a
            key works rather than erroring.
    """

    def __init__(self, provider: str | None = None, model: str | None = None,
                 record: bool = False, offline: bool | None = None):
        self.provider = provider or DEFAULT_PROVIDER
        if self.provider not in PROVIDERS:
            raise ValueError(
                f"Unknown provider {self.provider!r}; "
                f"known providers: {', '.join(PROVIDERS)}.")
        settings = PROVIDERS[self.provider]
        self.model = model or os.environ.get("MINIFEA_MODEL") or settings["model"]
        self.api_key = os.environ.get(settings["key_env"])
        self.record = record
        self.offline = (self.api_key is None) if offline is None else offline
        self._client = None

    @property
    def description(self) -> str:
        return (f"{self.model} (replayed from recordings)" if self.offline
                else f"{self.model} via {self.provider}")

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        """One completion from a single user message."""
        return self.chat(system, [{"role": "user", "content": user}], temperature)

    def chat(self, system: str, messages: list, temperature: float = 0.0) -> str:
        """One completion from a conversation.

        Needed because some of what this tutorial demonstrates only shows up
        across turns: whether a decision made in an earlier message is still
        being honoured several steps later.
        """
        key = _fingerprint(self.model, system, messages)

        if self.offline:
            recordings = self._load_recordings()
            if key not in recordings:
                raise RuntimeError(
                    f"No recorded response for this prompt under model "
                    f"{self.model!r}, and no API key in "
                    f"${PROVIDERS[self.provider]['key_env']} to make a live "
                    "call. Either set that key, or re-run the capture script "
                    "to record this prompt: python capture_recordings.py")
            return recordings[key]["response"]

        from openai import OpenAI
        if self._client is None:
            self._client = OpenAI(api_key=self.api_key,
                                  base_url=PROVIDERS[self.provider]["base_url"])
        response = self._client.chat.completions.create(
            model=self.model, temperature=temperature,
            messages=[{"role": "system", "content": system}] + list(messages))
        text = response.choices[0].message.content

        if self.record:
            recordings = self._load_recordings()
            recordings[key] = {"provider": self.provider, "model": self.model,
                               "system": system, "messages": messages,
                               "response": text}
            RECORDINGS.write_text(json.dumps(recordings, indent=2) + "\n")
        return text

    @staticmethod
    def _load_recordings() -> dict:
        if not RECORDINGS.exists():
            return {}
        return json.loads(RECORDINGS.read_text())
