"""Exercise 1.1 -- your first LLM API call.

No tools, no loop. One question in, one answer out. The point is to prove your
environment works and to see the shape of the request: a list of messages, each
with a `role` and `content`.

Run:  python ex1_first_call.py
"""

import os

from openai import OpenAI

# Any OpenAI-compatible endpoint works here -- the workshop server, api.openai.com,
# a local vLLM/Ollama server. Only the base URL, key, and model name change.
client = OpenAI(base_url=os.environ["LLM_BASE_URL"], api_key=os.environ["LLM_API_KEY"])
MODEL = os.environ.get("LLM_MODEL", "gemini-2.5-flash")

# "system" sets standing instructions; "user" is the turn you are taking now.
# The model sees the whole list every call -- it has no memory of its own.
response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "You are a concise computational mechanics assistant."},
        {
            "role": "user",
            "content": (
                "A steel cantilever beam is 0.2 m long with a 0.02 x 0.02 m square "
                "cross-section, loaded by 1 kN at the free end. Roughly how far does "
                "the tip deflect? Show your reasoning in three sentences or less."
            ),
        },
    ],
)

# `choices` is a list because the API can return several samples; we asked for one.
print(response.choices[0].message.content)
