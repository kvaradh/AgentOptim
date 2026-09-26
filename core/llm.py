"""Optional OpenAI-compatible JSON agent client.

The cached replay and deterministic agents never need a key. A live demo can
set AGENT_API_KEY, AGENT_MODEL, and optionally AGENT_API_URL.
"""

import json
import os

import requests


DEFAULT_API_URL = "https://api.openai.com/v1/chat/completions"


class OpenAICompatibleAgentClient:
    def __init__(self, *, api_key: str, model: str, api_url: str):
        self.api_key = api_key
        self.model = model
        self.api_url = api_url

    def __call__(self, system_prompt: str, payload: str) -> dict:
        response = requests.post(
            self.api_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "temperature": 0.2,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": payload},
                ],
            },
            timeout=45,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        value = json.loads(content)
        if not isinstance(value, dict):
            raise ValueError("Agent response must be a JSON object")
        return value


def client_from_env():
    api_key = os.getenv("AGENT_API_KEY")
    model = os.getenv("AGENT_MODEL")
    if not api_key or not model:
        return None
    return OpenAICompatibleAgentClient(
        api_key=api_key,
        model=model,
        api_url=os.getenv("AGENT_API_URL", DEFAULT_API_URL),
    )
