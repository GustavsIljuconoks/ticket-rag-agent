from __future__ import annotations

import json
import urllib.error
import urllib.request


class OllamaError(RuntimeError):
    """Raised when Ollama cannot produce embeddings or explanations."""


class OllamaClient:
    def __init__(self, base_url: str, embed_model: str, chat_model: str):
        self.base_url = base_url.rstrip("/")
        self.embed_model = embed_model
        self.chat_model = chat_model

    def embed(self, text: str) -> list[float]:
        try:
            return self._embed_with_modern_api(text)
        except OllamaError:
            return self._embed_with_legacy_api(text)

    def explain(self, prompt: str) -> dict:
        payload = {
            "model": self.chat_model,
            "stream": False,
            "format": "json",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You explain software task estimates using only the provided "
                        "historical ticket evidence and calculated statistics."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        }
        response = self._post_json("/api/chat", payload)
        content = response.get("message", {}).get("content", "")
        if not content:
            raise OllamaError("Ollama chat response did not include message content.")

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {
                "reasoning": content.strip(),
                "risks": [],
                "similar_tasks_used": [],
            }

    def _embed_with_modern_api(self, text: str) -> list[float]:
        response = self._post_json(
            "/api/embed",
            {
                "model": self.embed_model,
                "input": text,
            },
        )
        embeddings = response.get("embeddings")
        if not embeddings:
            raise OllamaError("Ollama /api/embed response did not include embeddings.")
        return [float(value) for value in embeddings[0]]

    def _embed_with_legacy_api(self, text: str) -> list[float]:
        response = self._post_json(
            "/api/embeddings",
            {
                "model": self.embed_model,
                "prompt": text,
            },
        )
        embedding = response.get("embedding")
        if not embedding:
            raise OllamaError("Ollama /api/embeddings response did not include embedding.")
        return [float(value) for value in embedding]

    def _post_json(self, path: str, payload: dict) -> dict:
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                body = response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise OllamaError(
                f"Could not reach Ollama at {self.base_url}. Is Ollama running?"
            ) from exc

        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise OllamaError("Ollama returned invalid JSON.") from exc
