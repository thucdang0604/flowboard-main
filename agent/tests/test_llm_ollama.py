"""Tests for the Ollama LLM provider.

No real Ollama server is contacted; httpx.AsyncClient is replaced with a
small async fake.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from flowboard.services.llm.base import LLMError
from flowboard.services.llm.ollama import OllamaProvider


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = json.dumps(self._payload)

    def json(self) -> dict:
        return self._payload


class _FakeClient:
    last_post_payload: dict | None = None

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def get(self, url: str):
        return _FakeResponse(200, {"models": [{"name": "llama3.1"}]})

    async def post(self, url: str, json: dict):
        _FakeClient.last_post_payload = json
        return _FakeResponse(200, {"response": " local answer "})


@pytest.mark.asyncio
async def test_ollama_available_when_model_is_present(monkeypatch):
    monkeypatch.setattr("flowboard.services.llm.ollama.httpx.AsyncClient", _FakeClient)
    monkeypatch.delenv("FLOWBOARD_OLLAMA_MODEL", raising=False)

    provider = OllamaProvider()
    assert await provider.is_available() is True


@pytest.mark.asyncio
async def test_ollama_run_posts_generate_payload(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("flowboard.services.llm.ollama.httpx.AsyncClient", _FakeClient)
    img = tmp_path / "ref.png"
    img.write_bytes(b"fake")

    provider = OllamaProvider()
    out = await provider.run(
        "write prompt",
        system_prompt="be concise",
        attachments=[str(img)],
    )

    assert out == "local answer"
    payload = _FakeClient.last_post_payload
    assert payload is not None
    assert payload["model"] == "llama3.1"
    assert payload["prompt"] == "write prompt"
    assert payload["system"] == "be concise"
    assert payload["stream"] is False
    assert payload["images"] == ["ZmFrZQ=="]


@pytest.mark.asyncio
async def test_ollama_missing_model_is_unavailable(monkeypatch):
    class MissingModelClient(_FakeClient):
        async def get(self, url: str):
            return _FakeResponse(200, {"models": [{"name": "other"}]})

    monkeypatch.setattr(
        "flowboard.services.llm.ollama.httpx.AsyncClient",
        MissingModelClient,
    )

    provider = OllamaProvider()
    assert await provider.is_available() is False


@pytest.mark.asyncio
async def test_ollama_http_error_raises_llm_error(monkeypatch):
    class ErrorClient(_FakeClient):
        async def post(self, url: str, json: dict):
            return _FakeResponse(500, {"error": "model failed"})

    monkeypatch.setattr("flowboard.services.llm.ollama.httpx.AsyncClient", ErrorClient)

    provider = OllamaProvider()
    with pytest.raises(LLMError, match="model failed"):
        await provider.run("x")
