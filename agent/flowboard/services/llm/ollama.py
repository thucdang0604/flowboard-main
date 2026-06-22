"""Ollama provider — local HTTP backend for Flowboard LLM features.

Ollama runs locally (default ``http://127.0.0.1:11434``) and exposes a
simple ``/api/generate`` endpoint. The model is selected by environment
variable so users can keep Flowboard provider settings simple while still
choosing their local model:

  - ``FLOWBOARD_OLLAMA_MODEL``: model name, default ``llama3.1``
  - ``OLLAMA_HOST``: base URL, default ``http://127.0.0.1:11434``

Image attachments are forwarded as base64 strings in Ollama's ``images``
field. Text-only models will reject or ignore them; multimodal models can
power Vision through the same provider.
"""
from __future__ import annotations

import base64
import logging
import os
from pathlib import Path
from typing import Optional

import httpx

from .base import LLMError
from .cli_utils import validate_attachment_paths, validate_prompt_size

logger = logging.getLogger(__name__)

_DEFAULT_HOST = "http://127.0.0.1:11434"
_DEFAULT_MODEL = "llama3.1"
_PROBE_TIMEOUT = 2.0
_MAX_ATTACHMENT_BYTES = 8 * 1024 * 1024


class OllamaProvider:
    """Conforms to ``LLMProvider``. Uses a local Ollama server."""

    name: str = "ollama"
    supports_vision: bool = True
    test_timeout_secs: float = 120.0

    def __init__(self) -> None:
        self._available: Optional[bool] = None

    @property
    def host(self) -> str:
        return (os.environ.get("OLLAMA_HOST") or _DEFAULT_HOST).rstrip("/")

    @property
    def model(self) -> str:
        return os.environ.get("FLOWBOARD_OLLAMA_MODEL") or _DEFAULT_MODEL

    def reset_cache(self) -> None:
        self._available = None

    async def is_available(self) -> bool:
        if self._available is None:
            self._available = await self._probe()
            logger.info("ollama: available=%s host=%s model=%s", self._available, self.host, self.model)
        return self._available

    async def _probe(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=_PROBE_TIMEOUT) as client:
                resp = await client.get(f"{self.host}/api/tags")
        except httpx.HTTPError as exc:
            logger.warning("ollama: probe failed: %s", exc)
            return False
        if resp.status_code != 200:
            logger.warning("ollama: probe HTTP %s", resp.status_code)
            return False
        try:
            data = resp.json()
        except ValueError:
            logger.warning("ollama: probe returned non-JSON")
            return False
        models = data.get("models") if isinstance(data, dict) else None
        if not isinstance(models, list):
            return True
        wanted = self.model
        for row in models:
            if not isinstance(row, dict):
                continue
            name = row.get("name")
            model = row.get("model")
            if name == wanted or model == wanted:
                return True
        logger.warning("ollama: model %s not found in /api/tags", wanted)
        return False

    async def run(
        self,
        user_prompt: str,
        *,
        system_prompt: Optional[str] = None,
        attachments: Optional[list[str]] = None,
        timeout: float = 90.0,
    ) -> str:
        try:
            validate_prompt_size(user_prompt)
            if system_prompt:
                validate_prompt_size(system_prompt)
            validate_attachment_paths(attachments)
        except ValueError as exc:
            raise LLMError(f"Invalid input: {exc}") from exc

        payload: dict = {
            "model": self.model,
            "prompt": user_prompt,
            "stream": False,
        }
        if system_prompt:
            payload["system"] = system_prompt
        if attachments:
            payload["images"] = [_image_b64(path) for path in attachments]

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(f"{self.host}/api/generate", json=payload)
        except httpx.TimeoutException as exc:
            raise LLMError(f"ollama timed out after {timeout}s") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"ollama transport error: {exc}") from exc

        if resp.status_code != 200:
            raise LLMError(f"ollama HTTP {resp.status_code}: {_safe_error_message(resp)}")
        try:
            data = resp.json()
        except ValueError as exc:
            raise LLMError("ollama response was not JSON") from exc
        text = data.get("response") if isinstance(data, dict) else None
        if not isinstance(text, str):
            raise LLMError(f"ollama response missing text: {data!r:.200}")
        return text.strip()


def _image_b64(path: str) -> str:
    p = Path(path)
    size = p.stat().st_size
    if size > _MAX_ATTACHMENT_BYTES:
        raise LLMError(
            f"attachment too large for ollama: {size // (1024 * 1024)}MB > 8MB cap"
        )
    return base64.b64encode(p.read_bytes()).decode("ascii")


def _safe_error_message(resp: httpx.Response) -> str:
    try:
        data = resp.json()
    except ValueError:
        return resp.text[:200] if resp.text else "(non-JSON body)"
    if isinstance(data, dict):
        for key in ("error", "message"):
            val = data.get(key)
            if isinstance(val, str):
                return val[:200]
    return "(unrecognised body)"
