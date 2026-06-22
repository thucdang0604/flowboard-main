"""Ollama provider — local HTTP backend for Flowboard LLM features.

Ollama runs locally (default ``http://127.0.0.1:11434``) and exposes a
simple ``/api/generate`` endpoint. The model is selected by environment
variable so users can keep Flowboard provider settings simple while still
choosing their local model:

  - ``FLOWBOARD_OLLAMA_MODEL``: fallback text model, default ``llama3.1``
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
from typing import Any, Optional

import httpx

from .base import LLMError
from .cli_utils import validate_attachment_paths, validate_prompt_size
from . import secrets

logger = logging.getLogger(__name__)

_DEFAULT_HOST = "http://127.0.0.1:11434"
_DEFAULT_MODEL = "llama3.1"
_PROBE_TIMEOUT = 2.0
_MAX_ATTACHMENT_BYTES = 8 * 1024 * 1024
_VISION_HINTS = (
    "vision",
    "visual",
    "multimodal",
    "projector",
    "clip",
    "llava",
    "bakllava",
    "moondream",
    "minicpm",
    "qwen2-vl",
    "qwen2.5-vl",
    "qwen2vl",
    "qwen2.5vl",
    "gemma3",
    "mllama",
    "llama3.2-vision",
    "llama4",
    "granite-vision",
)


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
        return self.text_model

    @property
    def text_model(self) -> str:
        saved = secrets.get_provider_settings("ollama").get("textModel")
        if isinstance(saved, str) and saved:
            return saved
        return os.environ.get("FLOWBOARD_OLLAMA_MODEL") or _DEFAULT_MODEL

    @property
    def vision_model(self) -> Optional[str]:
        saved = secrets.get_provider_settings("ollama").get("visionModel")
        if isinstance(saved, str) and saved:
            return saved
        return None

    def reset_cache(self) -> None:
        self._available = None

    async def is_available(self) -> bool:
        if self._available is None:
            self._available = await self._probe()
            logger.info(
                "ollama: available=%s host=%s text_model=%s",
                self._available,
                self.host,
                self.text_model,
            )
        return self._available

    async def _probe(self) -> bool:
        models = await self.list_models(include_details=False)
        if models is None:
            return False
        return len(models) > 0

    async def list_models(self, *, include_details: bool = True) -> Optional[list[dict[str, Any]]]:
        """Return local Ollama models, annotated with best-effort vision support.

        ``None`` means Ollama itself is unreachable. Empty list means the
        server is reachable but has no models.
        """
        try:
            async with httpx.AsyncClient(timeout=_PROBE_TIMEOUT) as client:
                resp = await client.get(f"{self.host}/api/tags")
        except httpx.HTTPError as exc:
            logger.warning("ollama: probe failed: %s", exc)
            return None
        if resp.status_code != 200:
            logger.warning("ollama: probe HTTP %s", resp.status_code)
            return None
        try:
            data = resp.json()
        except ValueError:
            logger.warning("ollama: probe returned non-JSON")
            return None
        models = data.get("models") if isinstance(data, dict) else None
        if not isinstance(models, list):
            return []
        out: list[dict[str, Any]] = []
        for row in models:
            if not isinstance(row, dict):
                continue
            name = row.get("name")
            if not isinstance(name, str) or not name:
                model = row.get("model")
                name = model if isinstance(model, str) else None
            if not name:
                continue
            details = row.get("details") if isinstance(row.get("details"), dict) else {}
            show: dict[str, Any] = {}
            if include_details:
                show = await self._show_model(name)
            vision = _looks_vision_capable(name, row, show)
            out.append({
                "name": name,
                "size": row.get("size") if isinstance(row.get("size"), int) else None,
                "modifiedAt": row.get("modified_at") if isinstance(row.get("modified_at"), str) else None,
                "family": _string_or_none(details.get("family") or show.get("family")),
                "families": _string_list(details.get("families") or show.get("families")),
                "parameterSize": _string_or_none(details.get("parameter_size")),
                "quantizationLevel": _string_or_none(details.get("quantization_level")),
                "vision": vision,
            })
        return out

    async def _show_model(self, name: str) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=_PROBE_TIMEOUT) as client:
                resp = await client.post(f"{self.host}/api/show", json={"model": name})
        except httpx.HTTPError:
            return {}
        if resp.status_code != 200:
            return {}
        try:
            data = resp.json()
        except ValueError:
            return {}
        return data if isinstance(data, dict) else {}

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

        model = self._resolve_model(wants_vision=bool(attachments))
        payload: dict = {
            "model": model,
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

    def _resolve_model(self, *, wants_vision: bool) -> str:
        if wants_vision:
            model = self.vision_model
            if not model:
                raise LLMError(
                    "Ollama Vision model is not configured. Pick a vision-capable "
                    "local model in Settings or use a cloud/CLI vision provider."
                )
            return model
        return self.text_model


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


def _string_or_none(value: Any) -> Optional[str]:
    return value if isinstance(value, str) and value else None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [v for v in value if isinstance(v, str) and v]


def _looks_vision_capable(name: str, tag_row: dict[str, Any], show: dict[str, Any]) -> bool:
    # Newer Ollama exposes explicit capabilities in /api/show.
    caps = show.get("capabilities")
    if isinstance(caps, list) and any(str(c).lower() == "vision" for c in caps):
        return True

    haystack = " ".join([
        name,
        _flatten_for_detection(tag_row),
        _flatten_for_detection(show),
    ]).lower()
    return any(hint in haystack for hint in _VISION_HINTS)


def _flatten_for_detection(value: Any) -> str:
    if isinstance(value, dict):
        return " ".join(
            f"{k} {_flatten_for_detection(v)}"
            for k, v in value.items()
            if k not in {"license", "modelfile", "template", "parameters"}
        )
    if isinstance(value, list):
        return " ".join(_flatten_for_detection(v) for v in value)
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return ""
