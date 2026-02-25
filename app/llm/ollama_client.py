from __future__ import annotations

import json
import urllib.request
from typing import Any, Optional

from ..settings import settings


def ollama_generate(
    prompt: str,
    *,
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
) -> str:
    """
    Uses Ollama /api/generate (simple completion style).
    """
    model = model or settings.ollama_model
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    if max_tokens is not None:
        payload["options"]["num_predict"] = int(max_tokens)

    req = urllib.request.Request(
        url=f"{settings.ollama_base_url}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return (data.get("response") or "").strip()