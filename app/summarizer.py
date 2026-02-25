import re
import json
import urllib.request
from .settings import settings

def heuristic_summary(text: str, max_sentences: int = 3) -> str:
    """
    Deterministic summary:
    - take first docstring-like paragraph or first ~2-3 sentences
    - for code, try to detect 'does X, then Y' patterns from comments
    """
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return ""

    m = re.search(r'("""|\'\'\')(.{20,600}?)(\1)', text, flags=re.DOTALL)
    if m:
        block = re.sub(r"\s+", " ", m.group(2)).strip()
        return _take_sentences(block, max_sentences)

    return _take_sentences(cleaned, max_sentences)

def _take_sentences(s: str, n: int) -> str:
    parts = re.split(r"(?<=[.!?])\s+", s)
    parts = [p.strip() for p in parts if p.strip()]
    return " ".join(parts[:n])[:500]

def summarize(text: str) -> str:
    mode = settings.summarizer_mode.lower()
    if mode == "ollama":
        return ollama_summary(text)
    elif mode == "llamacpp":
        return heuristic_summary(text)
    return heuristic_summary(text)

def ollama_summary(text: str) -> str:
    """
    Calls Ollama (requires: `ollama serve` and `ollama pull <model>`)
    """ 
    prompt = (
        "Summarize the following technical snippet in 2-4 sentences, focusing on what it does, "
        "key inputs/outputs, and important behavior. Be precise.\n\n"
        f"SNIPPET:\n{text[:6000]}"
    )

    payload = {
        "model": settings.ollama_model,
        "prompt": prompt,
        "stream": False,
    }

    req = urllib.request.Request(
        url=f"{settings.ollama_base_url}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            out = (data.get("response") or "").strip()
            return out[:600]
    except Exception:
        # fall back if Ollama not running
        return heuristic_summary(text)