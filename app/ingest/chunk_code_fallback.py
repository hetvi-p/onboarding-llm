import re
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class CodeSymbolChunk:
    chunk_type: str
    symbol: Optional[str]
    signature: Optional[str]
    content: str

DEF_RE = re.compile(r"(?m)^\s*(export\s+)?(async\s+)?(function|class)\s+([A-Za-z0-9_]+)\b.*?$")

def chunk_code_fallback(code: str, *, path: str = "") -> List[CodeSymbolChunk]:
    """
    Heuristic: detect 'function|class Name' blocks for JS/TS/etc.
    If not found, store module chunk.
    """
    code = code.replace("\r\n", "\n")
    matches = list(DEF_RE.finditer(code))
    if not matches:
        return [CodeSymbolChunk("module", None, None, f"PATH: {path}\nKIND: module\n\n{code}".strip())]

    chunks: list[CodeSymbolChunk] = []
    for i, m in enumerate(matches):
        kind = m.group(3)
        name = m.group(4)
        start = m.start()
        end = matches[i+1].start() if i+1 < len(matches) else len(code)
        body = code[start:end].strip()
        sig = body.splitlines()[0][:200]
        chunks.append(CodeSymbolChunk(kind if kind in ("function","class") else "block", name, sig, f"PATH: {path}\nKIND: {kind}\nSIGNATURE: {sig}\n\n{body}"))
    return chunks
