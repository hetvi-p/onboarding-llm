import re
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class DocChunk:
    heading_chain: list[str]
    content: str
    chunk_type: str = "section"

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)\s*$", re.MULTILINE)

def chunk_markdown(md: str, *, max_chars: int = 8000) -> List[DocChunk]:
    """
    - split by heading boundaries
    - keep heading chain metadata
    - preserve step-ish blocks by not splitting lists aggressively
    """
    md = md.replace("\r\n", "\n")
    md = _strip_boilerplate(md)

    matches = list(_HEADING_RE.finditer(md))
    if not matches:
        return [DocChunk(heading_chain=[], content=md.strip(), chunk_type="section")]

    chunks: List[DocChunk] = []
    stack: list[tuple[int, str]] = []  # (level, title)

    for i, m in enumerate(matches):
        level = len(m.group(1))
        title = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md)
        body = md[start:end].strip()

        # update heading stack
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        heading_chain = [t for _, t in stack]

        if not body:
            continue

        # if section is huge, do a soft split by blank lines, but keep lists intact
        if len(body) > max_chars:
            chunks.extend(_split_large_section(heading_chain, body, max_chars=max_chars))
        else:
            chunks.append(DocChunk(heading_chain=heading_chain, content=_format_section(heading_chain, body)))

    # drop tiny chunks
    chunks = [c for c in chunks if len(c.content.strip()) >= 80]
    return chunks

def _format_section(chain: list[str], body: str) -> str:
    header = " > ".join(chain)
    return f"SECTION: {header}\n\n{body}".strip()

def _split_large_section(chain: list[str], body: str, *, max_chars: int) -> List[DocChunk]:
    paras = body.split("\n\n")
    out: list[DocChunk] = []
    buf: list[str] = []

    def flush():
        if buf:
            text = "\n\n".join(buf).strip()
            if text:
                out.append(DocChunk(chain, _format_section(chain, text)))

    for p in paras:
        p = p.strip()
        if not p:
            continue
        # keep lists together: if paragraph is list-heavy, treat it as atomic
        is_listy = p.startswith("- ") or re.match(r"^\d+\.\s", p) is not None
        if is_listy and len(p) > max_chars:
            # very large list: just take as one chunk anyway
            flush()
            out.append(DocChunk(chain, _format_section(chain, p)))
            continue

        candidate = ("\n\n".join(buf + [p])).strip()
        if len(candidate) <= max_chars:
            buf.append(p)
        else:
            flush()
            buf = [p]

    flush()
    return out

def _strip_boilerplate(md: str) -> str:
    # TODO: expand this with project-specific boilerplate patterns.
    md = re.sub(r"(?im)^\s*Last updated:.*$", "", md)
    md = re.sub(r"(?im)^\s*Table of contents\s*$", "", md)
    return md