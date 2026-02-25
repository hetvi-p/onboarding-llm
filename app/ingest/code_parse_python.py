import ast
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class CodeSymbolChunk:
    chunk_type: str     # function|class|module|block
    symbol: Optional[str]
    signature: Optional[str]
    content: str

def parse_python_symbols(code: str, *, path: str = "") -> List[CodeSymbolChunk]:
    """
    Extract:
    - top-level functions
    - classes + their methods (methods become separate chunks)
    Falls back to module chunk if AST fails
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return [CodeSymbolChunk("module", None, None, _wrap_module(path, code))]

    lines = code.splitlines()
    chunks: list[CodeSymbolChunk] = []

    def get_src(node: ast.AST) -> str:
        if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
            return ""
        start = max(node.lineno - 1, 0)
        end = min(node.end_lineno, len(lines))
        return "\n".join(lines[start:end]).strip()

    def fn_signature(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
        args = []
        for a in fn.args.args:
            args.append(a.arg)
        if fn.args.vararg:
            args.append("*" + fn.args.vararg.arg)
        for a in fn.args.kwonlyargs:
            args.append(a.arg)
        if fn.args.kwarg:
            args.append("**" + fn.args.kwarg.arg)
        return f"{fn.name}({', '.join(args)})"

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            src = get_src(node)
            sig = fn_signature(node)
            chunks.append(CodeSymbolChunk("function", node.name, sig, _wrap_symbol(path, "function", sig, src)))
        elif isinstance(node, ast.ClassDef):
            class_src = get_src(node)
            chunks.append(CodeSymbolChunk("class", node.name, f"class {node.name}", _wrap_symbol(path, "class", f"class {node.name}", class_src)))

            # methods as separate chunks
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    msrc = get_src(sub)
                    msig = fn_signature(sub)
                    symbol = f"{node.name}.{sub.name}"
                    chunks.append(CodeSymbolChunk("function", symbol, msig, _wrap_symbol(path, "method", f"{symbol}{msig[len(sub.name):]}", msrc)))

    if not chunks:
        chunks.append(CodeSymbolChunk("module", None, None, _wrap_module(path, code)))
    return chunks

def _wrap_symbol(path: str, kind: str, sig: str, body: str) -> str:
    return f"PATH: {path}\nKIND: {kind}\nSIGNATURE: {sig}\n\n{body}".strip()

def _wrap_module(path: str, code: str) -> str:
    return f"PATH: {path}\nKIND: module\n\n{code}".strip()