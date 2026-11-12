from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

SERVING_MODULES = [
    "chat_search",
    "chat_search.api.app",
    "chat_search.bootstrap",
    "chat_search.cache",
    "chat_search.cli",
    "chat_search.concepts",
    "chat_search.corpus",
    "chat_search.mcp_server",
    "chat_search.models",
    "chat_search.pipeline.agent",
    "chat_search.pipeline.chat",
    "chat_search.pipeline.faq",
    "chat_search.pipeline.provider",
    "chat_search.providers.search",
    "chat_search.retrieval",
    "chat_search.runtime",
    "chat_search.spelling",
    "chat_search.state",
    "chat_search.synonyms",
    "chat_search.text",
    "chat_search.understanding",
]


def test_serving_imports_no_indexing_code():
    root = Path(__file__).resolve().parents[1]
    code = (
        "import importlib, sys\n"
        f"for name in {SERVING_MODULES!r}:\n"
        "    importlib.import_module(name)\n"
        "loaded = sorted(\n"
        "    name for name in sys.modules\n"
        "    if name.startswith(('chat_search.indexing', 'chat_search.providers.indexing'))\n"
        ")\n"
        "assert not loaded, loaded\n"
    )
    pythonpath = os.pathsep.join(filter(None, [str(root / "src"), os.environ.get("PYTHONPATH")]))
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "PYTHONPATH": pythonpath},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
