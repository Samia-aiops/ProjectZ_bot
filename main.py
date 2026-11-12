"""Compatibility launcher.

The application lives in ``src/chat_search``. Prefer ``python -m chat_search``.
This file keeps existing ``python main.py ...`` commands working from a source
checkout (and inside the image) while callers migrate.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from chat_search.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
