"""Stream Airflow task logs to stdout.

Airflow writes task output to files under ``/opt/airflow/logs``; the scheduler
process captures the task subprocess stdout into those files, so it never
reaches the container console. This sidecar follows new task logs and prints
them, which makes ``docker compose logs -f`` show DAG runs as they happen.

Scheduler and DAG-processor logs are skipped because ``airflow standalone``
already prints those.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

LOGS = Path(os.getenv("AIRFLOW_LOG_DIR", "/opt/airflow/logs"))
SKIP = {"scheduler", "dag_processor_manager"}
POLL_SECONDS = float(os.getenv("FORWARD_POLL_SECONDS", "1"))


def task_logs() -> list[Path]:
    try:
        return [path for path in LOGS.rglob("*.log") if not SKIP.intersection(path.parts)]
    except OSError:
        return []


def label(path: Path) -> str:
    parts = dict(piece.split("=", 1) for piece in path.parts if "=" in piece)
    dag, task = parts.get("dag_id"), parts.get("task_id")
    return f"{dag}.{task}" if dag and task else path.stem


def size_of(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except OSError:
        return None


def main() -> None:
    # Start at the end of existing files: history is already in the UI.
    offsets = {path: size_of(path) or 0 for path in task_logs()}
    while True:
        for path in task_logs():
            size = size_of(path)
            if size is None:
                continue
            start = offsets.get(path)
            if start is None:
                print(f"---- {path.relative_to(LOGS)} ----", flush=True)
                start = 0
            if size < start:  # truncated
                start = 0
            if size == start:
                offsets[path] = size
                continue
            try:
                with path.open("rb") as handle:
                    handle.seek(start)
                    data = handle.read()
            except OSError:
                continue
            offsets[path] = start + len(data)
            tag = label(path)
            for line in data.decode("utf-8", "replace").splitlines():
                print(f"{tag} | {line}", flush=True)
        offsets = {path: offset for path, offset in offsets.items() if path.exists()}
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
