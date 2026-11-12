"""Application-level errors.

Handlers translate ``AppError`` into the API's stable error envelope without
leaking internals or user input.
"""

from __future__ import annotations


class AppError(Exception):
    """An expected failure with a safe, user-facing message."""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(code)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.headers = headers
