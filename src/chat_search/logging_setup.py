"""Logging configuration and redaction.

Every record passes through a redacting formatter before a handler writes it:
configured secret values, URL userinfo, bearer tokens and credential-like
assignments are replaced. Tracebacks are redacted too, so enabling them is a
debugging switch, not a permission to leak.

Log calls use exception *types* and static messages. Raw exception text (which
may contain a request body, a URL or provider data) is only emitted when
``LOG_TRACEBACKS`` is on, and even then it is redacted.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Iterable
from urllib.parse import urlsplit, urlunsplit

from . import config

_NOISY_LOGGERS = (
    "elastic_transport",
    "elasticsearch",
    "httpx",
    "httpcore",
    "qdrant_client",
    "urllib3",
)

_SECRET_ENV_MARKERS = ("SECRET", "TOKEN", "PASSWORD", "API_KEY", "PRIVATE_KEY")
_SECRET_CONFIG_ATTRS = (
    "JWT_SECRET",
    "LLM_API_KEY",
    "QDRANT_API_KEY",
    "ES_API_KEY",
    "ES_SYNC_API_KEY",
    "ES_PASSWORD",
    "PROVIDERS_API_TOKEN",
)

# scheme://user:password@host -> scheme://[redacted]@host
_USERINFO_RE = re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://)(?:[^/\s:@]*:[^/\s@]*@)")
_AUTH_HEADER_RE = re.compile(
    r"(?i)\b(authorization)\b(\s*[:=]\s*)"
    r"(?:(?:bearer|basic|token|digest)\s+)?"
    r"(\"[^\"]*\"|'[^']*'|[^\s,;]+)"
)
_BEARER_RE = re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._~+/=-]{8,}")
_CREDENTIAL_RE = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|refresh[_-]?token|client[_-]?secret|"
    r"secret|password|passwd|token|jwt)\b([\"']?\s*[:=]\s*)(\"[^\"]*\"|'[^']*'|[^\s,;]+)"
)
_TOKEN_PREFIX_RE = re.compile(
    r"\b(?:sk|pk|rk)-[A-Za-z0-9_-]{16,}\b|\bgh[pousr]_[A-Za-z0-9]{20,}\b|\bAKIA[0-9A-Z]{16}\b"
)
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_FIELD_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,32}$")

_SECRETS: tuple[str, ...] = ()


def secret_values() -> tuple[str, ...]:
    """Collect secret-looking environment values once, longest first."""
    values = {getattr(config, name, "") for name in _SECRET_CONFIG_ATTRS}
    values.update(os.getenv(name, "") for name in _SECRET_CONFIG_ATTRS)
    for name, value in os.environ.items():
        if any(marker in name.upper() for marker in _SECRET_ENV_MARKERS):
            values.add(value)
    clean = {value for value in values if isinstance(value, str) and len(value) >= 8}
    return tuple(sorted(clean, key=len, reverse=True))


def _redact_credential(match: re.Match[str]) -> str:
    key, separator = match.group(1), match.group(2)
    return f"{key}{separator}[redacted]"


def redact(text: str, secrets: Iterable[str] | None = None) -> str:
    """Remove known secrets and credential-shaped text from one message."""
    if not text:
        return text
    for secret in _SECRETS if secrets is None else secrets:
        if secret and secret in text:
            text = text.replace(secret, "[REDACTED]")
    text = _USERINFO_RE.sub(r"\1[redacted]@", text)
    text = _AUTH_HEADER_RE.sub(r"\1\2[redacted]", text)
    text = _BEARER_RE.sub(r"\1 [redacted]", text)
    text = _CREDENTIAL_RE.sub(_redact_credential, text)
    return _TOKEN_PREFIX_RE.sub("[redacted]", text)


def _loggable(value: object, limit: int = 200) -> str:
    try:
        text = str(value)
    except Exception:
        return "<unprintable>"
    return _CONTROL_RE.sub("?", text)[:limit]


def _safe_field(value: object, limit: int = 160) -> str:
    # Redact the full value before trimming: a truncated secret can survive partly.
    return _loggable(redact(_loggable(value, 4096)), limit)


def sanitize_url(url: str) -> str:
    """A URL that is safe to log: no userinfo, query or fragment."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return "<invalid-url>"
    if not parts.scheme or not parts.netloc:
        return _loggable(redact(url), 300)
    netloc = parts.netloc.rsplit("@", 1)[-1]
    return _loggable(urlunsplit((parts.scheme, netloc, parts.path, "", "")), 300)


class RedactingFormatter(logging.Formatter):
    """Formatter that redacts the message and the traceback before output."""

    def __init__(self, *args: object, secrets: Iterable[str] | None = None, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self._secrets = secrets

    def format(self, record: logging.LogRecord) -> str:
        return redact(super().format(record), self._secrets)


def log_failure(
    logger: logging.Logger,
    message: str,
    exc: BaseException | None = None,
    *,
    level: int = logging.ERROR,
    **fields: object,
) -> None:
    """Log a static failure message plus non-sensitive fields.

    The exception's text is never part of the message. With LOG_TRACEBACKS the
    traceback is attached for debugging and still passes through redaction.
    """
    parts = [message]
    if exc is not None:
        parts.append(f"error_type={type(exc).__name__}")
    for key, value in fields.items():
        if _FIELD_NAME_RE.match(key):
            parts.append(f"{key}={_safe_field(value)}")
    exc_info = exc if config.LOG_TRACEBACKS and exc is not None else None
    logger.log(level, "%s", " ".join(parts), exc_info=exc_info)


def _install_redacting_formatter() -> None:
    formatter = RedactingFormatter()
    for handler in logging.root.handlers:
        handler.setFormatter(formatter)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "py.warnings"):
        for handler in logging.getLogger(name).handlers:
            handler.setFormatter(formatter)


def configure_logging() -> None:
    """Configure logging once for every entry point (API, CLI and jobs)."""
    global _SECRETS
    _SECRETS = secret_values()
    logging.basicConfig(
        level=config.LOG_LEVEL.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logging.captureWarnings(True)
    _install_redacting_formatter()
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
