"""JWT authentication and the caller's RBAC identity.

Everything the service knows about a caller comes from the verified token:
role, tenant, subject and permissions. The token is issued by Laravel and its
algorithm is pinned in configuration, never taken from the token header.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from . import config
from .errors import AppError
from .observability import metric

bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class Principal:
    subject: str
    role: str
    tenant_id: str
    patient_id: str | None
    provider_id: str | None
    permissions: frozenset[str]


def _claim_str(claims: dict[str, Any], *names: str) -> str:
    for name in names:
        value = claims.get(name)
        if isinstance(value, (str, int)) and not isinstance(value, bool) and str(value).strip():
            return str(value).strip()
    return ""


def principal_from_claims(claims: dict[str, Any]) -> Principal:
    subject = _claim_str(claims, "sub")
    role = _claim_str(claims, "role").lower()
    tenant = _claim_str(claims, "tenant_id", "org_id")
    patient_id = _claim_str(claims, "patient_id") or None
    provider_id = _claim_str(claims, "provider_id") or None

    raw = claims.get("permissions", claims.get("scopes", claims.get("scope", [])))
    if isinstance(raw, str):
        raw = raw.split()
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise AppError(403, "forbidden", "Access denied")
    permissions = frozenset(item.strip() for item in raw if item.strip())

    if role not in config.ROLE_AUDIENCES or not subject or not config.TENANT_RE.match(tenant):
        raise AppError(403, "forbidden", "Access denied")
    if role == "patient" and not patient_id:
        raise AppError(403, "forbidden", "Access denied")
    if role == "provider" and not provider_id:
        raise AppError(403, "forbidden", "Access denied")
    if config.REQUIRED_PERMISSION not in permissions:
        raise AppError(403, "forbidden", "Access denied")

    return Principal(subject, role, tenant, patient_id, provider_id, permissions)


def dev_principal(role: str, tenant_id: str = "default") -> Principal:
    """Bypass identity for ``DEV_BYPASS_ROLE`` and the stdio MCP transport.

    The HTTP call sites check ``dev_bypass_enabled`` first; the stdio transport
    is local-only and identifies itself through ``MCP_ROLE``.
    """
    if role not in config.ROLE_AUDIENCES:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Bad DEV_BYPASS_ROLE")
    permissions = {config.REQUIRED_PERMISSION}
    if role == "patient" and config.PROVIDER_SEARCH_PERMISSION:
        permissions.add(config.PROVIDER_SEARCH_PERMISSION)
    if role == "provider" and config.PROVIDER_SQL_PERMISSION:
        permissions.add(config.PROVIDER_SQL_PERMISSION)
    return Principal(
        subject=f"dev-{role}",
        role=role,
        tenant_id=tenant_id,
        patient_id=f"dev-{role}-1" if role == "patient" else None,
        provider_id=(config.DEV_PROVIDER_ID or f"dev-{role}-1") if role == "provider" else None,
        permissions=frozenset(permissions),
    )


def principal_from_token(token: str) -> Principal:
    """Verify a Laravel-issued JWT and build the caller's identity."""
    try:
        claims = jwt.decode(
            token,
            config.JWT_KEY,
            algorithms=[config.JWT_ALGORITHM],  # pinned, never taken from the token header
            audience=config.JWT_AUDIENCE,
            issuer=config.JWT_ISSUER,
            leeway=config.JWT_LEEWAY,
            options={"require": ["exp", "sub"], "verify_aud": config.JWT_AUDIENCE is not None},
        )
    except jwt.PyJWTError as exc:
        metric(auth_error=type(exc).__name__)
        raise AppError(401, "unauthorized", "Invalid or expired session", {"WWW-Authenticate": "Bearer"}) from None

    principal = principal_from_claims(claims)
    metric(role=principal.role, tenant=principal.tenant_id)
    return principal


def dev_bypass_enabled() -> bool:
    """The HTTP bypass needs both a role and an explicitly declared dev environment."""
    return bool(config.DEV_BYPASS_ROLE) and config.APP_ENV in config.DEV_ENVIRONMENTS


async def current_principal(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
) -> Principal:
    # A presented token is always verified: the bypass must never shadow a real
    # credential, and one config value alone must not turn authentication off.
    if credentials is not None:
        return principal_from_token(credentials.credentials)
    if dev_bypass_enabled():
        return dev_principal(config.DEV_BYPASS_ROLE)
    if config.DEV_BYPASS_ROLE:
        # Startup validation refuses this combination; fail closed if reached
        # outside the application lifespan.
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "DEV_BYPASS_ROLE requires a dev APP_ENV")
    raise AppError(401, "unauthorized", "Authentication required", {"WWW-Authenticate": "Bearer"})


def auth_scope(principal: Principal) -> str:
    """Cache identity = exactly what the Qdrant RBAC filter lets this caller read."""
    parts = [
        ",".join(sorted({principal.tenant_id, config.GLOBAL_TENANT})),
        principal.role,
        ",".join(config.ROLE_AUDIENCES[principal.role]),
    ]
    if config.CACHE_PER_USER:
        parts.append(principal.subject)
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:24]


def mint_dev_token(role: str, tenant: str = "default") -> str:
    """Dev/testing only. In production Laravel issues the token."""
    if role not in config.ROLE_AUDIENCES:
        raise ValueError(f"role must be one of {sorted(config.ROLE_AUDIENCES)}")

    now = int(time.time())
    claims: dict[str, Any] = {
        "sub": f"dev-{role}",
        "role": role,
        "tenant_id": tenant,
        "permissions": [config.REQUIRED_PERMISSION]
        + ([config.PROVIDER_SEARCH_PERMISSION] if role == "patient" and config.PROVIDER_SEARCH_PERMISSION else [])
        + ([config.PROVIDER_SQL_PERMISSION] if role == "provider" and config.PROVIDER_SQL_PERMISSION else []),
        "iat": now,
        "exp": now + 3600,
    }
    dev_id = config.DEV_PROVIDER_ID if role == "provider" else ""
    claims["patient_id" if role == "patient" else "provider_id"] = dev_id or f"dev-{role}-1"
    if config.JWT_ISSUER:
        claims["iss"] = config.JWT_ISSUER
    if config.JWT_AUDIENCE:
        claims["aud"] = config.JWT_AUDIENCE
    return jwt.encode(claims, config.JWT_KEY, algorithm=config.JWT_ALGORITHM)
