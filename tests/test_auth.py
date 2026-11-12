from __future__ import annotations

import time

import jwt
import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from chat_search import config
from chat_search.auth import (
    Principal,
    auth_scope,
    current_principal,
    mint_dev_token,
    principal_from_claims,
)
from chat_search.errors import AppError

TEST_SECRET = "test-secret-0123456789abcdef0123456789"


def claims(**overrides):
    value = {
        "sub": "user-1",
        "role": "patient",
        "tenant_id": "acme",
        "patient_id": "patient-1",
        "permissions": [config.REQUIRED_PERMISSION],
    }
    value.update(overrides)
    return value


def principal(**overrides) -> Principal:
    return principal_from_claims(claims(**overrides))


def test_patient_principal_is_built_from_claims():
    result = principal()

    assert result.subject == "user-1"
    assert result.role == "patient"
    assert result.tenant_id == "acme"
    assert result.patient_id == "patient-1"
    assert result.provider_id is None
    assert result.permissions == frozenset({config.REQUIRED_PERMISSION})


def test_org_id_is_an_alias_for_tenant_id():
    value = claims()
    value.pop("tenant_id")
    value["org_id"] = "acme"

    assert principal_from_claims(value).tenant_id == "acme"


def test_space_separated_scope_is_split():
    value = claims()
    value.pop("permissions")
    value["scope"] = f"{config.REQUIRED_PERMISSION} profile"

    result = principal_from_claims(value)

    assert result.permissions == frozenset({config.REQUIRED_PERMISSION, "profile"})


@pytest.mark.parametrize(
    "overrides",
    [
        {"role": "admin"},
        {"role": ""},
        {"sub": ""},
        {"tenant_id": "bad tenant"},
        {"tenant_id": "x" * 65},
        {"role": "patient", "patient_id": None},
        {"role": "provider", "patient_id": None},
        {"permissions": []},
        {"permissions": [1]},
        {"permissions": 42},
    ],
)
def test_invalid_principals_are_rejected(overrides):
    with pytest.raises(AppError) as excinfo:
        principal_from_claims(claims(**overrides))

    assert excinfo.value.status_code == 403
    assert excinfo.value.code == "forbidden"


def test_provider_principal_keeps_provider_id():
    result = principal(role="provider", patient_id=None, provider_id="provider-1")

    assert result.provider_id == "provider-1"
    assert result.patient_id is None


def test_auth_scope_tracks_role_tenant_and_optional_user(monkeypatch):
    monkeypatch.setattr(config, "CACHE_PER_USER", False)
    first = principal()
    same_access = principal(sub="user-2", patient_id="patient-2")
    other_role = principal(role="provider", patient_id=None, provider_id="provider-1")
    other_tenant = principal(tenant_id="other")

    assert auth_scope(first) == auth_scope(same_access)
    assert auth_scope(first) != auth_scope(other_role)
    assert auth_scope(first) != auth_scope(other_tenant)

    monkeypatch.setattr(config, "CACHE_PER_USER", True)

    assert auth_scope(first) != auth_scope(same_access)


def test_mint_dev_token_round_trips(monkeypatch):
    monkeypatch.setattr(config, "JWT_KEY", TEST_SECRET)
    monkeypatch.setattr(config, "JWT_ALGORITHM", "HS256")
    monkeypatch.setattr(config, "JWT_ISSUER", None)
    monkeypatch.setattr(config, "JWT_AUDIENCE", None)

    token = mint_dev_token("patient", "acme")
    decoded = jwt.decode(token, TEST_SECRET, algorithms=["HS256"], options={"verify_aud": False})

    assert decoded["sub"] == "dev-patient"
    assert decoded["role"] == "patient"
    assert decoded["tenant_id"] == "acme"
    assert decoded["patient_id"] == "dev-patient-1"
    assert config.REQUIRED_PERMISSION in decoded["permissions"]


def test_mint_dev_token_rejects_unknown_role(monkeypatch):
    monkeypatch.setattr(config, "JWT_KEY", TEST_SECRET)
    monkeypatch.setattr(config, "JWT_ALGORITHM", "HS256")

    with pytest.raises(ValueError):
        mint_dev_token("admin")


async def test_missing_credentials_are_unauthorized(monkeypatch):
    monkeypatch.setattr(config, "DEV_BYPASS_ROLE", "")

    with pytest.raises(AppError) as excinfo:
        await current_principal(None)

    assert excinfo.value.status_code == 401
    assert excinfo.value.headers == {"WWW-Authenticate": "Bearer"}


async def test_dev_bypass_returns_a_principal(monkeypatch):
    monkeypatch.setattr(config, "DEV_BYPASS_ROLE", "patient")
    monkeypatch.setattr(config, "APP_ENV", "test")

    result = await current_principal(None)

    assert result.subject == "dev-patient"
    assert result.tenant_id == "default"
    assert result.patient_id == "dev-patient-1"
    assert config.REQUIRED_PERMISSION in result.permissions


async def test_dev_provider_id_overrides_the_placeholder(monkeypatch):
    monkeypatch.setattr(config, "DEV_BYPASS_ROLE", "provider")
    monkeypatch.setattr(config, "APP_ENV", "test")
    monkeypatch.setattr(config, "DEV_PROVIDER_ID", "263")

    result = await current_principal(None)

    assert result.provider_id == "263"


async def test_the_bypass_is_refused_outside_a_dev_environment(monkeypatch):
    monkeypatch.setattr(config, "DEV_BYPASS_ROLE", "patient")
    monkeypatch.setattr(config, "APP_ENV", "production")

    with pytest.raises(HTTPException) as excinfo:
        await current_principal(None)

    assert excinfo.value.status_code == 500


async def test_a_presented_token_is_verified_even_when_the_bypass_is_active(monkeypatch):
    monkeypatch.setattr(config, "DEV_BYPASS_ROLE", "patient")
    monkeypatch.setattr(config, "APP_ENV", "test")
    real = principal()
    monkeypatch.setattr("chat_search.auth.principal_from_token", lambda _: real)
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="token")

    assert await current_principal(credentials) is real


def test_startup_refuses_the_bypass_outside_a_dev_environment(monkeypatch):
    monkeypatch.setattr(config, "JWT_KEY", TEST_SECRET)
    monkeypatch.setattr(config, "LLM_API_KEY", "test-key")
    monkeypatch.setattr(config, "DEV_BYPASS_ROLE", "patient")
    monkeypatch.setattr(config, "APP_ENV", "production")

    with pytest.raises(RuntimeError, match="APP_ENV"):
        config.validate_runtime()


def test_startup_refuses_an_unknown_bypass_role(monkeypatch):
    monkeypatch.setattr(config, "JWT_KEY", TEST_SECRET)
    monkeypatch.setattr(config, "LLM_API_KEY", "test-key")
    monkeypatch.setattr(config, "DEV_BYPASS_ROLE", "admin")
    monkeypatch.setattr(config, "APP_ENV", "test")

    with pytest.raises(RuntimeError, match="DEV_BYPASS_ROLE"):
        config.validate_runtime()


async def test_invalid_dev_bypass_role_fails_loudly(monkeypatch):
    monkeypatch.setattr(config, "DEV_BYPASS_ROLE", "admin")
    monkeypatch.setattr(config, "APP_ENV", "test")

    with pytest.raises(HTTPException) as excinfo:
        await current_principal(None)

    assert excinfo.value.status_code == 500


async def test_current_principal_decodes_a_valid_token(monkeypatch):
    monkeypatch.setattr(config, "DEV_BYPASS_ROLE", "")
    monkeypatch.setattr(config, "JWT_KEY", TEST_SECRET)
    monkeypatch.setattr(config, "JWT_ALGORITHM", "HS256")
    monkeypatch.setattr(config, "JWT_ISSUER", None)
    monkeypatch.setattr(config, "JWT_AUDIENCE", None)

    now = int(time.time())
    token = jwt.encode(
        {
            **claims(),
            "iat": now,
            "exp": now + 60,
        },
        TEST_SECRET,
        algorithm="HS256",
    )
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

    result = await current_principal(credentials)

    assert result.subject == "user-1"
    assert result.tenant_id == "acme"


async def test_invalid_token_is_unauthorized(monkeypatch):
    monkeypatch.setattr(config, "DEV_BYPASS_ROLE", "")
    monkeypatch.setattr(config, "JWT_KEY", TEST_SECRET)
    monkeypatch.setattr(config, "JWT_ALGORITHM", "HS256")
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="not-a-token")

    with pytest.raises(AppError) as excinfo:
        await current_principal(credentials)

    assert excinfo.value.status_code == 401
    assert excinfo.value.code == "unauthorized"
