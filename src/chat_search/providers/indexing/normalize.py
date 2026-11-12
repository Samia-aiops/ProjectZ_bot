"""Normalise provider records from the backend API into Elasticsearch documents.

Only the fields in ``PROVIDER_FIELDS`` are ever copied, so emails, phone numbers,
notes and anything else the API sends never reach the search index.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Any

from ... import config

log = logging.getLogger(__name__)

PROVIDER_FIELDS = (
    "provider_id",
    "tenant_id",
    "active",
    "name",
    "specialty",
    "subspecialties",
    "conditions_treated",
    "languages",
    "city",
    "state",
    "zipcode",
    "npi",
    "bio",
    "rating",
    "review_count",
    "accepting_new_patients",
    "next_available",
)

_DEFAULT_PATHS = {
    "provider_id": "user_id",
    "name": "name",
    "active": "status",
    "specialty": "primary_specialty",
    "subspecialties": "subspecialties",
    "conditions_treated": "conditions_treated",
    "languages": "languages",
    "city": "city",
    "state": "state",
    "zipcode": "zipcode",
    "bio": "bio",
    "rating": "rating",
    "review_count": "review_count",
    "accepting_new_patients": "accepting_new_patients",
    "next_available": "next_available",
}

_FALLBACK_PATHS = {
    "provider_id": ("user_id", "id", "provider_id"),
    "name": ("name",),
    "active": ("status", "active"),
    "specialty": ("primary_specialty", "speciality", "specialty"),
    "subspecialties": ("subspecialties", "sub_specialties"),
    "bio": ("bio", "about_provider"),
    "city": ("city",),
    "state": ("state",),
    "zipcode": ("zipcode", "zip_code", "postal_code"),
    "npi": ("npi",),
}

if unknown := set(config.PROVIDERS_FIELD_MAP) - set(PROVIDER_FIELDS):
    raise RuntimeError(f"PROVIDERS_FIELD_MAP has unknown keys: {sorted(unknown)}")

_rejection_counts: dict[str, int] = {}
_MAX_LOGGED_REJECTIONS_PER_REASON = 5


def _reject(reason: str) -> None:
    """Log each normalisation rejection reason at most five times."""
    count = _rejection_counts.get(reason, 0) + 1
    _rejection_counts[reason] = count
    if count <= _MAX_LOGGED_REJECTIONS_PER_REASON:
        log.warning("Provider normalization rejected record: reason=%s occurrence=%s", reason, count)


def get_path(obj: Any, path: str) -> Any:
    """'specialty.name' -> obj['specialty']['name'] (None if anything is missing)."""
    for part in path.split("."):
        if not isinstance(obj, dict):
            return None
        obj = obj.get(part)
    return obj


def _names(value: Any) -> list[str]:
    """List of strings, list of {"name": ...} objects, or a comma-separated string."""
    if value is None:
        return []
    if isinstance(value, str):
        value = value.split(",")
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        item = item.get("name") if isinstance(item, dict) else item
        if item is not None and str(item).strip():
            out.append(str(item).strip()[:100])
    return out


def _to_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y", "active")  # unknown text = False
    return bool(value)


def _to_date(value: Any) -> str | None:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).isoformat()
    except (TypeError, ValueError, OverflowError):
        return None


def build_provider_doc(item: Any) -> dict[str, Any] | None:
    """Normalise one provider record into an Elasticsearch document."""
    if not isinstance(item, dict):
        _reject("invalid_record")
        return None

    field_map = {
        **{field: field for field in PROVIDER_FIELDS},
        **_DEFAULT_PATHS,
        **config.PROVIDERS_FIELD_MAP,
    }

    def get(field: str) -> Any:
        for path in (field_map.get(field, field), *_FALLBACK_PATHS.get(field, ())):
            value = get_path(item, path)
            if value is not None:
                return value
        return None

    provider_id = get("provider_id")
    if provider_id is None:
        for key in ("user_id", "id", "provider_id"):
            value = item.get(key)
            if value is not None and str(value).strip():
                provider_id = value
                break
    if provider_id is None or not str(provider_id).strip():
        _reject("missing_provider_id")
        return None

    name = str(get("name") or "").strip()
    if not name:
        first_name = str(item.get("firstname") or "").strip()
        last_name = str(item.get("lastname") or "").strip()
        name = f"{first_name} {last_name}".strip()
    if not name:
        _reject("missing_name")
        return None

    raw_status = get("active")
    if isinstance(raw_status, str):
        status = raw_status.strip().lower()
        if status in {"active", "true", "1", "yes"}:
            active = True
        elif status in {"inactive", "false", "0", "no", "disabled"}:
            active = False
        else:
            active = True
    elif raw_status is None:
        active = True
    else:
        active = _to_bool(raw_status, True)

    doc: dict[str, Any] = {
        "provider_id": str(provider_id).strip(),
        "name": name[:200],
        "active": active,
    }

    specialty = get("specialty")
    specialties = item.get("specialties")
    if not specialty and isinstance(specialties, list) and specialties:
        specialty = specialties[0]
    if specialty:
        specialty = str(specialty).strip()
        if specialty:
            doc["specialty"] = specialty[:100]

    for field in ("subspecialties", "conditions_treated", "languages"):
        values = _names(get(field))
        if field == "subspecialties" and not values:
            values = _names(item.get("sub_specialty_details") or item.get("sub_specialties"))
        if values:
            doc[field] = values[:30]

    for field, limit in (
        ("city", 100),
        ("state", 100),
        ("zipcode", 20),
        ("bio", 2000),
        ("npi", 30),
    ):
        value = get(field)
        if value is not None and str(value).strip():
            doc[field] = str(value).strip()[:limit]

    for field in ("rating", "review_count"):
        value = get(field)
        if value is None or str(value).strip() == "":
            continue
        try:
            number = float(value)
            if not math.isfinite(number):
                _reject(f"invalid_{field}")
                continue
            if field == "rating":
                doc[field] = min(max(number, 0.0), 5.0)
            else:
                doc[field] = max(int(number), 0)
        except (TypeError, ValueError, OverflowError):
            _reject(f"invalid_{field}")

    accepting = get("accepting_new_patients")
    if accepting is not None:
        doc["accepting_new_patients"] = _to_bool(accepting, False)

    next_available = get("next_available")
    if next_available:
        when = _to_date(next_available)
        if when:
            doc["next_available"] = when
        else:
            _reject("invalid_next_available")

    tenant = get("tenant_id") or config.PROVIDERS_API_TENANT
    if tenant is not None and str(tenant).strip():
        doc["tenant_id"] = str(tenant).strip()[:64]

    return doc
