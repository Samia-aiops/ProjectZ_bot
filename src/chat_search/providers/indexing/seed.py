"""Development helper: load sample providers into Elasticsearch."""

from __future__ import annotations

import logging
from datetime import date, timedelta

from ... import config
from ..client import make_es_client
from .schema import ensure_providers_index

log = logging.getLogger(__name__)


async def seed_providers(tenant: str) -> None:
    """Load sample providers for ``tenant``, plus one in another tenant to show isolation."""
    if not config.ES_URL:
        raise RuntimeError("Set ES_URL first")

    def soon(days: int) -> str:
        return (date.today() + timedelta(days=days)).isoformat()

    sample = [
        {
            "provider_id": "prov-1",
            "name": "Dr. Sara Khan",
            "specialty": "Cardiology",
            "subspecialties": ["Interventional Cardiology"],
            "conditions_treated": ["heart disease", "hypertension", "chest pain"],
            "languages": ["English", "Urdu"],
            "city": "Islamabad",
            "rating": 4.8,
            "review_count": 212,
            "accepting_new_patients": True,
            "next_available": soon(2),
        },
        {
            "provider_id": "prov-2",
            "name": "Dr. Omar Farooq",
            "specialty": "Cardiology",
            "subspecialties": ["Heart Failure"],
            "conditions_treated": ["heart failure", "arrhythmia"],
            "languages": ["English", "Punjabi"],
            "city": "Lahore",
            "rating": 4.2,
            "review_count": 87,
            "accepting_new_patients": True,
            "next_available": soon(5),
        },
        {
            "provider_id": "prov-3",
            "name": "Dr. Ahmed Raza",
            "specialty": "Dermatology",
            "subspecialties": ["Cosmetic Dermatology"],
            "conditions_treated": ["acne", "eczema", "psoriasis"],
            "languages": ["English", "Urdu", "Punjabi"],
            "city": "Lahore",
            "rating": 4.5,
            "review_count": 140,
            "accepting_new_patients": True,
            "next_available": soon(3),
        },
        {
            "provider_id": "prov-4",
            "name": "Dr. Imran Malik",
            "specialty": "Orthopedics",
            "subspecialties": ["Sports Medicine"],
            "conditions_treated": ["knee pain", "ACL injury", "fractures"],
            "languages": ["English", "Urdu"],
            "city": "Islamabad",
            "rating": 4.6,
            "review_count": 96,
            "accepting_new_patients": False,
            "next_available": soon(9),
        },
        {
            "provider_id": "prov-5",
            "name": "Dr. Ayesha Siddiqui",
            "specialty": "General Medicine",
            "subspecialties": [],
            "conditions_treated": ["fever", "diabetes", "hypertension"],
            "languages": ["English", "Urdu"],
            "city": "Islamabad",
            "rating": 4.7,
            "review_count": 301,
            "accepting_new_patients": True,
            "next_available": soon(1),
        },
        {
            "provider_id": "prov-6",
            "name": "Dr. Maria Lopez",
            "specialty": "Pediatrics",
            "subspecialties": ["Neonatology"],
            "conditions_treated": ["childhood vaccination", "asthma"],
            "languages": ["English", "Spanish"],
            "city": "Karachi",
            "rating": 4.9,
            "review_count": 178,
            "accepting_new_patients": True,
            "next_available": soon(4),
        },
    ]

    client = make_es_client(write=True)
    try:
        await ensure_providers_index(client)
        for doc in sample:
            await client.index(
                index=config.ES_PROVIDERS_INDEX,
                id=f"{tenant}-{doc['provider_id']}",
                document={"tenant_id": tenant, "active": True, **doc},
            )
        await client.index(
            index=config.ES_PROVIDERS_INDEX,
            id="other-tenant-prov-1",
            document={
                "tenant_id": "other-tenant",
                "active": True,
                "provider_id": "prov-1",
                "name": "Dr. Other Tenant",
                "specialty": "Cardiology",
                "languages": ["English"],
                "city": "Islamabad",
                "rating": 5.0,
                "accepting_new_patients": True,
            },
        )
        await client.indices.refresh(index=config.ES_PROVIDERS_INDEX)
        log.info(
            "Indexed %d providers for tenant '%s' (+1 in 'other-tenant') into '%s'",
            len(sample),
            tenant,
            config.ES_PROVIDERS_INDEX,
        )
    finally:
        await client.close()
