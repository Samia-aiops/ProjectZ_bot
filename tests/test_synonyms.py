from __future__ import annotations

import json

import pytest

from chat_search import config
from chat_search.synonyms import (
    expand_synonyms,
    get_catalog,
    protected_words,
    reset_synonyms,
    synonym_terms,
)

CATALOG = {
    "specialties": {
        "pediatrics": ["pediatrician", "child specialist", "kids doctor", "children's doctor"],
        "cardiology": ["cardiologist", "heart doctor"],
        "gynecology": ["ob-gyn", "obgyn"],
    },
    "protected_terms": ["hypertension", "atrial fibrillation"],
}


@pytest.fixture(autouse=True)
def inline_catalog(monkeypatch):
    monkeypatch.setattr(config, "SYNONYMS_ENABLED", True)
    monkeypatch.setattr(config, "SYNONYMS_JSON", json.dumps(CATALOG))
    monkeypatch.setattr(config, "SYNONYMS_RELOAD_SECONDS", 60.0)
    reset_synonyms()


def test_variant_is_expanded_to_the_canonical_term():
    text = "I need a child specialist in Boston"

    expanded = expand_synonyms(text)

    assert expanded == "pediatrics " + text
    assert synonym_terms(text) == ["pediatrics"]


def test_plural_variants_are_expanded_too():
    assert synonym_terms("best child specialists") == ["pediatrics"]
    assert synonym_terms("heart doctors nearby") == ["cardiology"]
    assert synonym_terms("cardiologists") == ["cardiology"]
    assert expand_synonyms("pediatricians") == "pediatrics pediatricians"


def test_canonical_term_is_not_duplicated():
    assert synonym_terms("pediatrics") == []
    assert expand_synonyms("pediatrics") == "pediatrics"


def test_each_group_is_added_once_in_text_order():
    assert synonym_terms("child specialist and heart doctor") == ["pediatrics", "cardiology"]


def test_matching_respects_word_boundaries():
    assert synonym_terms("This is childlike, not a specialist") == []


def test_hyphen_and_apostrophe_variants_match():
    assert synonym_terms("I need an OB-GYN") == ["gynecology"]
    assert synonym_terms("childrens doctor") == ["pediatrics"]


def test_matching_is_capped(monkeypatch):
    monkeypatch.setattr(config, "SYNONYM_MAX_EXPANSIONS", 1)

    assert synonym_terms("child specialist and heart doctor") == ["pediatrics"]


def test_protected_and_domain_words_come_from_the_catalog():
    protected = protected_words()
    domain = get_catalog().domain

    assert {"hypertension", "atrial", "fibrillation"} <= protected
    assert {"pediatrics", "cardiologist", "obgyn"} <= domain


def test_disabled_catalog_leaves_text_unchanged(monkeypatch):
    monkeypatch.setattr(config, "SYNONYMS_ENABLED", False)

    assert expand_synonyms("child specialist") == "child specialist"


def test_invalid_inline_catalog_falls_back_to_defaults(monkeypatch):
    monkeypatch.setattr(config, "SYNONYMS_JSON", "{not-json")
    reset_synonyms()

    assert expand_synonyms("child specialist").startswith("pediatrics ")


def test_file_catalog_reloads_on_force(monkeypatch, tmp_path):
    path = tmp_path / "synonyms.json"
    path.write_text(json.dumps({"specialties": {"pediatrics": ["child specialist"]}}), encoding="utf-8")
    monkeypatch.setattr(config, "SYNONYMS_JSON", "")
    monkeypatch.setattr(config, "SYNONYMS_FILE", str(path))
    reset_synonyms()
    assert synonym_terms("child specialist") == ["pediatrics"]

    path.write_text(json.dumps({"specialties": {"cardiology": ["heart doctor"]}}), encoding="utf-8")
    assert synonym_terms("heart doctor", get_catalog(force=True)) == ["cardiology"]


def test_the_default_catalogue_adds_the_file_wording_for_attachments(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "SYNONYMS_JSON", "")
    monkeypatch.setattr(config, "SYNONYMS_FILE", str(tmp_path / "missing.json"))
    reset_synonyms()

    assert synonym_terms("maximum attachment size") == ["file attachment"]
    assert synonym_terms("maximum size of an attachment") == ["file attachment"]
    assert synonym_terms("how do I attach a file") == ["file attachment"]
    assert synonym_terms("maximum attachment size for appointments") == ["file attachment", "appointment booking"]


def test_the_default_catalogue_covers_booking_and_proxy_booking_wording(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "SYNONYMS_JSON", "")
    monkeypatch.setattr(config, "SYNONYMS_FILE", str(tmp_path / "missing.json"))
    reset_synonyms()

    assert synonym_terms("how do I book a doctor") == ["appointment booking"]
    assert synonym_terms("reschedule my visit to next friday") == ["appointment booking reschedule"]
    assert synonym_terms("move my visit to next friday") == ["appointment booking reschedule"]
    assert synonym_terms("a video appointment for my mother") == ["appointment booking"]
    assert synonym_terms("make sure it is for her rather than me") == ["book for someone else"]
    assert synonym_terms("book for a family member") == ["book for someone else"]
