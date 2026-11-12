from __future__ import annotations

import json

import pytest

from chat_search import config
from chat_search.spelling import reset_spelling, spell_correct
from chat_search.synonyms import reset_synonyms

pytest.importorskip("symspellpy")

CATALOG = {
    "specialties": {
        "pediatrics": ["pediatrician", "child specialist"],
        "cardiology": ["cardiologist", "heart doctor"],
    },
    "protected_terms": ["hypertension", "arrhythmia", "npi"],
}


@pytest.fixture(autouse=True)
def spelling(monkeypatch):
    monkeypatch.setattr(config, "SPELLCHECK_ENABLED", True)
    monkeypatch.setattr(config, "SPELL_MAX_EDIT_DISTANCE", 2)
    monkeypatch.setattr(config, "SPELL_MIN_FREQUENCY", 20)
    monkeypatch.setattr(config, "SPELL_MAX_CORRECTIONS", 4)
    monkeypatch.setattr(config, "SYNONYMS_ENABLED", True)
    monkeypatch.setattr(config, "SYNONYMS_JSON", json.dumps(CATALOG))
    reset_synonyms()
    reset_spelling()


def test_common_typo_is_corrected():
    assert spell_correct("please recieve this") == "please receive this"


def test_capitalised_first_word_keeps_its_case():
    assert spell_correct("Recieve this") == "Receive this"


def test_domain_words_are_correction_targets():
    assert spell_correct("i need a cardiologst") == "i need a cardiologist"
    assert spell_correct("find a pediatricain") == "find a pediatrician"


def test_long_domain_typo_can_take_a_third_edit():
    assert spell_correct("best paditricin") == "best pediatrician"
    assert spell_correct("Paditricin") == "Pediatrician"


def test_third_edit_fallback_only_touches_unknown_words():
    assert spell_correct("the president") == "the president"
    assert spell_correct("everything is fine") == "everything is fine"


def test_protected_medical_terms_are_left_alone():
    assert spell_correct("hypertension and arrhythmia") == "hypertension and arrhythmia"


def test_the_provider_identifier_is_not_corrected_to_another_word():
    # Without protection "npi" was rewritten to "dpi", which let a request for a
    # named doctor's identifier slip past the privacy rule.
    assert spell_correct("give me dr. joe npi") == "give me dr. joe npi"


def test_names_in_the_middle_of_a_query_are_left_alone():
    assert spell_correct("find a doctor at Aetna") == "find a doctor at Aetna"


def test_email_addresses_and_urls_are_left_verbatim():
    text = "write to john.doe@example.com or https://example.com/contact"

    assert spell_correct(text) == text


def test_acronyms_numbers_and_short_tokens_are_left_alone():
    assert spell_correct("ENT 02101 GP Dr X") == "ENT 02101 GP Dr X"


def test_corrections_are_capped_per_query(monkeypatch):
    monkeypatch.setattr(config, "SPELL_MAX_CORRECTIONS", 1)

    assert spell_correct("recieve recieve") == "receive recieve"


def test_disabled_spelling_returns_the_input(monkeypatch):
    monkeypatch.setattr(config, "SPELLCHECK_ENABLED", False)

    assert spell_correct("recieve this") == "recieve this"
