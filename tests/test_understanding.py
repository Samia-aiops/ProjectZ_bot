from __future__ import annotations

from types import SimpleNamespace

import pytest

from chat_search import config, understanding
from chat_search.routing import route
from chat_search.runtime import UserSlots
from chat_search.understanding import (
    expand_query,
    reset_understanding,
    understand,
)


@pytest.fixture(autouse=True)
def _no_clinical_model(monkeypatch):
    # Unit tests run without the scispaCy model; the catalogue path is the
    # documented degradation and must produce the same structured fields.
    monkeypatch.setattr(config, "SCISPACY_ENABLED", False)
    monkeypatch.setattr(config, "UMLS_ENABLED", False)
    reset_understanding()
    yield
    reset_understanding()


def test_a_lay_specialty_maps_to_the_canonical_concept():
    result = understand("I need a heart doctor")

    assert result.text == "i need a heart doctor"
    assert result.intent == "provider_search"
    assert result.specialties == ("cardiology",)
    assert result.canonical_terms == ("cardiology",)
    assert result.provider_nouns == ("doctor",)
    concept = result.medical_concepts[0]
    assert (concept.text, concept.canonical, concept.category, concept.cui) == (
        "heart doctor",
        "cardiology",
        "specialty",
        "C0007222",
    )


def test_the_structured_view_is_json_ready():
    data = understand("I need a heart doctor").as_dict()

    assert data == {
        "original": "I need a heart doctor",
        "normalized": "i need a heart doctor",
        "intent": "provider_search",
        "specialties": ["cardiology"],
        "faq_topics": [],
        "medical_concepts": [
            {
                "text": "heart doctor",
                "canonical": "cardiology",
                "category": "specialty",
                "source": "catalogue",
                "cui": "C0007222",
            }
        ],
        "provider_nouns": ["doctor"],
        "appointment": None,
        "languages": [],
        "canonical_terms": ["cardiology"],
    }


@pytest.mark.parametrize(
    ("text", "action", "object_"),
    [
        ("book a video consultation for my mother", "book", "consultation"),
        ("Move my visit to next Friday", "move", "visit"),
        ("cancel my appointment", "cancel", "appointment"),
    ],
)
def test_an_explicit_action_and_object_is_an_appointment(text, action, object_):
    result = understand(text)

    assert result.intent == "appointment"
    assert result.appointment is not None
    assert result.appointment.as_dict() == {"action": action, "object": object_}
    assert any(topic.startswith("appointment booking") for topic in result.faq_topics)


def test_a_specialty_alone_is_never_an_appointment():
    result = understand("I need a cardiologist")

    assert result.intent == "provider_search"
    assert result.appointment is None


def test_booking_a_doctor_counts_as_booking_the_appointment_topic():
    result = understand("how do I book a doctor")

    assert result.intent == "appointment"
    assert result.appointment.as_dict() == {"action": "book", "object": "appointment"}


def test_a_policy_question_stays_in_faq():
    result = understand("How do I update my profile?")

    assert result.intent == "faq"
    assert result.appointment is None
    assert result.specialties == ()


def test_a_condition_alias_maps_to_its_canonical_name():
    result = understand("history of high blood pressure")

    assert result.intent == "faq"
    concept = result.medical_concepts[0]
    assert (concept.canonical, concept.category, concept.cui) == (
        "hypertension",
        "condition",
        "C0020538",
    )


def test_normalization_strips_punctuation_and_case():
    assert understanding.normalize_query("  Heart-Doctor!!  ") == "heart doctor"


def test_the_regex_fallback_matches_the_spacy_matcher(monkeypatch):
    expected = understand("I need a heart doctor").specialties

    monkeypatch.setattr(understanding, "_ensure_matcher", lambda _catalogue: (None, None))
    fallback = understand("I need a heart doctor")

    assert fallback.specialties == expected == ("cardiology",)


def test_expansion_prefixes_the_canonical_and_keeps_the_wording():
    assert expand_query("I need a heart doctor") == "cardiology I need a heart doctor"


def test_routing_and_understanding_agree():
    cases = [
        ("I need a heart doctor", "providers", "provider_search"),
        ("book a video consultation", "faq", "appointment"),
        ("How do I update my profile?", "faq", "faq"),
        ("Find a doctor who accepts my insurance", "providers", "provider_search"),
    ]
    for text, node, intent in cases:
        assert route(text, "auto") == node
        assert understand(text, medical=False).intent == intent


@pytest.mark.parametrize(
    "text",
    [
        "Does DocHyve itself provide medical care?",
        "I am authorized to manage her care.",
        "How do I make sure it is for her rather than me?",
        "How do I add my parent as a member?",
        "Can I see their Health Check too?",
        "What is the limit for Settings lab uploads?",
    ],
)
def test_product_questions_stay_in_the_documents(text):
    assert route(text, "auto") == "faq"
    assert understand(text, medical=False).intent == "faq"


@pytest.mark.parametrize(
    ("text", "node"),
    [
        ("I use Plan A in New York.", "faq"),
        ("Actually Plan B in Arizona.", "faq"),
        ("Which plan should I choose in search?", "faq"),
        ("what insurance plans do you accept?", "faq"),
        ("find a doctor on my plan", "providers"),
        ("I need a doctor in New York", "providers"),
    ],
)
def test_plan_questions_are_not_turned_into_a_directory_search(text, node):
    assert route(text, "auto") == node


async def test_the_chat_done_event_carries_the_structured_understanding(monkeypatch):
    from chat_search.auth import Principal
    from chat_search.pipeline import agent, chat

    principal = Principal(
        subject="user-1",
        role="patient",
        tenant_id="acme",
        patient_id="patient-1",
        provider_id=None,
        permissions=frozenset({"docs:read", "providers:search"}),
    )

    async def history(*_):
        return []

    async def no_store(*_args):
        return None

    async def fake_pipeline(*_args):
        yield {"type": "token", "text": "ok"}
        yield {"type": "done", "sources": [], "cached": False}

    monkeypatch.setattr(chat, "load_history", history)
    monkeypatch.setattr(chat, "append_turns", no_store)
    monkeypatch.setattr(agent, "rag_pipeline", fake_pipeline)
    monkeypatch.setattr(chat, "get_runtime", lambda: SimpleNamespace(user_slots=UserSlots(1)))

    events = [event async for event in chat.rag_events(principal, "I need a heart doctor", "faq")]

    assert events[-1]["type"] == "done"
    assert events[-1]["data"]["understanding"]["specialties"] == ["cardiology"]
    assert events[-1]["data"]["understanding"]["intent"] == "provider_search"
