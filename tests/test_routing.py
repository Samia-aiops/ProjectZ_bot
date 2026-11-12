from __future__ import annotations

import pytest

from chat_search import config
from chat_search.routing import route
from chat_search.synonyms import expand_synonyms, reset_synonyms


@pytest.mark.parametrize(
    "text",
    [
        "Can you recommend a cardiologist in Boston?",
        "I'm looking for a dermatologist who speaks Urdu",
        "Find a GP near me",
        "Do you have any neurologists available?",
        "I need a pediatrician",
    ],
)
def test_patients_are_routed_to_provider_search(text):
    assert route(text, "auto") == "providers"


@pytest.mark.parametrize(
    "text",
    [
        "How do I find a doctor?",
        "Where can I update my profile?",
        "Tell me about the refund policy",
        "I need a new password",
        "What are the opening hours?",
    ],
)
def test_faq_hints_and_non_provider_questions_stay_in_rag(text):
    assert route(text, "auto") == "faq"


def test_directory_intent_is_routed_for_every_role():
    # The provider pipeline refuses non-patients, so a provider's directory
    # question reaches the feature-unavailable reply instead of a document lookup.
    assert route(expand_synonyms("Recommend a cardiologist"), "auto") == "providers"
    assert route(expand_synonyms("who speaks Urdu"), "auto") == "providers"


def test_primary_care_synonyms_route_to_the_directory():
    assert route(expand_synonyms("show me best pcp in newyork"), "auto") == "providers"
    assert route(expand_synonyms("best primary care doctor in new york"), "auto") == "providers"


@pytest.mark.parametrize(
    "text",
    ["Who speaks Urdu?", "Find a doctor fluent in Spanish", "I need a provider who speaks Arabic"],
)
def test_a_language_request_routes_to_the_directory(text):
    assert route(text, "auto") == "providers"


@pytest.mark.parametrize(
    "text",
    [
        "spanish doctors",
        "English-speaking doctors",
        "haitian creole doctors",
        "find cardiologists whose primary language is English",
    ],
)
def test_a_language_attached_to_a_provider_noun_routes_to_the_directory(text):
    assert route(text, "auto") == "providers"


def test_a_language_outside_the_map_still_routes_to_the_directory():
    assert route("Who speaks Azerbaijani?", "auto") == "providers"


@pytest.mark.parametrize(
    "text",
    [
        "Which languages are supported in the app?",
        "Does the platform support Spanish?",
        "What language is the patient manual written in?",
    ],
)
def test_a_language_question_that_is_not_a_search_stays_in_rag(text):
    assert route(text, "auto") == "faq"


@pytest.mark.parametrize(
    "text",
    [
        "doctors with best ratings",
        "doctors with the best ratings",
        "highest rated physicians",
        "top-rated doctors",
    ],
)
def test_a_rating_request_routes_to_the_directory(text):
    assert route(text, "auto") == "providers"


@pytest.mark.parametrize("text", ["How do doctors get rated?", "Are doctors rated by patients?"])
def test_a_question_about_how_ratings_work_stays_in_rag(text):
    assert route(text, "auto") == "faq"


@pytest.mark.parametrize("text", ["I need cardiology", "Find someone in pediatrics", "Who does gynecology?"])
def test_department_names_route_to_providers(text):
    assert route(text, "auto") == "providers"


def test_lay_terms_route_after_synonym_expansion(monkeypatch):
    monkeypatch.setattr(
        config,
        "SYNONYMS_JSON",
        '{"specialties": {"pediatrics": ["child specialist"], "cardiology": ["heart doctor"]}}',
    )
    reset_synonyms()

    assert route(expand_synonyms("I need a child specialist"), "auto") == "providers"
    assert route(expand_synonyms("Find a heart doctor"), "auto") == "providers"


@pytest.mark.parametrize(
    "text",
    [
        "I forgot my password",
        "How does billing work?",
        "Is insurance covered for a visit?",
        "How do I update my profile?",
    ],
)
def test_account_billing_and_insurance_questions_stay_in_rag(text):
    assert route(text, "auto") == "faq"


def test_provider_search_with_insurance_still_routes_to_providers():
    assert route("Find a doctor who accepts my insurance", "auto") == "providers"


@pytest.mark.parametrize("mode", ["faq", "providers"])
def test_explicit_mode_wins(mode):
    assert route("How do I reset my password?", mode) == mode
    assert route("Recommend a cardiologist", mode) == mode
