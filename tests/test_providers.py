from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from chat_search import config
from chat_search.guardrails import normalize
from chat_search.providers import search as provider_search
from chat_search.providers.indexing.normalize import PROVIDER_FIELDS, build_provider_doc, get_path
from chat_search.providers.indexing.schema import provider_synonym_rules
from chat_search.providers.search import (
    format_providers,
    language_provider_request,
    provider_languages,
    provider_query,
    provider_terms,
)
from chat_search.spelling import reset_spelling, spell_correct
from chat_search.synonyms import reset_synonyms


def record(**overrides):
    value = {
        "user_id": "provider-1",
        "name": "Dr Ada Lovelace",
        "status": "active",
        "primary_specialty": "Cardiology",
        "subspecialties": [{"name": "Electrophysiology"}],
        "conditions_treated": "Atrial fibrillation, Hypertension",
        "languages": ["English", "Urdu"],
        "city": "Boston",
        "state": "MA",
        "zipcode": "02101",
        "bio": "Board certified cardiologist.",
        "rating": 4.8,
        "review_count": "123",
        "accepting_new_patients": "yes",
        "next_available": "2026-11-01T09:00:00Z",
        "tenant_id": "acme",
        "email": "ada@example.com",
        "phone": "+15551234567",
    }
    value.update(overrides)
    return value


def test_get_path_walks_nested_objects():
    assert get_path({"a": {"b": 1}}, "a.b") == 1
    assert get_path({"a": None}, "a.b") is None
    assert get_path({"a": {"b": 1}}, "a.c") is None


def test_build_provider_doc_copies_only_public_fields():
    doc = build_provider_doc(record())

    assert doc is not None
    assert doc["provider_id"] == "provider-1"
    assert doc["name"] == "Dr Ada Lovelace"
    assert doc["active"] is True
    assert doc["specialty"] == "Cardiology"
    assert doc["subspecialties"] == ["Electrophysiology"]
    assert doc["conditions_treated"] == ["Atrial fibrillation", "Hypertension"]
    assert doc["languages"] == ["English", "Urdu"]
    assert doc["city"] == "Boston"
    assert doc["state"] == "MA"
    assert doc["zipcode"] == "02101"
    assert doc["rating"] == 4.8
    assert doc["review_count"] == 123
    assert doc["accepting_new_patients"] is True
    assert doc["next_available"] == "2026-11-01T09:00:00+00:00"
    assert doc["tenant_id"] == "acme"
    assert "email" not in doc
    assert "phone" not in doc
    assert set(doc) <= set(PROVIDER_FIELDS)


def test_provider_terms_leave_specialty_wording_for_the_search_analyzer(monkeypatch):
    monkeypatch.setattr(
        config,
        "SYNONYMS_JSON",
        '{"specialties": {"pediatrics": ["child specialist"]}}',
    )
    reset_synonyms()

    terms = provider_terms("child specialist in Boston")

    assert terms.split() == ["child", "specialist", "boston"]
    assert "child specialist => pediatrics" in provider_synonym_rules()


def test_the_search_analyzer_adds_the_directory_label(monkeypatch):
    monkeypatch.setattr(
        config,
        "SYNONYMS_JSON",
        '{"specialties": {"gynecology": ["gynecologist", "gynaecologist", "ob-gyn"],'
        ' "ophthalmology": ["ophthalmologist"]}}',
    )
    reset_synonyms()

    # The app no longer translates labels; the index analyzer adds them.
    assert provider_terms("best gynecologist").split() == ["gynecologist"]
    assert provider_terms("best gynaecologist").split() == ["gynaecologist"]
    assert provider_terms("best ob-gyn").split() == ["ob-gyn"]
    assert provider_terms("ophthalmology") == "ophthalmology"

    rules = provider_synonym_rules()
    assert "gynecologist => gynecology, ob gyn" in rules
    assert "ophthalmology => ophthalmology, eye doctor" in rules


def test_a_misspelled_gynecologist_query_reaches_the_ob_gyn_label(monkeypatch):
    pytest.importorskip("symspellpy")
    monkeypatch.setattr(config, "SPELLCHECK_ENABLED", True)
    monkeypatch.setattr(
        config,
        "SYNONYMS_JSON",
        '{"specialties": {"gynecology": ["gynecologist", "gynaecologist"]}}',
    )
    reset_synonyms()
    reset_spelling()

    corrected = spell_correct(normalize("best gyneacologist"))

    assert corrected == "best gynaecologist"
    assert provider_terms(corrected).split() == ["gynaecologist"]
    assert "gynaecologist => gynecology, ob gyn" in provider_synonym_rules()


def test_build_provider_doc_requires_an_identity():
    assert build_provider_doc(record(user_id=None)) is None
    assert build_provider_doc(record(user_id=None, name="")) is None
    assert build_provider_doc("not a record") is None


def test_build_provider_doc_falls_back_to_first_and_last_name():
    doc = build_provider_doc({"user_id": "provider-2", "firstname": "Ada", "lastname": "Lovelace"})

    assert doc is not None
    assert doc["name"] == "Ada Lovelace"
    assert doc["active"] is True


def test_build_provider_doc_interprets_status():
    assert build_provider_doc(record(status="inactive"))["active"] is False
    assert build_provider_doc(record(status="disabled"))["active"] is False
    assert build_provider_doc(record(status=0))["active"] is False
    assert build_provider_doc(record(status="something-else"))["active"] is True
    assert build_provider_doc(record(status=None))["active"] is True


def test_build_provider_doc_clamps_and_validates_numbers():
    assert build_provider_doc(record(rating=9))["rating"] == 5.0
    assert build_provider_doc(record(rating=-2))["rating"] == 0.0
    assert "rating" not in build_provider_doc(record(rating=float("nan")))
    assert "review_count" not in build_provider_doc(record(review_count="many"))
    assert build_provider_doc(record(review_count=-5))["review_count"] == 0


def test_custom_field_map_is_honoured(monkeypatch):
    monkeypatch.setattr(
        config,
        "PROVIDERS_FIELD_MAP",
        {"provider_id": "id", "name": "full_name", "city": "location.city"},
    )

    doc = build_provider_doc({"id": 7, "full_name": "Dr X", "location": {"city": "NYC"}})

    assert doc is not None
    assert doc["provider_id"] == "7"
    assert doc["name"] == "Dr X"
    assert doc["city"] == "NYC"


def test_tenant_falls_back_to_configuration(monkeypatch):
    monkeypatch.setattr(config, "PROVIDERS_API_TENANT", "default")
    without_tenant = {key: value for key, value in record().items() if key != "tenant_id"}

    assert build_provider_doc(without_tenant)["tenant_id"] == "default"
    assert build_provider_doc(record(tenant_id="other"))["tenant_id"] == "other"


def test_provider_terms_strips_request_words():
    assert provider_terms("Find me a cardiologist near Boston") == "cardiologist boston"
    assert provider_terms("!!") == ""


def test_rating_words_rank_instead_of_narrowing_the_search():
    # Rating is a sort signal; leaving the words in the text would search for a
    # field that is not in search_text and return nobody.
    assert provider_terms("doctors with best ratings") == ""
    assert provider_terms("highest rated cardiologists") == "cardiologists"


def test_a_language_is_detected_and_kept_out_of_the_search_terms():
    assert provider_languages("dermatologist who speaks Urdu") == ["urdu"]
    assert provider_languages("find a cardiologist who speaks French in New York") == ["french"]
    assert provider_terms("dermatologist who speaks Urdu") == "dermatologist"

    # A multi-word language is matched before its shorter label.
    assert provider_languages("doctor who speaks Haitian Creole") == ["haitian creole", "creole"]


def test_a_language_missing_from_the_map_is_still_detected():
    assert provider_languages("who speaks Azerbaijani") == ["azerbaijani"]
    assert provider_terms("who speaks Azerbaijani") == ""
    assert provider_languages("I need a doctor fluent in Catalan") == ["catalan"]
    # A speak cue followed by a provider word names no language.
    assert provider_languages("how do I speak to a doctor") == []


def test_a_language_before_the_gerund_is_not_a_specialty():
    # "urdu speaking dermatologist": the word after "speaking" is the provider.
    assert provider_languages("urdu speaking dermatologist in new york") == ["urdu"]
    assert provider_terms("urdu speaking dermatologist in new york") == "dermatologist york"
    assert provider_languages("spanish speaking staff") == ["spanish"]
    # And a specialty before the gerund is still not a language.
    assert provider_languages("dermatologist speaking urdu") == ["urdu"]


def test_a_primary_language_phrase_is_detected_and_kept_out_of_the_terms():
    text = "find cardiologists whose primary language is English"
    assert provider_languages(text) == ["english"]
    assert provider_terms(text) == "cardiologists"

    # The cue names a language the map does not have as well.
    assert provider_languages("doctors whose primary language is Catalan") == ["catalan"]
    assert provider_terms("doctors whose primary language is Catalan") == ""
    assert provider_languages("language: Spanish") == ["spanish"]


def test_a_language_next_to_a_provider_noun_is_a_directory_request():
    assert language_provider_request("spanish doctors")
    assert language_provider_request("English-speaking doctors")
    assert language_provider_request("haitian creole doctors")
    assert not language_provider_request("Does the platform support Spanish?")
    assert provider_terms("spanish doctors") == ""


def test_language_words_that_are_not_a_filter_stay_in_the_terms():
    # No language cue and no language name: "language services" is not a language.
    assert provider_languages("language services") == []
    assert provider_languages("Which languages are supported in the app?") == []


def test_a_language_filter_is_required_in_both_passes():
    strict = provider_query("dermatology", strict=True, tenant_id="acme", languages=["urdu"])
    strict_filter = strict["function_score"]["query"]["bool"]["filter"]
    assert {"match_phrase": {"languages": "urdu"}} in strict_filter

    relaxed = provider_query("dermatology", strict=False, tenant_id="acme", languages=["urdu"])
    relaxed_filter = relaxed["function_score"]["query"]["bool"]["filter"]
    assert {"match_phrase": {"languages": "urdu"}} in relaxed_filter

    assert provider_query("dermatology", strict=True, tenant_id="acme")["function_score"]["query"]["bool"][
        "filter"
    ] == [{"term": {"active": True}}, {"term": {"tenant_id": "acme"}}]


def test_a_language_alias_matches_the_directory_wording():
    body = provider_query("", strict=True, tenant_id="acme", languages=["farsi"])["function_score"]["query"]["bool"]
    alias = body["filter"][-1]["bool"]

    assert alias["minimum_should_match"] == 1
    assert {"match_phrase": {"languages": "farsi"}} in alias["should"]
    assert {"match_phrase": {"languages": "persian"}} in alias["should"]


def test_a_language_only_query_lists_the_directory_for_that_language():
    body = provider_query("", strict=True, tenant_id="acme", languages=["urdu"])["function_score"]["query"]["bool"]

    assert body["must"] == [{"match_all": {}}]
    assert {"match_phrase": {"languages": "urdu"}} in body["filter"]


def test_provider_query_filters_on_tenant_and_active():
    query = provider_query("cardiology boston", strict=True, tenant_id="acme")
    body = query["function_score"]["query"]["bool"]

    assert {"term": {"active": True}} in body["filter"]
    assert {"term": {"tenant_id": "acme"}} in body["filter"]
    strict_match = body["must"][0]["match"]["search_text"]
    assert strict_match["minimum_should_match"] == "2<75%"
    assert "fuzziness" not in strict_match  # exact pass: spelling already corrected the text

    relaxed = provider_query("cardiology", strict=False, tenant_id="acme")
    relaxed_body = relaxed["function_score"]["query"]["bool"]
    assert relaxed_body["must"][0]["match"]["search_text"]["minimum_should_match"] == "1"
    assert relaxed_body["must"][0]["match"]["search_text"]["fuzziness"] == "AUTO"


def test_provider_query_without_terms_lists_the_tenant_directory():
    body = provider_query("", strict=True, tenant_id="acme")["function_score"]["query"]["bool"]

    assert body["must"] == [{"match_all": {}}]
    assert {"term": {"tenant_id": "acme"}} in body["filter"]


def test_format_providers_builds_a_directory_answer():
    hits = [
        {
            "name": "Dr Ada Lovelace",
            "specialty": "Cardiology",
            "subspecialties": ["Electrophysiology", "Heart failure"],
            "city": "Boston",
            "languages": ["English", "Urdu"],
            "rating": 4.8,
            "review_count": 12,
            "accepting_new_patients": True,
            "next_available": "2026-11-01T09:00:00+00:00",
        }
    ]

    text = format_providers(hits, relaxed=False, searched=True)

    assert "Here are providers that match your search:" in text
    assert "Dr Ada Lovelace" in text
    assert "Cardiology" in text
    assert "Electrophysiology" in text
    assert "Boston" in text
    assert "Speaks English, Urdu" in text
    assert "Rating 4.8 (12 reviews)" in text
    assert "Accepting new patients" in text
    assert "Next available 2026-11-01" in text
    assert "directory search, not medical advice" in text


def test_format_providers_handles_empty_and_relaxed_results():
    hits = [{"name": "Dr X"}]

    assert "I couldn't find any providers" in format_providers([], relaxed=False, searched=True)
    assert "closest providers" in format_providers(hits, relaxed=True, searched=True)
    assert "top-rated providers" in format_providers(hits, relaxed=False, searched=False)


def test_format_providers_explains_a_language_fallback():
    hits = [{"name": "Lisa Newman", "city": "Mountain View", "languages": ["English", "Spanish"]}]

    text = format_providers(hits, relaxed=False, searched=True, language="spanish")

    assert text.startswith("I couldn't find a provider matching all of those details who speaks Spanish.")
    assert "These providers speak Spanish:" in text
    assert "Lisa Newman" in text


class _FakeES:
    """Returns one canned batch of sources per search call, in order."""

    def __init__(self, *batches):
        self.batches = list(batches)
        self.queries = []

    async def search(self, **kwargs):
        self.queries.append(kwargs)
        sources = self.batches.pop(0) if self.batches else []
        return {"hits": {"hits": [{"_source": source} for source in sources]}}


def _use_es(monkeypatch, es):
    monkeypatch.setattr(
        provider_search,
        "get_runtime",
        lambda: SimpleNamespace(es=es, sem_rag=asyncio.Semaphore(1)),
    )


async def test_a_language_and_city_that_match_nobody_falls_back_to_the_language(monkeypatch):
    es = _FakeES(
        [],  # spanish + newyork: no provider matches both
        [{"name": "Lisa Newman", "city": "Mountain View", "languages": ["English", "Spanish"]}],
    )
    _use_es(monkeypatch, es)

    hits, relaxed, language = await provider_search.search_providers(
        SimpleNamespace(tenant_id="acme"), "newyork", ["spanish"]
    )

    assert language == "spanish"
    assert relaxed is False
    assert [hit["name"] for hit in hits] == ["Lisa Newman"]
    assert len(es.queries) == 2
    first, second = (query["query"]["function_score"]["query"]["bool"] for query in es.queries)
    assert first["must"][0]["match"]["search_text"]["query"] == "newyork"
    assert second["must"] == [{"match_all": {}}]
    assert {"match_phrase": {"languages": "spanish"}} in second["filter"]


async def test_a_language_fallback_with_no_speakers_answers_no_match(monkeypatch):
    es = _FakeES([], [])
    _use_es(monkeypatch, es)

    hits, relaxed, language = await provider_search.search_providers(
        SimpleNamespace(tenant_id="acme"), "newyork", ["klingon"]
    )

    assert (hits, relaxed, language) == ([], False, None)


async def test_a_language_query_that_matches_does_not_retry(monkeypatch):
    es = _FakeES([{"name": "Dr Urdu", "languages": ["Urdu"]}])
    _use_es(monkeypatch, es)

    hits, relaxed, language = await provider_search.search_providers(
        SimpleNamespace(tenant_id="acme"), "dermatologist", ["urdu"]
    )

    assert len(es.queries) == 1
    assert language is None
    assert relaxed is False
    assert [hit["name"] for hit in hits] == ["Dr Urdu"]
