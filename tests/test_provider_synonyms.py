from __future__ import annotations

import json

import pytest

from chat_search import config
from chat_search.providers.indexing.schema import (
    PROVIDERS_MAPPING,
    ensure_providers_index,
    provider_synonym_rules,
    providers_settings,
)
from chat_search.synonyms import reset_synonyms

CATALOG = {
    "specialties": {
        "pediatrics": ["pediatrician", "child specialist", "children's doctor"],
        "gynecology": ["gynecologist", "ob-gyn"],
        "ophthalmology": ["ophthalmologist"],
        "appointment booking": ["book a visit"],
        "file attachment": ["attachment"],
    },
}

TEXT_FIELDS = (
    "name",
    "specialty",
    "subspecialties",
    "conditions_treated",
    "languages",
    "city",
    "bio",
    "search_text",
)


@pytest.fixture(autouse=True)
def catalogue(monkeypatch):
    monkeypatch.setattr(config, "SYNONYMS_ENABLED", True)
    monkeypatch.setattr(config, "SYNONYMS_JSON", json.dumps(CATALOG))
    reset_synonyms()
    yield
    reset_synonyms()


def test_rules_are_one_way_and_add_the_directory_label():
    rules = provider_synonym_rules()

    assert "child specialist => pediatrics" in rules
    assert "childrens doctor => pediatrics" in rules  # apostrophes normalized
    assert "pediatricians => pediatrics" in rules  # generated plural
    assert "gynecologist => gynecology, ob gyn" in rules
    assert "ob gyn => gynecology, ob gyn" in rules
    assert "gynecology => gynecology, ob gyn" in rules
    assert "ophthalmologist => ophthalmology, eye doctor" in rules
    assert "ophthalmology => ophthalmology, eye doctor" in rules


def test_faq_groups_are_not_provider_synonyms():
    rules = " ".join(provider_synonym_rules())

    assert "appointment" not in rules
    assert "attachment" not in rules


def test_synonyms_run_at_search_time_only():
    analysis = providers_settings()["analysis"]

    assert analysis["filter"]["provider_synonyms"]["type"] == "synonym_graph"
    assert analysis["char_filter"]["strip_apostrophes"]["type"] == "pattern_replace"
    assert analysis["analyzer"]["provider_text"]["char_filter"] == ["strip_apostrophes"]
    assert analysis["analyzer"]["provider_search"]["char_filter"] == ["strip_apostrophes"]
    assert "provider_synonyms" not in analysis["analyzer"]["provider_text"]["filter"]
    assert analysis["analyzer"]["provider_search"]["filter"] == [
        "lowercase",
        "asciifolding",
        "provider_synonyms",
        "stop",
        "porter_stem",
    ]
    for field in TEXT_FIELDS:
        assert PROVIDERS_MAPPING["properties"][field]["search_analyzer"] == "provider_search"


def test_disabled_synonyms_still_create_a_working_search_analyzer(monkeypatch):
    monkeypatch.setattr(config, "SYNONYMS_ENABLED", False)
    reset_synonyms()

    analysis = providers_settings()["analysis"]

    assert "filter" not in analysis
    assert analysis["analyzer"]["provider_search"]["filter"] == [
        "lowercase",
        "asciifolding",
        "stop",
        "porter_stem",
    ]


class _Indices:
    def __init__(self, *, exists: bool, settings=None, mapping=None):
        self._exists = exists
        self._settings = settings or {}
        self._mapping = mapping or {}
        self.calls: list[tuple[str, dict]] = []

    async def exists(self, *, index):
        self.calls.append(("exists", {}))
        return self._exists

    async def create(self, **kwargs):
        self.calls.append(("create", kwargs))

    async def get_settings(self, *, index, flat_settings=False):
        self.calls.append(("get_settings", {}))
        return {index: {"settings": self._settings}}

    async def get_mapping(self, *, index):
        self.calls.append(("get_mapping", {}))
        return {index: {"mappings": {"properties": self._mapping}}}

    async def close(self, *, index):
        self.calls.append(("close", {}))

    async def put_settings(self, *, index, settings):
        self.calls.append(("put_settings", settings))

    async def open(self, *, index):
        self.calls.append(("open", {}))

    async def put_mapping(self, *, index, properties):
        self.calls.append(("put_mapping", properties))


class _ES:
    def __init__(self, indices: _Indices):
        self.indices = indices


async def test_a_new_index_gets_the_search_time_graph():
    indices = _Indices(exists=False)

    await ensure_providers_index(_ES(indices))

    assert [name for name, _ in indices.calls] == ["exists", "create"]
    created = indices.calls[1][1]
    assert created["settings"]["analysis"]["filter"]["provider_synonyms"]["type"] == "synonym_graph"
    assert created["mappings"]["properties"]["search_text"]["search_analyzer"] == "provider_search"


async def test_an_existing_index_is_upgraded_from_index_time_synonyms():
    old_settings = {
        "index.analysis.filter.provider_synonyms.type": "synonym",
        "index.analysis.filter.provider_synonyms.synonyms": ["cardiologist => cardiology"],
        "index.analysis.analyzer.provider_text.filter": [
            "lowercase",
            "asciifolding",
            "provider_synonyms",
            "stop",
            "porter_stem",
        ],
    }
    old_mapping = {"search_text": {"type": "text", "analyzer": "provider_text"}}
    indices = _Indices(exists=True, settings=old_settings, mapping=old_mapping)

    await ensure_providers_index(_ES(indices))

    assert [name for name, _ in indices.calls] == [
        "exists",
        "get_settings",
        "get_mapping",
        "close",
        "put_settings",
        "open",
        "put_mapping",
    ]
    upgraded = indices.calls[4][1]["analysis"]
    assert upgraded["filter"]["provider_synonyms"]["type"] == "synonym_graph"
    assert "provider_synonyms" not in upgraded["analyzer"]["provider_text"]["filter"]
    assert "provider_synonyms" in upgraded["analyzer"]["provider_search"]["filter"]
    assert indices.calls[-1][1]["search_text"]["search_analyzer"] == "provider_search"


def _installed_settings() -> dict:
    analysis = providers_settings()["analysis"]
    return {
        "index.analysis.analyzer.provider_search.filter": analysis["analyzer"]["provider_search"]["filter"],
        "index.analysis.filter.provider_synonyms.type": "synonym_graph",
        "index.analysis.filter.provider_synonyms.synonyms": analysis["filter"]["provider_synonyms"]["synonyms"],
    }


async def test_matching_settings_and_mapping_need_no_update():
    mapping = {
        field: {"type": "text", "analyzer": "provider_text", "search_analyzer": "provider_search"}
        for field in TEXT_FIELDS
    }
    indices = _Indices(exists=True, settings=_installed_settings(), mapping=mapping)

    await ensure_providers_index(_ES(indices))

    assert [name for name, _ in indices.calls] == ["exists", "get_settings", "get_mapping"]
