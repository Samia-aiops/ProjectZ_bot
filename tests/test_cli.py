from __future__ import annotations

import chat_search.cli as cli
from chat_search import config
from chat_search.cli import _build_parser


def test_serve_is_the_default_command():
    assert _build_parser().parse_args([]).command is None


def test_bare_invocation_serves_with_defaults(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "_serve", lambda host, port: calls.append((host, port)) or 0)

    assert cli.main([]) == 0
    assert calls == [("0.0.0.0", config.PORT)]


def test_serve_flags_are_honoured(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "_serve", lambda host, port: calls.append((host, port)) or 0)

    assert cli.main(["serve", "--host", "127.0.0.1", "--port", "9001"]) == 0
    assert calls == [("127.0.0.1", 9001)]


def test_index_documents_is_a_batch_command():
    assert _build_parser().parse_args(["index-documents"]).command == "index-documents"


def test_provider_sync_command_is_still_available():
    assert _build_parser().parse_args(["sync-providers"]).command == "sync-providers"


def test_provider_synonym_sync_command_is_available():
    assert _build_parser().parse_args(["sync-provider-synonyms"]).command == "sync-provider-synonyms"
