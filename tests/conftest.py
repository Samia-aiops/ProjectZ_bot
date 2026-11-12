import pytest

from chat_search.runtime import clear_runtime
from chat_search.spelling import warm_spelling
from chat_search.synonyms import reset_synonyms
from chat_search.understanding import warm_understanding


@pytest.fixture(scope="session", autouse=True)
def _warm_models():
    """Load the models once: a cold load costs seconds and would trip the
    latency budgets of tests that are not about loading. The modules that
    inspect the cold path (spelling, understanding) reset the caches themselves."""
    warm_understanding()
    warm_spelling()
    yield


@pytest.fixture(autouse=True)
def _clear_runtime():
    clear_runtime()
    reset_synonyms()
    yield
    clear_runtime()
    reset_synonyms()
