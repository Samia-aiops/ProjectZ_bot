"""Elasticsearch client factory.

Serving uses the read-only client; indexing jobs pass ``write=True``.
"""

from __future__ import annotations

from typing import Any

from elasticsearch import AsyncElasticsearch

from .. import config


def make_es_client(write: bool = False) -> AsyncElasticsearch:
    kwargs: dict[str, Any] = {
        "request_timeout": max(config.ES_TIMEOUT, 60),
        "verify_certs": bool(config.ES_CA_CERTS),
        "max_retries": 3,
        "retry_on_timeout": True,
    }

    if write:
        if not config.ES_SYNC_API_KEY:
            raise RuntimeError("ES_SYNC_API_KEY is required for write operations")
        kwargs["api_key"] = config.ES_SYNC_API_KEY
    elif config.ES_API_KEY:
        kwargs["api_key"] = config.ES_API_KEY
    elif config.ES_USER and config.ES_PASSWORD:
        kwargs["basic_auth"] = (config.ES_USER, config.ES_PASSWORD)
    else:
        raise RuntimeError("Elasticsearch credentials are not configured")

    if config.ES_CA_CERTS:
        kwargs["ca_certs"] = config.ES_CA_CERTS

    return AsyncElasticsearch(config.ES_URL, **kwargs)
