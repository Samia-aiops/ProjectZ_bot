"""
Weekly provider sync, scheduled by Airflow.

Runs the same image as the chatbot, one-shot:  python -m chat_search sync-providers
  - exit 0  -> everything done
  - exit 1  -> the sync finished but refused to remove providers (API returned far less than
               before, bulk indexing errors, ...) or failed outright. Airflow retries, then alerts.

The sync is idempotent (documents are upserted by "{tenant}-{provider_id}") and removal only
happens after a complete, healthy pull, so retries and manual re-runs are safe.

Manual trigger (any time):
    airflow dags trigger provider_sync_weekly

Requires: Airflow 2.4+ and apache-airflow-providers-docker. The job needs only the settings below:
no JWT secret, LLM key, Qdrant or Redis.
Create these Airflow Variables (names containing "token"/"api_key" are masked in the UI and logs):
  es_url                 Elasticsearch endpoint (e.g. http://elasticsearch:9200)
  es_sync_api_key        API key with write access to the providers index
  providers_api_url      source API endpoint
  providers_api_token    source API bearer token

Optional Variables (defaults shown):
  index_image            faq-bot:latest
  es_providers_index     providers
  providers_api_tenant   (empty)
  providers_api_page_size 100
  providers_api_items_key data
  providers_api_params   {}
  providers_field_map    {}
  providers_api_timeout  30
  provider_sync_min_ratio 0.5

The index name and field map must match what FastAPI serves; otherwise the bot reads
an index the sync never wrote to. Keep them in Airflow Variables, not in code.
"""

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator

default_args = {
    "owner": "ai-team",
    "retries": 3,
    "retry_delay": timedelta(minutes=30),
    "retry_exponential_backoff": True,
    "email_on_failure": True,  # or use on_failure_callback to post to Slack / PagerDuty
}

with DAG(
    dag_id="provider_sync_weekly",
    schedule="0 2 * * 0",  # Sundays 02:00 (the Airflow scheduler's timezone)
    start_date=datetime(2026, 1, 1),
    catchup=False,  # never replay missed weeks: only the latest data matters
    max_active_runs=1,  # two syncs must never overlap
    default_args=default_args,
    tags=["chatbot", "elasticsearch"],
) as dag:
    DockerOperator(
        task_id="sync_providers",
        image="{{ var.value.get('index_image', 'faq-bot:latest') }}",
        command=["python", "-m", "chat_search", "sync-providers"],
        environment={
            "ES_URL": "{{ var.value.es_url }}",
            "ES_SYNC_API_KEY": "{{ var.value.es_sync_api_key }}",
            "ES_PROVIDERS_INDEX": "{{ var.value.get('es_providers_index', 'providers') }}",
            "PROVIDERS_API_URL": "{{ var.value.providers_api_url }}",
            "PROVIDERS_API_TOKEN": "{{ var.value.providers_api_token }}",
            "PROVIDERS_API_TENANT": "{{ var.value.get('providers_api_tenant') or '' }}",
            "PROVIDERS_API_PAGE_SIZE": "{{ var.value.get('providers_api_page_size') or '100' }}",
            "PROVIDERS_API_ITEMS_KEY": "{{ var.value.get('providers_api_items_key') or 'data' }}",
            "PROVIDERS_API_PARAMS": "{{ var.value.get('providers_api_params') or '{}' }}",
            "PROVIDERS_FIELD_MAP": "{{ var.value.get('providers_field_map') or '{}' }}",
            "PROVIDERS_API_TIMEOUT": "{{ var.value.get('providers_api_timeout') or '30' }}",
            "PROVIDER_SYNC_MIN_RATIO": "{{ var.value.get('provider_sync_min_ratio') or '0.5' }}",
        },
        docker_url=os.getenv("DOCKER_HOST", "unix://var/run/docker.sock"),
        # The compose stack names its project chat_search, so the job resolves the same
        # elasticsearch/qdrant hosts as the app. Point ES_URL at a reachable endpoint.
        network_mode="chat_search_default",
        auto_remove="success",  # keep failed containers around for debugging
        mount_tmp_dir=False,
        execution_timeout=timedelta(hours=1),
    )
