"""
Weekly Qdrant document index, scheduled by Airflow.

Runs the same image as the chatbot, one-shot:  python -m chat_search index-documents
  - exit 0  -> PDFs were chunked, embedded and upserted; the corpus fingerprint was published
               to Redis so the semantic cache invalidates correctly
  - exit 1  -> models could not load, DATA_DIR was missing, or Qdrant failed. Airflow retries,
               then alerts.

The job is idempotent: chunks are upserted by deterministic point id, so retries and
manual re-runs are safe. It needs only Qdrant, Redis (for the corpus fingerprint), the
documents volume and the embedding models baked into the image. No JWT secret, LLM key
or Elasticsearch access.

Manual trigger (any time):
    airflow dags trigger document_index_weekly

Airflow Variables (names containing "token"/"api_key" are masked in the UI and logs):
    qdrant_url             default http://qdrant:6333
    qdrant_api_key         default empty (local Qdrant)
    redis_url              default redis://redis:6379/0
    index_image            default faq-bot:latest
    qdrant_collection      default faq_documents_v2
    embedding_model        default BAAI/bge-small-en-v1.5
    sparse_model           default Qdrant/bm25
    chunk_tokens           default 300
    chunk_max_tokens       default 380
    chunk_overlap_tokens   default 40
    min_chunk_tokens       default 60
    whole_doc_tokens       default 350

The collection name, models and chunk settings must match the serving process, otherwise
the bot queries vectors that no run produced. Change them in the shared Airflow Variables,
not in this DAG.

Airflow environment variable:
    DOCUMENTS_PATH   absolute HOST path containing the PDFs (mounted at /data)
"""

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

default_args = {
    "owner": "ai-team",
    "retries": 2,
    "retry_delay": timedelta(minutes=15),
    "retry_exponential_backoff": True,
    "email_on_failure": True,  # or use on_failure_callback to post to Slack / PagerDuty
}

# Set on the Airflow scheduler/worker (compose does this from .env). DockerOperator
# asks the host Docker daemon to bind this path, so it must exist on the host.
DOCUMENTS_PATH = os.getenv("DOCUMENTS_PATH", "/opt/chat_search/data")

with DAG(
    dag_id="document_index_weekly",
    schedule="0 3 * * 0",  # Sundays 03:00 (the Airflow scheduler's timezone)
    start_date=datetime(2026, 1, 1),
    catchup=False,  # only the latest corpus matters
    max_active_runs=1,  # two index runs must never overlap
    default_args=default_args,
    tags=["chatbot", "qdrant"],
) as dag:
    DockerOperator(
        task_id="index_documents",
        image="{{ var.value.get('index_image', 'faq-bot:latest') }}",
        command=["python", "-m", "chat_search", "index-documents"],
        environment={
            "QDRANT_URL": "{{ var.value.get('qdrant_url', 'http://qdrant:6333') }}",
            "QDRANT_API_KEY": "{{ var.value.get('qdrant_api_key', '') }}",
            "QDRANT_COLLECTION": "{{ var.value.get('qdrant_collection', 'faq_documents_v2') }}",
            "REDIS_URL": "{{ var.value.get('redis_url', 'redis://redis:6379/0') }}",
            "DATA_DIR": "/data",
            # Must match the serving process, otherwise vectors/chunk ids differ.
            "EMBEDDING_MODEL": "{{ var.value.get('embedding_model', 'BAAI/bge-small-en-v1.5') }}",
            "SPARSE_MODEL": "{{ var.value.get('sparse_model', 'Qdrant/bm25') }}",
            "CHUNK_TOKENS": "{{ var.value.get('chunk_tokens', '300') }}",
            "CHUNK_MAX_TOKENS": "{{ var.value.get('chunk_max_tokens', '380') }}",
            "CHUNK_OVERLAP_TOKENS": "{{ var.value.get('chunk_overlap_tokens', '40') }}",
            "MIN_CHUNK_TOKENS": "{{ var.value.get('min_chunk_tokens', '60') }}",
            "WHOLE_DOC_TOKENS": "{{ var.value.get('whole_doc_tokens', '350') }}",
        },
        mounts=[Mount(source=DOCUMENTS_PATH, target="/data", type="bind", read_only=True)],
        docker_url=os.getenv("DOCKER_HOST", "unix://var/run/docker.sock"),
        network_mode="chat_search_default",
        auto_remove="success",  # keep failed containers around for debugging
        mount_tmp_dir=False,
        execution_timeout=timedelta(hours=2),
    )
