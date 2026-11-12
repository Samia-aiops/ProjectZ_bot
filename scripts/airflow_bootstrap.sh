#!/usr/bin/env bash
# Start Airflow standalone with the admin password taken from the environment.
#
# `airflow standalone` only sets an admin password when it creates the user for
# the first time: on later starts the password in the database wins and the
# value in .env would silently do nothing. Applying it here keeps .env the
# single source of truth, and also clears the pid files a hard container stop
# leaves in the volume (standalone refuses to start when airflow-webserver.pid
# names a process that no longer exists).
set -euo pipefail

: "${AIRFLOW_ADMIN_USERNAME:=admin}"
: "${AIRFLOW_ADMIN_PASSWORD:?AIRFLOW_ADMIN_PASSWORD is not set (add it to .env)}"

home="${AIRFLOW_HOME:-/opt/airflow}"

rm -f "$home"/airflow-*.pid

airflow db migrate

python - <<'PY'
import os

from airflow.utils.cli_app_builder import get_application_builder
from flask_appbuilder.security.manager import BaseSecurityManager

username = os.environ["AIRFLOW_ADMIN_USERNAME"]
password = os.environ["AIRFLOW_ADMIN_PASSWORD"]
email = os.environ.get("AIRFLOW_ADMIN_EMAIL", "admin@example.com")

with get_application_builder() as appbuilder:
    security_manager = appbuilder.sm
    user = security_manager.find_user(username=username)
    if user is None:
        role = security_manager.find_role("Admin")
        security_manager.add_user(username, "Admin", "User", email, role, password)
        print(f"Created Airflow admin user {username!r}")
    else:
        # Airflow's reset_password flashes a message (needs a request context and
        # a secret key); the base FAB implementation only updates the hash.
        BaseSecurityManager.reset_password(security_manager, user.id, password)
        print(f"Applied the configured password to Airflow admin user {username!r}")

# standalone prints this file in its startup banner; keep it equal to the
# configured password instead of a value generated on the first run.
path = os.path.join(os.environ.get("AIRFLOW_HOME", "/opt/airflow"), "standalone_admin_password.txt")
with open(path, "w") as handle:
    handle.write(password)
os.chmod(path, 0o600)
PY

exec /entrypoint standalone
