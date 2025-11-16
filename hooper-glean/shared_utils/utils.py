import logging
from typing import Optional

import time
from typing import Optional
from google.cloud.firestore import Client
from firebase_admin import storage, firestore, credentials, get_app, initialize_app

import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration
from sentry_sdk.integrations.redis import RedisIntegration
from sentry_sdk.integrations.logging import LoggingIntegration

from .constants import (
    SERVICE_ACCOUNT_FILE,
    FIREBASE_STORAGE_BUCKET,
    GCLOUD_FIREBASE_PROJECT_ID,
)


def tick():
    return int(time.time())


def get_firebase_client(
    service_account_file: Optional[str] = SERVICE_ACCOUNT_FILE,
    storage_bucket: Optional[str] = FIREBASE_STORAGE_BUCKET,
) -> Client:
    """Instantiate a firebase client.

    Args:
        service_account_file: path to service account file
        storage_bucket: firebase storage bucket

    Return:
        client: Firebase client instance
    """
    firebase_credentials = credentials.Certificate(service_account_file)
    try:
        get_app()
    except ValueError:
        initialize_app(firebase_credentials, {
            "storageBucket": storage_bucket,
            "projectId": GCLOUD_FIREBASE_PROJECT_ID,
        })
    client = firestore.client()
    return client


def get_task_id_from_name(task_name: str) -> str:
    """Return task ID (FB document object) from task name

    Args:
        task_name: Name returned by CloudTasks

    Return:
        task_id
    """
    task_id = task_name.split("/")[-1]
    return task_id


def upload_file_to_firebase(file_path: str, blob_path: str) -> str:
    """Upload a file to firebase.

    Args:
        file_path: Local file path to upload
        blob_path: Path to upload to in Firebase

    Return:
        public_url: URL to the uploaded file
    """
    bucket = storage.bucket()
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(file_path)
    blob.make_public()

    return blob.public_url


# Enhanced Sentry configuration
def configure_sentry(
    sdk_dsn: Optional[str] = None,
    traces_sample_rate: float = 0.1,
    app_version: str = "unknown",
    environment: str = "production",
):
    """Configure Sentry with proper integrations and settings.

    Args:
        sdk_dsn: Required. Private DSN link for Sentry.
        traces_sample_rate: Rate at which the full trace is provided
        environment: production or development
    """
    if not sdk_dsn:
        return  # nothing to do if no DSN provided

    # Configure logging integration to capture logs as breadcrumbs
    logging_integration = LoggingIntegration(
        level=logging.INFO,  # Capture info and above as breadcrumbs
        event_level=logging.ERROR,  # Send errors as events
    )

    sentry_sdk.init(
        dsn=sdk_dsn,
        # Add integrations for better context
        integrations=[
            FastApiIntegration(),
            SqlalchemyIntegration(),
            RedisIntegration(),
            logging_integration,
        ],
        # Performance monitoring
        enable_tracing=True,
        traces_sample_rate=traces_sample_rate,
        # Release tracking
        release=app_version,
        environment=environment,
        # Error filtering
        before_send=filter_sentry_events,
        # Additional options
        attach_stacktrace=True,
        send_default_pii=False,  # Don't send personally identifiable information
        max_breadcrumbs=50,
    )


def filter_sentry_events(event, hint):
    """Filter out events we don't want to send to Sentry."""
    # Don't send 404 errors
    if "exc_info" in hint:
        exc_type, exc_value, tb = hint["exc_info"]
        if exc_value.status_code == 404:
            return None
    # Filter out health check requests
    if event.get("request", {}).get("url", "").endswith("/"):
        return None
    return event
