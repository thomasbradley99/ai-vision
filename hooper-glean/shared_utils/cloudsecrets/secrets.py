import re
from urllib.parse import urlparse
from dataclasses import dataclass


@dataclass
class SecretConfig:
    """Configuration for a secret"""

    name: str
    description: str
    env_var: str


SECRET_CONFIGS = {
    # Replicate secrets
    "replicate-api-token": SecretConfig(
        name="replicate-api-token",
        description="API token for Replicate ML service",
        env_var="REPLICATE_API_TOKEN",
    ),
    "replicate-hoop-fast-model": SecretConfig(
        name="replicate-hoop-fast-model",
        description="Replicate ID for a fast hoop detector",
        env_var="REPLICATE_HOOP_FAST_MODEL",
    ),
    "replicate-hoop-slow-model": SecretConfig(
        name="replicate-hoop-slow-model",
        description="Replicate ID for a slow hoop detector",
        env_var="REPLICATE_HOOP_SLOW_MODEL",
    ),
    "replicate-clip-model": SecretConfig(
        name="replicate-clip-model",
        description="Replicate ID for a CLIP model",
        env_var="REPLICATE_CLIP_MODEL",
    ),
    # Google Cloud SQL secrets
    "cloudsql-connection-name": SecretConfig(
        name="cloudsql-connection-name",
        description="Cloud SQL connection name (project:region:instance)",
        env_var="CLOUDSQL_CONNECTION_NAME",
    ),
    "cloudsql-user": SecretConfig(
        name="cloudsql-user",
        description="Cloud SQL database username",
        env_var="CLOUDSQL_USER",
    ),
    "cloudsql-password": SecretConfig(
        name="cloudsql-password",
        description="Cloud SQL database password",
        env_var="CLOUDSQL_PASSWORD",
    ),
    "cloudsql-db": SecretConfig(
        name="cloudsql-db",
        description="Cloud SQL database name",
        env_var="CLOUDSQL_DB",
    ),
    "cloudsql-private-ip": SecretConfig(
        name="cloudsql-private-ip",
        description="Whether to use private IP for Cloud SQL connection",
        env_var="CLOUDSQL_PRIVATE_IP",
    ),
    # Google Cloud Run secrets
    "cloudrun-cpu-service-url": SecretConfig(
        name="cloudrun-cpu-service-url",
        description="Cloud Run (CPU) service URL",
        env_var="CLOUDRUN_CPU_SERVICE_URL",
    ),
    "cloudrun-cpu-invoker-sa": SecretConfig(
        name="cloudrun-cpu-invoker-sa",
        description="Cloud Run (CPU) invoker service account",
        env_var="CLOUDRUN_CPU_INVOKER_SA",
    ),
    "cloudrun-gpu-service-url": SecretConfig(
        name="cloudrun-gpu-service-url",
        description="Cloud Run (GPU) service URL",
        env_var="CLOUDRUN_GPU_SERVICE_URL",
    ),
    "cloudrun-gpu-invoker-sa": SecretConfig(
        name="cloudrun-gpu-invoker-sa",
        description="Cloud Run (GPU) invoker service account",
        env_var="CLOUDRUN_GPU_INVOKER_SA",
    ),
    # SendGrid secrets
    "sendgrid-api-key": SecretConfig(
        name="sendgrid-api-key",
        description="SendGrid API key for email notifications",
        env_var="SENDGRID_API_KEY",
    ),
    # Sentry secrets
    "sentry-sdk-dsn": SecretConfig(
        name="sentry-sdk-dsn",
        description="Sentry DSN for error tracking and monitoring",
        env_var="SENTRY_SDK_DSN",
    ),
    # Slack secrets
    "slack-webhook-url": SecretConfig(
        name="slack-webhook-url",
        description="Slack webhook URL for notifications",
        env_var="SLACK_WEBHOOK_URL",
    ),
    # App configuration secrets
    "app-ios-version": SecretConfig(
        name="app-ios-version",
        description="Current iOS app version for version checking",
        env_var="APP_IOS_VERSION",
    ),
    "app-android-version": SecretConfig(
        name="app-android-version",
        description="Current Android app version for version checking",
        env_var="APP_ANDROID_VERSION",
    ),
    "app-outage": SecretConfig(
        name="app-outage",
        description="App outage flag to turn on/off maintenance banner",
        env_var="APP_OUTAGE",
    ),
    "inf-outage": SecretConfig(
        name="inf-outage",
        description="Inference outage flag to turn on/off waittime banner",
        env_var="INF_OUTAGE",
    ),
    # Admin authentication secrets
    "admin-auth-username": SecretConfig(
        name="admin-auth-username",
        description="Admin panel authentication username",
        env_var="ADMIN_AUTH_USERNAME",
    ),
    "admin-auth-password": SecretConfig(
        name="admin-auth-password",
        description="Admin panel authentication password",
        env_var="ADMIN_AUTH_PASSWORD",
    ),
}


def validate_secret_format(secret_name: str, value: str) -> bool:
    """Validate secret value format.

    Args:
        secret_name (str): The name of the secret.
        value (str): The value of the secret.

    Returns:
        bool: True if the secret value is valid, False otherwise.
    """
    if not value or not value.strip():
        return False

    # Add specific validations
    if secret_name == "replicate-api-token" and not value.startswith("r8_"):
        return False

    if secret_name == "slack-webhook-url" and not value.startswith(
        "https://hooks.slack.com"
    ):
        return False

    if secret_name == "sentry-sdk-dsn" and not value.startswith("https://"):
        return False

    if secret_name == "cloudsql-connection-name" and value.count(":") != 2:
        return False

    if secret_name in ["app-ios-version", "app-android-version"]:
        # The versions must be numbers
        if not value.isdigit():
            return False

    if secret_name == "cloud-run-service-url":
        # Cloud Run Service URL validation
        return validate_cloud_run_url(value)

    if secret_name == "cloud-run-invoker-sa":
        # Service Account Email validation
        return validate_service_account_email(value)

    if secret_name == "app-outage" or secret_name == "inf-outage":
        return validate_boolean_flag(value)

    return True


def validate_cloud_run_url(url: str) -> bool:
    """Validate Cloud Run service URL format"""
    try:
        parsed = urlparse(url)

        # Must be HTTPS in production (allow HTTP for local dev)
        if parsed.scheme not in ["https", "http"]:
            return False

        # Must have a hostname
        if not parsed.hostname:
            return False

        # Cloud Run URLs typically end with .a.run.app or .run.app
        if parsed.hostname.endswith((".a.run.app", ".run.app")):
            return True

        # Allow localhost for development
        if parsed.hostname in ["localhost", "127.0.0.1"]:
            return True

        # Allow custom domains (basic hostname validation)
        if "." in parsed.hostname and len(parsed.hostname) > 3:
            return True

        return False

    except Exception:
        return False


def validate_service_account_email(email: str) -> bool:
    """Validate Google Cloud service account email format"""
    # Format: service-account-name@project-id.iam.gserviceaccount.com
    pattern = r"^[a-z0-9\-]+@[a-z0-9\-]+\.iam\.gserviceaccount\.com$"

    if not re.match(pattern, email):
        return False

    # Additional checks
    parts = email.split("@")
    if len(parts) != 2:
        return False

    account_name, domain = parts

    # Service account name validation
    if len(account_name) < 6 or len(account_name) > 30:
        return False

    # Domain validation
    if not domain.endswith(".iam.gserviceaccount.com"):
        return False

    project_id = domain.replace(".iam.gserviceaccount.com", "")

    # Basic project ID validation (6-30 chars, lowercase, numbers, hyphens)
    if not re.match(r"^[a-z0-9\-]{6,30}$", project_id):
        return False

    return True


def validate_boolean_flag(value: str) -> bool:
    """Validate boolean flag values"""
    return value.lower() in ["0", "1"]
