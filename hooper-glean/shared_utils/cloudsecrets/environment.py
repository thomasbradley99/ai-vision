import socket
from os import environ as env
from dotenv import load_dotenv
from typing import Optional

from ..constants import GCLOUD_DEFAULT_LOCATION, GCLOUD_COMPUTE_PROJECT_ID
from .manager import CloudSecretsManager


def env_var_to_secret_name(env_var_name: str) -> str:
    """Converts an environment variable name to a secret name.

    Args:
        env_var_name (str): The environment variable name to convert.

    Returns:
        str: The converted secret name.
    """
    return env_var_name.replace("_", "-").lower()


class Environment:
    """Environment configuration using Cloud Secret Manager with fallbacks to environment variables."""

    def __init__(self, project_id: Optional[str] = None):
        # Initialize all your secrets
        self._load_secrets(project_id)

    def _load_secrets(self, project_id: Optional[str] = None):
        """Load all secrets at startup."""

        # Fetch variables stored in `.env` file - it searches for one.
        load_dotenv()

        # Load the environment
        self.environment = env.get("ENVIRONMENT", "production")

        # Database secrets
        self.redis_url = env.get("CACHE_URL", "redis://localhost:6379")

        if self.environment == "production":
            if project_id is None:
                raise ValueError("project_id is required for production environment")

            # Initialize Cloud Secret Manager
            secrets_manager = CloudSecretsManager(project_id)

            # Admin secrets
            self.admin_auth_username = secrets_manager.get_secret_value(
                env_var_to_secret_name("ADMIN_AUTH_USERNAME")
            )
            self.admin_auth_password = secrets_manager.get_secret_value(
                env_var_to_secret_name("ADMIN_AUTH_PASSWORD")
            )

            # Replicate secrets
            self.replicate_api_token = secrets_manager.get_secret_value(
                env_var_to_secret_name("REPLICATE_API_TOKEN")
            )
            self.replicate_hoop_fast_model = secrets_manager.get_secret_value(
                env_var_to_secret_name("REPLICATE_HOOP_FAST_MODEL")
            )
            self.replicate_hoop_slow_model = secrets_manager.get_secret_value(
                env_var_to_secret_name("REPLICATE_HOOP_SLOW_MODEL")
            )
            self.replicate_clip_model = secrets_manager.get_secret_value(
                env_var_to_secret_name("REPLICATE_CLIP_MODEL")
            )

            # GCloud secrets
            self.cloudsql_private_ip = secrets_manager.get_secret_value(
                env_var_to_secret_name("CLOUDSQL_PRIVATE_IP")
            )
            self.cloudsql_connection_name = secrets_manager.get_secret_value(
                env_var_to_secret_name("CLOUDSQL_CONNECTION_NAME")
            )
            self.cloudsql_user = secrets_manager.get_secret_value(
                env_var_to_secret_name("CLOUDSQL_USER")
            )
            self.cloudsql_password = secrets_manager.get_secret_value(
                env_var_to_secret_name("CLOUDSQL_PASSWORD")
            )
            self.cloudsql_db = secrets_manager.get_secret_value(
                env_var_to_secret_name("CLOUDSQL_DB")
            )

            # CloudRun secrets
            self.cloudrun_cpu_service_url = secrets_manager.get_secret_value(
                env_var_to_secret_name("CLOUDRUN_CPU_SERVICE_URL")
            )
            self.cloudrun_cpu_invoker_sa = secrets_manager.get_secret_value(
                env_var_to_secret_name("CLOUDRUN_CPU_INVOKER_SA")
            )
            self.cloudrun_gpu_service_url = secrets_manager.get_secret_value(
                env_var_to_secret_name("CLOUDRUN_GPU_SERVICE_URL")
            )
            self.cloudrun_gpu_invoker_sa = secrets_manager.get_secret_value(
                env_var_to_secret_name("CLOUDRUN_GPU_INVOKER_SA")
            )

            # Sendgrid secrets
            self.sendgrid_api_key = secrets_manager.get_secret_value(
                env_var_to_secret_name("SENDGRID_API_KEY")
            )

            # Sentry secrets
            self.sentry_sdk_dsn = secrets_manager.get_secret_value(
                env_var_to_secret_name("SENTRY_SDK_DSN")
            )

            # Slack secrets
            self.slack_webhook_url = secrets_manager.get_secret_value(
                env_var_to_secret_name("SLACK_WEBHOOK_URL")
            )

            # App configuration
            self.app_ios_version = secrets_manager.get_secret_value(
                env_var_to_secret_name("APP_IOS_VERSION")
            )
            self.app_android_version = secrets_manager.get_secret_value(
                env_var_to_secret_name("APP_ANDROID_VERSION")
            )
            self.app_outage = int(
                secrets_manager.get_secret_value(env_var_to_secret_name("APP_OUTAGE"))
                or "0"
            )
            self.inf_outage = int(
                secrets_manager.get_secret_value(env_var_to_secret_name("INF_OUTAGE"))
                or "0"
            )
        else:
            # Admin secrets
            self.admin_auth_username = env.get("ADMIN_AUTH_USERNAME")
            self.admin_auth_password = env.get("ADMIN_AUTH_PASSWORD")

            # Replicate secrets
            self.replicate_api_token = env.get("REPLICATE_API_TOKEN")
            self.replicate_hoop_fast_model = env.get("REPLICATE_HOOP_FAST_MODEL")
            self.replicate_hoop_slow_model = env.get("REPLICATE_HOOP_SLOW_MODEL")
            self.replicate_clip_model = env.get("REPLICATE_CLIP_MODEL")

            # Google Cloud secrets
            self.cloudsql_private_ip = env.get("CLOUDSQL_PRIVATE_IP")
            self.cloudsql_connection_name = env.get("CLOUDSQL_CONNECTION_NAME")
            self.cloudsql_user = env.get("CLOUDSQL_USER")
            self.cloudsql_password = env.get("CLOUDSQL_PASSWORD")
            self.cloudsql_db = env.get("CLOUDSQL_DB")

            # Google Cloud Run secrets
            self.cloudrun_cpu_service_url = env.get("CLOUDRUN_CPU_SERVICE_URL")
            self.cloudrun_cpu_invoker_sa = env.get("CLOUDRUN_CPU_INVOKER_SA")
            self.cloudrun_gpu_service_url = env.get("CLOUDRUN_GPU_SERVICE_URL")
            self.cloudrun_gpu_invoker_sa = env.get("CLOUDRUN_GPU_INVOKER_SA")

            # Sendgrid secrets
            self.sendgrid_api_key = env.get("SENDGRID_API_KEY")

            # Sentry secrets
            self.sentry_sdk_dsn = env.get("SENTRY_SDK_DSN")

            # Slack secrets
            self.slack_webhook_url = env.get("SLACK_WEBHOOK_URL")

            # App configuration
            self.app_ios_version = env.get("APP_IOS_VERSION")
            self.app_android_version = env.get("APP_ANDROID_VERSION")
            self.app_outage = int(env.get("APP_OUTAGE") or "0")
            self.inf_outage = int(env.get("INF_OUTAGE") or "0")

        # Load general secrets
        self.sentry_traces_sample_rate = float(
            env.get("SENTRY_TRACES_SAMPLE_RATE", "0.1")
        )
        self.cloudtasks_location = GCLOUD_DEFAULT_LOCATION

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get a configuration value."""
        # Map common keys to our loaded secrets
        value = getattr(self, key.lower().replace("_", "_"), None)
        return value if value is not None else default



def debug_database_connection(private_ip: str) -> bool:
    """Test if we can connect to the database.

    Return:
        success - True/False
    """
    try:
        # Test port connectivity before attempting database connection
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)
        result = sock.connect_ex((private_ip, 5432))
        sock.close()

        return result == 0
    except Exception:
        return False
