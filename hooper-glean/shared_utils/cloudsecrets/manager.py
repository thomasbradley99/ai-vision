from typing import Optional, List

from google.cloud import secretmanager
from google.api_core import exceptions

from ..logger import get_logger
from .secrets import SecretConfig

logger = get_logger(__name__)


class CloudSecretsManager:
    """Managers secrets using Google Secret Manager."""

    def __init__(self, project_id: str):
        self.project_id = project_id
        self.client = secretmanager.SecretManagerServiceClient()
        self.parent = f"projects/{project_id}"

    def secret_exists(self, secret_name: str) -> bool:
        """Check if a secret exists"""
        try:
            secret_path = f"{self.parent}/secrets/{secret_name}"
            self.client.get_secret(name=secret_path)
            return True
        except exceptions.NotFound:
            return False
        except Exception as e:
            logger.error(f"Error checking if secret {secret_name} exists: {e}")
            return False

    def create_secret(self, secret_config: SecretConfig) -> str:
        """Create a new secret (without adding a version)."""
        try:
            secret = secretmanager.Secret()
            # Set labels
            secret.labels = {"managed-by": "deployment-script", "required": "true"}
            # Add description if provided
            if secret_config.description:
                if not hasattr(secret, "annotations") or secret.annotations is None:
                    secret.annotations = {}
                secret.annotations["description"] = secret_config.description
            # Set replication policy
            secret.replication = secretmanager.Replication()
            secret.replication.automatic = secretmanager.Replication.Automatic()
            # Create the secret
            response = self.client.create_secret(
                parent=self.parent, secret_id=secret_config.name, secret=secret
            )
            logger.info(f"Created secret: {secret_config.name}")
            return response.name
        except Exception as e:
            logger.error(f"Failed to create secret {secret_config.name}: {e}")
            raise

    def add_secret_version(self, secret_name: str, secret_value: str) -> str:
        """Add a new version to an existing secret"""
        try:
            secret_path = f"{self.parent}/secrets/{secret_name}"
            # Create payload
            payload = secretmanager.SecretPayload()
            payload.data = secret_value.encode("UTF-8")
            # Add the secret version
            response = self.client.add_secret_version(
                parent=secret_path,
                payload=payload,
            )
            logger.info(f"Added new version to secret: {secret_name}")
            return response.name
        except Exception as e:
            logger.error(f"Failed to add version to secret {secret_name}: {e}")
            raise

    def create_or_update_secret(
        self, secret_config: SecretConfig, secret_value: str
    ) -> str:
        """Create secret if it doesn't exist, or update if it does"""
        try:
            if not self.secret_exists(secret_config.name):
                # Create new secret
                self.create_secret(secret_config)
            # Add/update secret version
            version_name = self.add_secret_version(secret_config.name, secret_value)
            return version_name
        except Exception as e:
            logger.error(f"Failed to create or update secret {secret_config.name}: {e}")
            raise

    def get_secret_value(
        self, secret_name: str, version: str = "latest"
    ) -> Optional[str]:
        """Get the current value of a secret (for verification)"""
        try:
            version_path = f"{self.parent}/secrets/{secret_name}/versions/{version}"
            response = self.client.access_secret_version(name=version_path)
            return response.payload.data.decode("UTF-8")
        except Exception as e:
            logger.warning(f"Could not retrieve secret {secret_name}: {e}")
            return None

    def list_secrets(self) -> List[str]:
        """List all secrets in the project"""
        try:
            secrets = self.client.list_secrets(parent=self.parent)
            return [secret.name.split("/")[-1] for secret in secrets]
        except Exception as e:
            logger.error(f"Failed to list secrets: {e}")
            return []
