import openai
import os
from typing import Optional, Union, Dict, Tuple

# Custom Exceptions for more specific error handling
class OpenAIClientError(Exception):
    """Base exception for client factory errors."""
    pass

class MissingConfigurationError(OpenAIClientError):
    """Raised when a required configuration is missing."""
    pass

class UnsupportedClientTypeError(OpenAIClientError):
    """Raised when an unsupported client type is requested."""
    pass


class OpenAIClientFactory:
    """
    A singleton factory for creating and managing OpenAI client instances.
    It supports standard OpenAI, Azure OpenAI, and other OpenAI API-compatible providers
    through environment variables and direct parameter overrides.
    """
    _instance = None
    _clients: Dict[Tuple, Union[openai.OpenAI, openai.AzureOpenAI]] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(OpenAIClientFactory, cls).__new__(cls)
            cls._clients = {}
        return cls._instance

    def get_client(
        self,
        api_key: str,
        client_type_override: Optional[str] = None,
        base_url_override: Optional[str] = None,
        azure_endpoint_override: Optional[str] = None,
        api_version_override: Optional[str] = None,
        org_id_override: Optional[str] = None,
        project_id_override: Optional[str] = None,
    ) -> Union[openai.OpenAI, openai.AzureOpenAI]:
        """
        Retrieves or creates an OpenAI client based on the provided configuration.

        Args:
            api_key: The API key for the client.
            client_type_override: Override for OPENAI_CLIENT_TYPE env var.
            base_url_override: Override for OPENAI_BASE_URL env var.
            azure_endpoint_override: Override for AZURE_OPENAI_ENDPOINT env var.
            api_version_override: Override for OPENAI_API_VERSION env var (for Azure).
            org_id_override: Override for OPENAI_ORG_ID env var.
            project_id_override: Override for OPENAI_PROJECT_ID env var.

        Returns:
            An instance of openai.OpenAI or openai.AzureOpenAI.

        Raises:
            MissingConfigurationError: If required configuration for a client type is missing.
            UnsupportedClientTypeError: If the client type is not supported.
        """
        if not api_key:
            raise MissingConfigurationError("API key is required to get a client.")

        client_type = client_type_override or os.getenv("OPENAI_CLIENT_TYPE", "openai").lower()

        # Determine effective values from overrides or environment variables
        base_url = base_url_override or os.getenv("OPENAI_BASE_URL")
        azure_endpoint = azure_endpoint_override or os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = api_version_override or os.getenv("OPENAI_API_VERSION")
        org_id = org_id_override or os.getenv("OPENAI_ORG_ID")
        project_id = project_id_override or os.getenv("OPENAI_PROJECT_ID")

        # Create a unique key for caching based on all relevant parameters
        cache_key_parts = [api_key, client_type]
        if client_type == "azure":
            cache_key_parts.extend([azure_endpoint, api_version])
        else: # openai or compatible
            cache_key_parts.extend([base_url, org_id, project_id])

        cache_key = tuple(cache_key_parts)

        if cache_key in self._clients:
            return self._clients[cache_key]

        client: Union[openai.OpenAI, openai.AzureOpenAI]

        if client_type == "azure":
            if not azure_endpoint:
                raise MissingConfigurationError(
                    "Azure OpenAI client requires AZURE_OPENAI_ENDPOINT to be set."
                )
            if not api_version:
                # Fallback to a common default if not set, though it's better to require it
                api_version = "2024-02-01"
                # Or raise MissingConfigurationError("Azure OpenAI client requires OPENAI_API_VERSION to be set.")

            client = openai.AzureOpenAI(
                api_key=api_key,
                azure_endpoint=azure_endpoint,
                api_version=api_version,
            )
        elif client_type == "openai":
            client_params = {"api_key": api_key}
            if base_url:
                client_params["base_url"] = base_url
            if org_id:
                client_params["organization"] = org_id
            if project_id:
                client_params["project"] = project_id

            client = openai.OpenAI(**client_params)
        else:
            raise UnsupportedClientTypeError(
                f"Unsupported client type: {client_type}. Must be 'openai' or 'azure'."
            )

        self._clients[cache_key] = client
        return client

# Singleton instance of the factory
client_factory = OpenAIClientFactory()

# Convenience functions
def get_default_server_client() -> Union[openai.OpenAI, openai.AzureOpenAI]:
    """
    Gets an OpenAI client configured with SERVER_OPENAI_API_KEY.
    Reads OPENAI_CLIENT_TYPE, AZURE_OPENAI_ENDPOINT, etc., from environment variables.
    """
    api_key = os.getenv("SERVER_OPENAI_API_KEY")
    if not api_key:
        raise MissingConfigurationError("SERVER_OPENAI_API_KEY environment variable not set.")
    return client_factory.get_client(api_key=api_key)

def get_default_cli_client() -> Union[openai.OpenAI, openai.AzureOpenAI]:
    """
    Gets an OpenAI client configured with CLI_OPENAI_API_KEY.
    Reads OPENAI_CLIENT_TYPE, AZURE_OPENAI_ENDPOINT, etc., from environment variables.
    """
    api_key = os.getenv("CLI_OPENAI_API_KEY")
    if not api_key:
        # Fallback to server key if CLI key is not specifically set
        api_key = os.getenv("SERVER_OPENAI_API_KEY")
    if not api_key:
        raise MissingConfigurationError("CLI_OPENAI_API_KEY or SERVER_OPENAI_API_KEY environment variable not set.")
    return client_factory.get_client(api_key=api_key)

def get_openai_client_from_env(
    identifier: str = "default"
) -> Union[openai.OpenAI, openai.AzureOpenAI]:
    """
    Generic function to fetch an OpenAI client based on environment variables
    prefixed or identified by the `identifier`.

    Example environment variables for identifier='MY_SERVICE':
    MY_SERVICE_OPENAI_API_KEY
    MY_SERVICE_OPENAI_CLIENT_TYPE (optional, defaults to 'openai')
    MY_SERVICE_AZURE_OPENAI_ENDPOINT (if client_type is 'azure')
    MY_SERVICE_OPENAI_API_VERSION (if client_type is 'azure')
    MY_SERVICE_OPENAI_BASE_URL (if client_type is 'openai' and using compatible API)
    MY_SERVICE_OPENAI_ORG_ID (optional)
    MY_SERVICE_OPENAI_PROJECT_ID (optional)
    """
    prefix = identifier.upper() + "_" if identifier != "default" else ""

    api_key_env_var = f"{prefix}OPENAI_API_KEY"
    # For default, also check SERVER_OPENAI_API_KEY as a common convention
    api_key = os.getenv(api_key_env_var) or (os.getenv("SERVER_OPENAI_API_KEY") if identifier == "default" else None)


    if not api_key:
        raise MissingConfigurationError(
            f"API key environment variable ({api_key_env_var} or SERVER_OPENAI_API_KEY for default) not set."
        )

    client_type = os.getenv(f"{prefix}OPENAI_CLIENT_TYPE", os.getenv("OPENAI_CLIENT_TYPE", "openai")).lower()
    base_url = os.getenv(f"{prefix}OPENAI_BASE_URL", os.getenv("OPENAI_BASE_URL"))
    azure_endpoint = os.getenv(f"{prefix}AZURE_OPENAI_ENDPOINT", os.getenv("AZURE_OPENAI_ENDPOINT"))
    api_version = os.getenv(f"{prefix}OPENAI_API_VERSION", os.getenv("OPENAI_API_VERSION"))
    org_id = os.getenv(f"{prefix}OPENAI_ORG_ID", os.getenv("OPENAI_ORG_ID"))
    project_id = os.getenv(f"{prefix}OPENAI_PROJECT_ID", os.getenv("OPENAI_PROJECT_ID"))

    return client_factory.get_client(
        api_key=api_key,
        client_type_override=client_type,
        base_url_override=base_url,
        azure_endpoint_override=azure_endpoint,
        api_version_override=api_version,
        org_id_override=org_id,
        project_id_override=project_id,
    )

# Make main factory methods directly available if preferred, e.g.
# get_client = client_factory.get_client
# This allows `from aci.common.openai_clients import get_client`
# However, it might be cleaner to always use `client_factory.get_client` or the convenience functions.
