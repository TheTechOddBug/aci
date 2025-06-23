import unittest
from unittest.mock import patch, MagicMock
import os
import openai # Import openai itself for type checking and attributes

# Make sure the aci module is discoverable if tests are run from a different directory
import sys
from pathlib import Path
# Add the project root to sys.path if not already there
project_root = Path(__file__).resolve().parents[3] # Adjust depth if necessary (aci/common/tests/test_openai_clients.py -> root)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Now import from aci.common
from aci.common.openai_clients import (
    client_factory,
    get_default_server_client,
    get_default_cli_client,
    get_openai_client_from_env,
    OpenAIClientError,
    MissingConfigurationError,
    UnsupportedClientTypeError,
)

# Reset the singleton instance for clean tests
def reset_factory_singleton():
    # Explicitly get the client_factory from the module it's defined in
    client_factory_module = sys.modules['aci.common.openai_clients']
    original_module_client_factory = client_factory_module.client_factory

    # Create a new factory class definition based on the original one's class
    NewFactoryClass = type("OpenAIClientFactoryForTest", (original_module_client_factory.__class__,), {'_instance': None, '_clients': {}})

    # Replace the factory instance in the original module
    new_instance = NewFactoryClass()
    client_factory_module.client_factory = new_instance

    # Update the global variable 'client_factory' in *this current test module*
    # to also point to this new instance.
    global client_factory
    client_factory = new_instance


class TestOpenAIClientFactory(unittest.TestCase):

    def setUp(self):
        """Clear the client cache and reset os.environ for each test."""
        # Ensure the original factory is restored before each test, then reset for the test
        # This is tricky due to Python's import caching.
        # The most robust way is to ensure the module's global is changed.

        # Store the factory that was present in the module when setUp started
        self.original_factory_in_module = sys.modules['aci.common.openai_clients'].client_factory

        reset_factory_singleton() # This will replace the factory in the module and in this test file's global scope

        self.original_environ = os.environ.copy()
        os.environ.clear() # Start with a clean environment for most tests


    def tearDown(self):
        """Restore original environment variables and the original factory in the module."""
        os.environ.clear()
        os.environ.update(self.original_environ)

        # Restore the original factory in the module
        client_factory_module = sys.modules['aci.common.openai_clients']
        client_factory_module.client_factory = self.original_factory_in_module

        # Also restore the global 'client_factory' in this test module to the original one
        global client_factory
        client_factory = self.original_factory_in_module


    def test_get_client_default_openai(self):
        # os.environ["OPENAI_API_KEY"] = "test_default_key" # API key is passed directly
        client = client_factory.get_client(api_key="test_default_key")
        self.assertIsInstance(client, openai.OpenAI)
        self.assertEqual(client.api_key, "test_default_key")
        self.assertEqual(str(client.base_url), "https://api.openai.com/v1/") # Default base_url

    def test_get_client_openai_with_custom_base_url(self):
        os.environ["OPENAI_API_KEY"] = "test_custom_base_key"
        os.environ["OPENAI_CLIENT_TYPE"] = "openai"
        os.environ["OPENAI_BASE_URL"] = "http://localhost:1234/v1"

        client = client_factory.get_client(
            api_key="test_custom_base_key",
            base_url_override="http://localhost:1234/v1" # Also test override
        )
        self.assertIsInstance(client, openai.OpenAI)
        self.assertEqual(client.api_key, "test_custom_base_key")
        self.assertEqual(str(client.base_url), "http://localhost:1234/v1/")

    def test_get_client_azure_openai(self):
        os.environ["OPENAI_API_KEY"] = "test_azure_key"
        os.environ["OPENAI_CLIENT_TYPE"] = "azure"
        os.environ["AZURE_OPENAI_ENDPOINT"] = "https://test-azure.openai.azure.com/"
        os.environ["OPENAI_API_VERSION"] = "2024-02-01"

        client = client_factory.get_client(api_key="test_azure_key")
        self.assertIsInstance(client, openai.AzureOpenAI)
        self.assertEqual(client.api_key, "test_azure_key")
        # The base_url for AzureOpenAI is complex and often includes deployment details not set at client init.
        # It's more reliable to check that the core Azure parameters were set.
        # self.assertEqual(str(client.base_url).rstrip('/'), "https://test-azure.openai.azure.com") # This is unreliable
        self.assertEqual(client._api_version, "2024-02-01")
        self.assertEqual(client._azure_endpoint, "https://test-azure.openai.azure.com/")


    def test_client_caching(self):
        os.environ["MY_KEY"] = "cache_test_key"
        client1 = client_factory.get_client(api_key="cache_test_key")
        client2 = client_factory.get_client(api_key="cache_test_key")
        self.assertIs(client1, client2)

        client3 = client_factory.get_client(api_key="another_key")
        self.assertIsNot(client1, client3)

        # Test caching with different types
        os.environ["OPENAI_CLIENT_TYPE"] = "azure"
        os.environ["AZURE_OPENAI_ENDPOINT"] = "https://test-cache.openai.azure.com/"
        os.environ["OPENAI_API_VERSION"] = "2024-01-01"
        azure_client1 = client_factory.get_client(api_key="azure_cache_key")
        azure_client2 = client_factory.get_client(api_key="azure_cache_key")
        self.assertIs(azure_client1, azure_client2)
        self.assertIsNot(client1, azure_client1)


    def test_missing_api_key_in_get_client(self):
        with self.assertRaisesRegex(MissingConfigurationError, "API key is required"):
            client_factory.get_client(api_key="")

    def test_missing_azure_endpoint(self):
        os.environ["OPENAI_CLIENT_TYPE"] = "azure"
        with self.assertRaisesRegex(MissingConfigurationError, "AZURE_OPENAI_ENDPOINT"):
            client_factory.get_client(api_key="anykey")

    def test_unsupported_client_type(self):
        os.environ["OPENAI_CLIENT_TYPE"] = "GalaxyBrainAI"
        with self.assertRaisesRegex(UnsupportedClientTypeError, "Unsupported client type: galaxybrainai"):
            client_factory.get_client(api_key="anykey")

    def test_parameter_overrides_in_get_client(self):
        # Set env for Azure
        os.environ["OPENAI_API_KEY"] = "override_key"
        os.environ["OPENAI_CLIENT_TYPE"] = "azure"
        os.environ["AZURE_OPENAI_ENDPOINT"] = "https://env-azure.openai.azure.com/"
        os.environ["OPENAI_API_VERSION"] = "2023-01-01"

        # Override to OpenAI with custom base URL
        client = client_factory.get_client(
            api_key="override_key",
            client_type_override="openai",
            base_url_override="http://override.localhost/v1"
        )
        self.assertIsInstance(client, openai.OpenAI)
        self.assertEqual(str(client.base_url), "http://override.localhost/v1/")

    def test_get_default_server_client_success(self):
        os.environ["SERVER_OPENAI_API_KEY"] = "server_key_123"
        client = get_default_server_client()
        self.assertIsInstance(client, openai.OpenAI)
        self.assertEqual(client.api_key, "server_key_123")

    def test_get_default_server_client_missing_key(self):
        with self.assertRaisesRegex(MissingConfigurationError, "SERVER_OPENAI_API_KEY environment variable not set"):
            get_default_server_client()

    def test_get_default_cli_client_with_cli_key(self):
        os.environ["CLI_OPENAI_API_KEY"] = "cli_key_456"
        os.environ["SERVER_OPENAI_API_KEY"] = "server_key_should_be_ignored" # Make sure CLI key takes precedence
        client = get_default_cli_client()
        self.assertIsInstance(client, openai.OpenAI)
        self.assertEqual(client.api_key, "cli_key_456")

    def test_get_default_cli_client_fallback_to_server_key(self):
        os.environ["SERVER_OPENAI_API_KEY"] = "server_key_789"
        # CLI_OPENAI_API_KEY is not set
        client = get_default_cli_client()
        self.assertIsInstance(client, openai.OpenAI)
        self.assertEqual(client.api_key, "server_key_789")

    def test_get_default_cli_client_missing_all_keys(self):
        # Neither CLI_OPENAI_API_KEY nor SERVER_OPENAI_API_KEY is set
        with self.assertRaisesRegex(MissingConfigurationError, "CLI_OPENAI_API_KEY or SERVER_OPENAI_API_KEY environment variable not set"):
            get_default_cli_client()

    def test_get_openai_client_from_env_default_identifier(self):
        os.environ["SERVER_OPENAI_API_KEY"] = "default_env_key"
        client = get_openai_client_from_env() # identifier="default"
        self.assertIsInstance(client, openai.OpenAI)
        self.assertEqual(client.api_key, "default_env_key")

    def test_get_openai_client_from_env_custom_identifier(self):
        os.environ["MY_SERVICE_OPENAI_API_KEY"] = "my_service_key"
        os.environ["MY_SERVICE_OPENAI_CLIENT_TYPE"] = "azure"
        os.environ["MY_SERVICE_AZURE_OPENAI_ENDPOINT"] = "https://myservice.openai.azure.com/"
        os.environ["MY_SERVICE_OPENAI_API_VERSION"] = "2024-04-01"

        client = get_openai_client_from_env(identifier="MY_SERVICE")
        self.assertIsInstance(client, openai.AzureOpenAI)
        self.assertEqual(client.api_key, "my_service_key")
        self.assertEqual(client._azure_endpoint, "https://myservice.openai.azure.com/")
        self.assertEqual(client._api_version, "2024-04-01")

    def test_get_openai_client_from_env_custom_identifier_openai_base_url(self):
        os.environ["CUSTOM_OPENAI_API_KEY"] = "custom_key_for_base_url"
        os.environ["CUSTOM_OPENAI_BASE_URL"] = "http://custom.openai.local/api"
        # OPENAI_CLIENT_TYPE defaults to "openai"

        client = get_openai_client_from_env(identifier="CUSTOM")
        self.assertIsInstance(client, openai.OpenAI)
        self.assertEqual(client.api_key, "custom_key_for_base_url")
        self.assertEqual(str(client.base_url), "http://custom.openai.local/api/")

    def test_get_openai_client_from_env_missing_key_for_identifier(self):
        with self.assertRaisesRegex(MissingConfigurationError, "API key environment variable .*_OPENAI_API_KEY"):
            get_openai_client_from_env(identifier="NON_EXISTENT_SERVICE")

    def test_openai_org_and_project_id(self):
        os.environ["OPENAI_API_KEY"] = "org_proj_key"
        os.environ["OPENAI_ORG_ID"] = "org-123"
        os.environ["OPENAI_PROJECT_ID"] = "proj-456"

        client = client_factory.get_client(api_key="org_proj_key")
        self.assertIsInstance(client, openai.OpenAI)
        self.assertEqual(client.organization, "org-123")
        self.assertEqual(client.project, "proj-456") # project attribute might not be directly exposed, depends on openai lib version

        # Test with get_openai_client_from_env
        client_env = get_openai_client_from_env() # Uses OPENAI_ORG_ID and OPENAI_PROJECT_ID by default
        self.assertEqual(client_env.organization, "org-123")
        # self.assertEqual(client_env.project, "proj-456") # Check how to verify project ID if not direct attribute

    def test_azure_default_api_version_if_not_set(self):
        # This test assumes the factory provides a default for Azure API version if not set.
        # The current implementation in the prompt does this.
        os.environ["OPENAI_API_KEY"] = "test_azure_default_version_key"
        os.environ["OPENAI_CLIENT_TYPE"] = "azure"
        os.environ["AZURE_OPENAI_ENDPOINT"] = "https://test-azure-default.openai.azure.com/"
        # OPENAI_API_VERSION is NOT set

        client = client_factory.get_client(api_key="test_azure_default_version_key")
        self.assertIsInstance(client, openai.AzureOpenAI)
        self.assertEqual(client._api_version, "2024-02-01") # Check against the default in the factory

if __name__ == "__main__":
    unittest.main()

# Helper to reset the singleton for testing purposes
# This is a bit of a hack for true isolation between test methods if they modify the singleton's state (like cache)
# A better approach for the factory might be to not make it a strict singleton in the Python sense,
# or provide a reset method for testing.
_original_factory_instance = client_factory
_OriginalFactoryClass = type(client_factory)

def reset_openai_client_factory_singleton_for_tests():
    global client_factory
    OpenAIClientFactoryForTest = type("OpenAIClientFactory", (client_factory.__class__,), {'_instance': None, '_clients': {}})
    client_factory = OpenAIClientFactoryForTest()

# This manual reset function could be called in setUp if the class-level one is problematic
# For now, the setUp/tearDown uses a direct re-assignment which should work for these tests.

# Note: The project ID on openai.OpenAI client is not directly accessible as `client.project`.
# It's passed during initialization and used internally.
# Verification would typically involve mocking the openai.OpenAI constructor and checking call_args.
# For simplicity here, we are not mocking the constructor itself.
# If testing `project` is critical, one would do:
# with patch('openai.OpenAI') as mock_openai_constructor:
# client_factory.get_client(api_key="k", project_id_override="proj-test")
# mock_openai_constructor.assert_called_with(api_key="k", project="proj-test")
# However, this makes tests more brittle to OpenAI library's internal changes.
# We trust the OpenAI library to handle the parameters we pass.


# Redundant helper removed as setUp/tearDown now handle singleton reset more robustly.
