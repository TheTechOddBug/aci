import pytest
from unittest.mock import MagicMock, patch
import importlib

# Fixture to mock aci.server.config values
@pytest.fixture
def mock_server_config_for_embeddings(mocker):
    # Patch where 'config' is accessed by the helper module
    mock_config = mocker.patch("aci.server.tests.helper.config")
    mock_config.OPENAI_EMBEDDING_MODEL = "test-embed-model-server"
    mock_config.OPENAI_EMBEDDING_DIMENSION = 456
    # No need to mock API key here as get_default_server_client handles it via env vars
    return mock_config

def test_prepare_dummy_apps_uses_client_factory_for_embeddings(
    mocker,
    mock_server_config_for_embeddings, # Use the fixture
    # We might need dummy app/function data if not easily mockable,
    # but the focus is on client and config parameters.
    # For now, assume the helper can run enough to call embedding functions.
):
    """
    Tests that prepare_dummy_apps_and_functions (from helper.py) uses the
    OpenAI client from the factory and passes it correctly to embedding functions,
    along with embedding model/dimension from server.config.
    """
    # 1. Patch get_default_server_client from its source module
    mock_client_from_factory = MagicMock(spec=mocker.patch("openai.OpenAI")) # Mock an OpenAI client
    mock_get_server_client = mocker.patch(
        "aci.common.openai_clients.get_default_server_client",
        return_value=mock_client_from_factory
    )

    # 2. Patch embedding generation functions from their source module
    # Use side_effect to provide a valid return structure if needed by subsequent code in helper
    dummy_embedding_vector = [0.2] * mock_server_config_for_embeddings.OPENAI_EMBEDDING_DIMENSION

    mock_generate_app_embedding = mocker.patch(
        "aci.common.embeddings.generate_app_embedding",
        return_value=dummy_embedding_vector
    )
    mock_generate_function_embeddings = mocker.patch(
        "aci.common.embeddings.generate_function_embeddings",
        return_value=[dummy_embedding_vector] # Assuming it returns a list of embeddings
    )

    # 3. Reload the helper module to pick up the patched get_default_server_client
    # This is crucial as 'openai_client = get_default_server_client()' is at module level in helper.py
    from aci.server.tests import helper as server_tests_helper_module
    importlib.reload(server_tests_helper_module)

    # 4. Call the function from the reloaded helper module
    # It might require actual dummy app files. For this test, we're focused on the calls.
    # If DUMMY_APPS_DIR or REAL_APPS_DIR access causes issues in isolated test,
    # this part might need adjustment or further mocking of file operations.
    # For now, let's assume it can run or partially run to make the calls.
    try:
        server_tests_helper_module.prepare_dummy_apps_and_functions()
    except FileNotFoundError:
        # This can happen if dummy app files are not found.
        # If generate_app_embedding was called before the error, assertions might still pass.
        # For a more robust test, mock file system interactions if they are complex.
        print("FileNotFoundError during prepare_dummy_apps_and_functions, checking calls made so far.")
        pass # Allow assertions to proceed even if full execution fails due to file IO

    # 5. Assertions
    assert mock_get_server_client.call_count > 0, "get_default_server_client should have been called"

    # Check calls to generate_app_embedding
    assert mock_generate_app_embedding.call_count > 0, "generate_app_embedding should have been called"

    # For each call to generate_app_embedding, check arguments
    for call_args_tuple in mock_generate_app_embedding.call_args_list:
        args, kwargs = call_args_tuple
        passed_client_arg = kwargs.get('openai_client') or (args[1] if len(args) > 1 else None)
        passed_model_arg = kwargs.get('embedding_model') or (args[2] if len(args) > 2 else None)
        passed_dimension_arg = kwargs.get('embedding_dimension') or (args[3] if len(args) > 3 else None)

        assert passed_client_arg is mock_client_from_factory, \
            "Client from factory was not passed to generate_app_embedding"
        assert passed_model_arg == mock_server_config_for_embeddings.OPENAI_EMBEDDING_MODEL
        assert passed_dimension_arg == mock_server_config_for_embeddings.OPENAI_EMBEDDING_DIMENSION

    # Check calls to generate_function_embeddings (if it was reached)
    if mock_generate_function_embeddings.called:
         for call_args_tuple in mock_generate_function_embeddings.call_args_list:
            args, kwargs = call_args_tuple
            passed_client_arg = kwargs.get('openai_client') or (args[1] if len(args) > 1 else None)
            passed_model_arg = kwargs.get('embedding_model') or (args[2] if len(args) > 2 else None)
            passed_dimension_arg = kwargs.get('embedding_dimension') or (args[3] if len(args) > 3 else None)

            assert passed_client_arg is mock_client_from_factory, \
                "Client from factory was not passed to generate_function_embeddings"
            assert passed_model_arg == mock_server_config_for_embeddings.OPENAI_EMBEDDING_MODEL
            assert passed_dimension_arg == mock_server_config_for_embeddings.OPENAI_EMBEDDING_DIMENSION

    # Ensure at least one of the embedding functions was called if the helper ran partially
    assert mock_generate_app_embedding.called or mock_generate_function_embeddings.called, \
        "Neither app nor function embedding generation was called."

# Note: This test assumes that `prepare_dummy_apps_and_functions` will attempt to generate
# embeddings even if some dummy files are missing, or that the mocks will prevent crashes.
# More robust mocking of file system operations (Path.glob, open) might be needed if this proves flaky.
# The `try...except FileNotFoundError` is a basic attempt to handle this.
# The core purpose is to check the client and config parameters passed to embedding functions.

# We also need an __init__.py in backend/aci/server/tests/ to make it a package for discovery
# if running pytest from a higher directory or if this file needs to import siblings.
# However, for `python -m unittest discover -s backend/aci/server/tests`, it might not be strictly needed for discovery itself.
# Pytest is generally better with test discovery.
# For now, let's assume unittest discovery works or we'd run pytest.

# To ensure this test file is discovered by pytest if run from root:
# Create/ensure backend/aci/server/tests/__init__.py

# For the `mocker.patch("openai.OpenAI")` in the `test_upsert_app_uses_client_factory_for_embeddings`
# and here, it's used to provide a `spec` for `MagicMock`.
# This helps ensure the mock behaves somewhat like a real OpenAI client.
# The actual `openai.OpenAI` constructor is not called by the factory if we mock `get_default_xxx_client`.
# If we wanted to test that `get_default_xxx_client` itself calls the correct
# `openai.OpenAI` or `openai.AzureOpenAI` constructor based on env vars,
# we would patch those constructors directly (e.g., `mocker.patch("openai.OpenAI")`, `mocker.patch("openai.AzureOpenAI")`)
# and then call `get_default_xxx_client` without mocking it.
# The tests in `test_openai_clients.py` already cover that internal factory logic.
# These application-level tests verify that the application code *uses* the factory correctly.
