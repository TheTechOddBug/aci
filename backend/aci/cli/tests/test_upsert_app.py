import json
from pathlib import Path

import pytest
from click.testing import CliRunner
from sqlalchemy.orm import Session

from aci.cli.commands.upsert_app import upsert_app
from aci.common.db import crud
from aci.common.db.sql_models import SecurityScheme


@pytest.mark.parametrize("skip_dry_run", [True, False])
def test_create_app(
    db_session: Session,
    dummy_app_data: dict,
    dummy_app_file: Path,
    dummy_app_secrets_data: dict,
    dummy_app_secrets_file: Path,
    skip_dry_run: bool,
) -> None:
    runner = CliRunner()
    command = [
        "--app-file",
        dummy_app_file,
        "--secrets-file",
        dummy_app_secrets_file,
    ]
    if skip_dry_run:
        command.append("--skip-dry-run")

    result = runner.invoke(upsert_app, command)  # type: ignore
    assert result.exit_code == 0, result.output
    # new record is created by a different db session, so we need to
    # expire the injected db_session to see the new record
    db_session.expire_all()
    app = crud.apps.get_app(
        db_session, dummy_app_data["name"], public_only=False, active_only=False
    )

    if skip_dry_run:
        assert app is not None
        assert app.name == dummy_app_data["name"]
        assert (
            app.security_schemes[SecurityScheme.OAUTH2]["client_id"]
            == dummy_app_secrets_data["AIPOLABS_GOOGLE_APP_CLIENT_ID"]
        )
        assert (
            app.security_schemes[SecurityScheme.OAUTH2]["client_secret"]
            == dummy_app_secrets_data["AIPOLABS_GOOGLE_APP_CLIENT_SECRET"]
        )
    else:
        assert app is None, "App should not be created for dry run"


# To mock values from aci.cli.config that are used by upsert_app for embeddings
@pytest.fixture
def mock_cli_config_for_embeddings(mocker):
    mock_config = mocker.patch("aci.cli.commands.upsert_app.config")
    mock_config.OPENAI_EMBEDDING_MODEL = "test-embed-model-cli"
    mock_config.OPENAI_EMBEDDING_DIMENSION = 123
    # OPENAI_API_KEY is not directly used by upsert_app anymore for client instantiation
    # but get_default_cli_client will read CLI_OPENAI_API_KEY from env.
    # We don't need to mock config.OPENAI_API_KEY for the factory part.
    return mock_config

def test_upsert_app_uses_client_factory_for_embeddings(
    db_session: Session, # db_session fixture is available from conftest.py
    dummy_app_file: Path,
    dummy_app_secrets_file: Path,
    mocker,
    mock_cli_config_for_embeddings, # Use the fixture
):
    """
    Tests that upsert_app command uses the OpenAI client from the factory
    and passes it correctly to the embedding generation function, along with
    embedding model and dimension from cli.config.
    """
    # 1. Patch get_default_cli_client from its source module
    mock_client_from_factory = mocker.MagicMock(spec=mocker.patch("openai.OpenAI")) # Mock an OpenAI client
    mock_get_cli_client = mocker.patch(
        "aci.common.openai_clients.get_default_cli_client", # Target where factory client is obtained
        return_value=mock_client_from_factory
    )

    # 2. Patch generate_app_embedding from its source module
    mock_generate_embedding = mocker.patch(
        "aci.common.embeddings.generate_app_embedding",
        return_value=[0.1] * mock_cli_config_for_embeddings.OPENAI_EMBEDDING_DIMENSION # Return dummy embedding
    )

    # Reload the module where the client is instantiated at module level to pick up the patch
    # This is important because `openai_client = get_default_cli_client()` is at the top level
    # of `backend.aci.cli.commands.upsert_app`
    import importlib
    from aci.cli.commands import upsert_app as upsert_app_module
    importlib.reload(upsert_app_module)


    # 3. Run the upsert_app command (for creating a new app)
    runner = CliRunner()
    command = [
        "--app-file", str(dummy_app_file),
        "--secrets-file", str(dummy_app_secrets_file),
        "--skip-dry-run" # Ensure it actually tries to create and generate embeddings
    ]

    # Use the reloaded module's command
    result = runner.invoke(upsert_app_module.upsert_app, command)
    assert result.exit_code == 0, result.output

    # 4. Assertions
    mock_get_cli_client.assert_called_once() # Factory function was called

    mock_generate_embedding.assert_called_once()
    args, kwargs = mock_generate_embedding.call_args

    # Assert the client from the factory was passed to generate_app_embedding
    passed_client_arg = kwargs.get('openai_client') # generate_app_embedding takes it as kwarg or positional
    if passed_client_arg is None and args: # check if passed positionally
        passed_client_arg = args[1] # app_config, openai_client, model, dimension

    assert passed_client_arg is mock_client_from_factory, \
        "The OpenAI client from the factory was not passed to generate_app_embedding"

    # Assert model and dimension from (mocked) config were passed
    passed_model_arg = kwargs.get('embedding_model') or (args[2] if len(args) > 2 else None)
    passed_dimension_arg = kwargs.get('embedding_dimension') or (args[3] if len(args) > 3 else None)

    assert passed_model_arg == mock_cli_config_for_embeddings.OPENAI_EMBEDDING_MODEL
    assert passed_dimension_arg == mock_cli_config_for_embeddings.OPENAI_EMBEDDING_DIMENSION


@pytest.mark.parametrize("skip_dry_run", [True, False])
def test_update_app(
    db_session: Session,
    dummy_app_data: dict,
    dummy_app_file: Path,
    dummy_app_secrets_data: dict,
    dummy_app_secrets_file: Path,
    skip_dry_run: bool,
) -> None:
    # create the app first
    test_create_app(
        db_session,
        dummy_app_data,
        dummy_app_file,
        dummy_app_secrets_data,
        dummy_app_secrets_file,
        True,
    )

    # modify the app data
    new_oauth2_scope = "updated_scope"
    new_oauth2_client_id = "updated_client_id"
    new_api_key = {"location": "header", "name": "X-API-KEY"}

    dummy_app_data["security_schemes"]["oauth2"]["scope"] = new_oauth2_scope
    dummy_app_secrets_data["AIPOLABS_GOOGLE_APP_CLIENT_ID"] = new_oauth2_client_id
    dummy_app_data["security_schemes"]["api_key"] = new_api_key

    # write the modified app data and secrets to the files
    dummy_app_file.write_text(json.dumps(dummy_app_data))
    dummy_app_secrets_file.write_text(json.dumps(dummy_app_secrets_data))

    # update the app
    runner = CliRunner()
    command = [
        "--app-file",
        dummy_app_file,
        "--secrets-file",
        dummy_app_secrets_file,
    ]
    if skip_dry_run:
        command.append("--skip-dry-run")

    result = runner.invoke(upsert_app, command)  # type: ignore
    assert result.exit_code == 0, result.output

    db_session.expire_all()
    app = crud.apps.get_app(
        db_session, dummy_app_data["name"], public_only=False, active_only=False
    )
    assert app is not None
    assert app.name == dummy_app_data["name"]

    if skip_dry_run:
        assert app.security_schemes[SecurityScheme.OAUTH2]["scope"] == new_oauth2_scope
        assert app.security_schemes[SecurityScheme.OAUTH2]["client_id"] == new_oauth2_client_id
        assert app.security_schemes[SecurityScheme.API_KEY] == new_api_key
    else:
        # nothing should change for dry run
        assert (
            app.security_schemes[SecurityScheme.OAUTH2]["scope"]
            == "openid email profile https://www.googleapis.com/auth/calendar"
        )
        assert app.security_schemes[SecurityScheme.OAUTH2]["client_id"] == "dummy_client_id"
        assert SecurityScheme.API_KEY not in app.security_schemes
