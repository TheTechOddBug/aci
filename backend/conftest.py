import pytest
from dotenv import load_dotenv
import os

def pytest_configure(config):
    """
    Loads environment variables from .env file before any tests are collected or run.
    This hook is called after command line options have been parsed and before test collection.
    """
    # Assuming pytest is run from the 'backend' directory where this conftest.py and .env reside.
    print(f"pytest_configure: Attempting to load .env file. Current PWD for pytest: {os.getcwd()}")

    # load_dotenv() will search for .env in the current directory or parent directories.
    # override=True ensures that .env values take precedence over any system-set environment variables,
    # which is generally desired for test consistency.
    success = load_dotenv(override=True)

    if success:
        print("pytest_configure: .env file loaded successfully.")
        # You can uncomment the following to debug if a specific variable is loaded:
        # common_aws_region = os.getenv("COMMON_AWS_REGION")
        # print(f"pytest_configure: COMMON_AWS_REGION is '{common_aws_region}'")
    else:
        # This might happen if .env doesn't exist, though we copied it.
        # Or if there are permission issues, though unlikely in this sandbox.
        print("pytest_configure: .env file NOT loaded. This might indicate an issue with .env file presence or accessibility.")

# The previous fixture approach might not run early enough for imports during collection.
# pytest_configure is a more robust hook for this kind of very early setup.
