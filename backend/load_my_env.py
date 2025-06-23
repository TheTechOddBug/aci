import os
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), ".env.local")
print(f"load_my_env.py: Attempting to load .env file from: {env_path}", flush=True)
if os.path.exists(env_path):
    loaded = load_dotenv(dotenv_path=env_path, override=True)
    if loaded:
        print(f"load_my_env.py: .env file loaded successfully from {env_path}.", flush=True)
        # Optionally print some key env vars to confirm they are set
        # print(f"load_my_env.py: COMMON_AWS_REGION={os.getenv('COMMON_AWS_REGION')}", flush=True)
        # print(f"load_my_env.py: SERVER_DB_HOST={os.getenv('SERVER_DB_HOST')}", flush=True)
    else:
        print(f"load_my_env.py: Failed to load .env file from {env_path}.", flush=True)
else:
    print(f"load_my_env.py: .env file not found at {env_path}", flush=True)
