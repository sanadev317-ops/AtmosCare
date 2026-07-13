import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def _get_env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _load_env_file() -> None:
    env_paths = []
    current = os.path.dirname(__file__)
    for _ in range(6):
        candidate = os.path.join(current, ".env")
        if candidate not in env_paths:
            env_paths.append(candidate)
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent

    for env_path in reversed(env_paths):
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = value


_load_env_file()


# Shared workspace-wide database settings
DATABASE_URI = _get_env(
    "DATABASE_URI",
    _get_env("MONGO_URI", _get_env("MONGODB_URI", "mongodb://localhost:27017"))
)
DATABASE_NAME = _get_env(
    "DATABASE_NAME",
    _get_env("MONGO_DB", _get_env("MONGODB_DATABASE", "AtmosCareDB"))
)

# Backward-compatible aliases
MONGO_URI = DATABASE_URI
DB_NAME = DATABASE_NAME

SMTP_FROM = _get_env("SMTP_FROM", "")
SMTP_USERNAME = _get_env("SMTP_USERNAME", "")
SMTP_PASSWORD = _get_env("SMTP_PASSWORD", "")
