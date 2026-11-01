"""Settings for the backend, read from the .env file in the project root (Phase 7 Part B).

The database password is never written in code. Copy .env.example to .env and put your own values there.
Real environment variables win over the .env file, so a test or a server can override a value without editing it.

    from backend.app.config import get_settings
    settings = get_settings()
    settings.sqlalchemy_url      # a SQLAlchemy URL object (the password is escaped correctly, so @ # / : are fine)
    settings.describe()          # a printable summary with the password hidden
"""
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import URL, make_url

PROJECT_ROOT = Path(__file__).resolve().parents[2]      # backend/app/config.py -> project root
ENV_FILE = PROJECT_ROOT / ".env"
DRIVER = "postgresql+psycopg2"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(ENV_FILE), env_file_encoding="utf-8", extra="ignore")

    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "icu_risk"
    db_user: str = "postgres"
    db_password: SecretStr = SecretStr("")
    database_url: Optional[str] = None      # optional: a full URL replaces the five DB_* values above

    @property
    def sqlalchemy_url(self) -> URL:
        """The database address as a SQLAlchemy URL. Raises a clear error if no password has been set."""
        if self.database_url:
            return make_url(self.database_url)
        password = self.db_password.get_secret_value()
        if not password:
            raise RuntimeError(
                "DB_PASSWORD is empty. Copy .env.example to .env (Windows: copy .env.example .env) "
                f"and put your PostgreSQL password in it. Expected file: {ENV_FILE}")
        # URL.create escapes special characters in the password for us, so no manual quoting is needed.
        return URL.create(DRIVER, username=self.db_user, password=password,
                          host=self.db_host, port=self.db_port, database=self.db_name)

    def describe(self) -> str:
        """Where the app will connect, with the password hidden (safe to print)."""
        if self.database_url:
            shown = make_url(self.database_url).render_as_string(hide_password=True)
            return f"{shown}  [taken from DATABASE_URL, so the DB_* values are ignored]"
        return f"postgresql://{self.db_user}@{self.db_host}:{self.db_port}/{self.db_name}"


@lru_cache
def get_settings() -> Settings:
    return Settings()
