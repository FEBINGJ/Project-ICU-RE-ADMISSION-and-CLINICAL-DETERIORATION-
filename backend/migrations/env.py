"""Alembic environment (Phase 7 Part B): connects with the settings from .env and compares against our models."""
from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, pool

from backend.app.config import get_settings
from backend.app.db import models  # noqa: F401  (registers every table on Base.metadata)
from backend.app.db.base import Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Write the SQL instead of running it:  alembic upgrade head --sql"""
    url = get_settings().sqlalchemy_url.render_as_string(hide_password=False)
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True, compare_type=True,
                      dialect_opts={"paramstyle": "named"})
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = create_engine(get_settings().sqlalchemy_url, poolclass=pool.NullPool)
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
