"""Check the database connection and the tables (Phase 7 Part B).

Run from the project root:
    python -m backend.app.db.check

It prints where it connects (password hidden), the PostgreSQL version, the migration version and every table with its
row count. If something is wrong it says what to try. Exit code 0 means everything is fine, 1 means a problem.
"""
import sys

from sqlalchemy import inspect, text
from sqlalchemy.exc import OperationalError

from backend.app.config import get_settings
from backend.app.db.base import Base
from backend.app.db import models  # noqa: F401  (registers the tables on Base.metadata)
from backend.app.db.session import get_engine


def _hint(message: str) -> str:
    low = message.lower()
    if "password authentication failed" in low:
        return "The password or the user name is wrong. Check DB_USER and DB_PASSWORD in your .env file."
    if "does not exist" in low and "database" in low:
        return "The database has not been created yet. Create it (see docs/database.md), or fix DB_NAME in .env."
    if "refused" in low or "could not connect" in low or "timeout" in low:
        return ("PostgreSQL is not reachable at that host and port. Is the service running? "
                "Is DB_PORT right (native PostgreSQL: 5432, the Docker file: 5433)?")
    return "See the message above."


def main() -> int:
    settings = get_settings()
    print(f"Database settings: {settings.describe()}  (password hidden)")
    try:
        engine = get_engine()
        with engine.connect() as conn:
            version = conn.execute(text("select version()")).scalar()
    except RuntimeError as exc:                        # no password in .env
        print(f"\nPROBLEM: {exc}")
        return 1
    except UnicodeDecodeError:
        # On Windows with a non-English locale, PostgreSQL's error text cannot be decoded by the driver.
        print("\nPROBLEM: could not connect. PostgreSQL answered with an error message the driver cannot read "
              "(a Windows language issue). The usual causes are a wrong password or a database that does not exist. "
              "Check DB_USER, DB_PASSWORD and DB_NAME in .env.")
        return 1
    except OperationalError as exc:
        lines = list(dict.fromkeys(str(exc.orig).strip().splitlines()))      # the driver repeats the line once per address tried
        print(f"\nPROBLEM: could not connect.\n" + "\n".join(lines) + f"\nTry: {_hint(str(exc.orig))}")
        return 1

    print(f"Connected. {version.split(',')[0]}")
    expected = sorted(Base.metadata.tables)
    present = set(inspect(engine).get_table_names())
    found = [t for t in expected if t in present]
    with engine.connect() as conn:
        revision = conn.execute(text("select version_num from alembic_version")).scalar() if "alembic_version" in present else None
        print(f"Migration version: {revision or 'none (run: alembic upgrade head)'}")
        print(f"Tables: {len(found)} of {len(expected)} found")
        for name in found:
            rows = conn.execute(text(f'select count(*) from "{name}"')).scalar()
            print(f"  {name:<22}{rows:>8} rows")
    missing = [t for t in expected if t not in present]
    if missing:
        print(f"\nMissing tables: {', '.join(missing)}\nRun: alembic upgrade head")
        return 1
    print("\nAll good.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
