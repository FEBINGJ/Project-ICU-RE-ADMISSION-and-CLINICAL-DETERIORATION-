"""Database connection and sessions (Phase 7 Part B).

    from backend.app.db.session import get_db
    @router.get("/stays")
    def list_stays(db: Session = Depends(get_db)): ...

The engine is created on first use (not at import), so importing the code never needs a database or a .env file.
"""
from functools import lru_cache
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from backend.app.config import get_settings


@lru_cache
def get_engine() -> Engine:
    # pool_pre_ping drops a connection that PostgreSQL closed while the server was idle, instead of failing a request.
    return create_engine(get_settings().sqlalchemy_url, pool_pre_ping=True)


@lru_cache
def get_sessionmaker() -> sessionmaker:
    return sessionmaker(bind=get_engine(), autoflush=False, expire_on_commit=False)


def get_db() -> Iterator[Session]:
    """FastAPI dependency: one session per request, always closed afterwards."""
    db = get_sessionmaker()()
    try:
        yield db
    finally:
        db.close()
