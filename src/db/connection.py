"""Database connection utilities for the DatawarehouseDB SQL Server instance.

Builds a SQLAlchemy engine from environment variables (.env). Supports both
Windows Authentication (trusted connection) and SQL login.
"""
from __future__ import annotations

import logging
import os
import urllib.parse
from functools import lru_cache

from dotenv import load_dotenv
from sqlalchemy import Engine, create_engine, text

logger = logging.getLogger(__name__)

load_dotenv()


def _build_odbc_connection_string() -> str:
    """Build a raw ODBC connection string from environment variables."""
    server = os.getenv("DB_SERVER", "")
    database = os.getenv("DB_NAME", "")
    driver = os.getenv("DB_DRIVER", "ODBC Driver 17 for SQL Server")
    trusted = os.getenv("DB_TRUSTED_CONNECTION", "yes").lower() in {"yes", "true", "1"}
    encrypt = os.getenv("DB_ENCRYPT", "yes")
    trust_cert = os.getenv("DB_TRUST_SERVER_CERTIFICATE", "yes")

    if not server or not database:
        raise ValueError("DB_SERVER and DB_NAME must be set in the .env file.")

    parts = [
        f"DRIVER={{{driver}}}",
        f"SERVER={server}",
        f"DATABASE={database}",
        f"Encrypt={encrypt}",
        f"TrustServerCertificate={trust_cert}",
    ]

    if trusted:
        parts.append("Trusted_Connection=yes")
    else:
        username = os.getenv("DB_USERNAME", "")
        password = os.getenv("DB_PASSWORD", "")
        parts.append(f"UID={username}")
        parts.append(f"PWD={password}")

    return ";".join(parts) + ";"


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Return a cached SQLAlchemy engine for DatawarehouseDB."""
    odbc_str = _build_odbc_connection_string()
    url = "mssql+pyodbc:///?odbc_connect=" + urllib.parse.quote_plus(odbc_str)
    engine = create_engine(url, fast_executemany=True, pool_pre_ping=True)
    logger.info("Created SQLAlchemy engine for %s", os.getenv("DB_NAME"))
    return engine


def test_connection() -> bool:
    """Quick connectivity check. Returns True on success."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            version = conn.execute(text("SELECT @@VERSION")).scalar()
            db = conn.execute(text("SELECT DB_NAME()")).scalar()
        logger.info("Connected to database: %s", db)
        print(f"[OK] Connected to '{db}'.")
        print(f"     {str(version).splitlines()[0]}")
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("Connection failed: %s", exc)
        print(f"[FAIL] Could not connect: {exc}")
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    test_connection()
