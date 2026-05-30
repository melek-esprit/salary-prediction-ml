"""Inspect the DatawarehouseDB schema: tables, columns, row counts, keys.

Run with:  python -m src.db.inspect_schema
"""
from __future__ import annotations

import logging

import pandas as pd
from sqlalchemy import text

from src.db.connection import get_engine

logger = logging.getLogger(__name__)


def list_tables() -> pd.DataFrame:
    """Return all base tables with their schema and row counts."""
    query = text(
        """
        SELECT s.name AS [schema],
               t.name AS [table],
               p.rows AS [row_count]
        FROM sys.tables t
        JOIN sys.schemas s ON t.schema_id = s.schema_id
        JOIN sys.partitions p ON t.object_id = p.object_id AND p.index_id IN (0, 1)
        ORDER BY p.rows DESC;
        """
    )
    with get_engine().connect() as conn:
        return pd.read_sql(query, conn)


def list_columns() -> pd.DataFrame:
    """Return every column of every table with data types."""
    query = text(
        """
        SELECT TABLE_SCHEMA AS [schema],
               TABLE_NAME   AS [table],
               COLUMN_NAME  AS [column],
               DATA_TYPE    AS [type],
               CHARACTER_MAXIMUM_LENGTH AS [max_len],
               IS_NULLABLE  AS [nullable]
        FROM INFORMATION_SCHEMA.COLUMNS
        ORDER BY TABLE_NAME, ORDINAL_POSITION;
        """
    )
    with get_engine().connect() as conn:
        return pd.read_sql(query, conn)


def list_foreign_keys() -> pd.DataFrame:
    """Return foreign-key relationships (the star-schema joins)."""
    query = text(
        """
        SELECT fk.name AS fk_name,
               tp.name AS parent_table,
               cp.name AS parent_column,
               tr.name AS referenced_table,
               cr.name AS referenced_column
        FROM sys.foreign_keys fk
        JOIN sys.foreign_key_columns fkc ON fk.object_id = fkc.constraint_object_id
        JOIN sys.tables tp ON fkc.parent_object_id = tp.object_id
        JOIN sys.columns cp ON fkc.parent_object_id = cp.object_id
             AND fkc.parent_column_id = cp.column_id
        JOIN sys.tables tr ON fkc.referenced_object_id = tr.object_id
        JOIN sys.columns cr ON fkc.referenced_object_id = cr.object_id
             AND fkc.referenced_column_id = cr.column_id
        ORDER BY tp.name;
        """
    )
    with get_engine().connect() as conn:
        return pd.read_sql(query, conn)


def main() -> None:
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.width", 200)

    print("\n" + "=" * 70)
    print("TABLES (by row count)")
    print("=" * 70)
    tables = list_tables()
    print(tables.to_string(index=False))

    print("\n" + "=" * 70)
    print("COLUMNS")
    print("=" * 70)
    cols = list_columns()
    print(cols.to_string(index=False))

    print("\n" + "=" * 70)
    print("FOREIGN KEYS (joins)")
    print("=" * 70)
    fks = list_foreign_keys()
    if fks.empty:
        print("No foreign keys defined.")
    else:
        print(fks.to_string(index=False))


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")
    main()
