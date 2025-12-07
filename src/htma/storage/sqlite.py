"""SQLite storage implementation for HTMA.

Provides async SQLite operations with migration support and transaction management.
"""

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import aiosqlite

from htma.core.exceptions import (
    DatabaseConnectionError,
    DatabaseError,
    MigrationError,
)

logger = logging.getLogger(__name__)


class SQLiteStorage:
    """Async SQLite storage with migration support.

    This class provides the core database operations for HTMA, including:
    - Connection management
    - Migration execution
    - CRUD operations
    - Transaction support

    Attributes:
        db_path: Path to the SQLite database file.
        _conn: Active database connection (when connected).
    """

    def __init__(self, db_path: str | Path):
        """Initialize SQLite storage.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = Path(db_path)
        self._conn: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Initialize the database by running migrations.

        Creates the database file and parent directories if they don't exist,
        then runs all pending migrations.

        Raises:
            DatabaseError: If database initialization fails.
            MigrationError: If migrations fail to apply.
        """
        try:
            # Ensure parent directory exists
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

            # Enable foreign key constraints
            async with aiosqlite.connect(self.db_path) as conn:
                await conn.execute("PRAGMA foreign_keys = ON")
                await conn.commit()

            # Run migrations
            await self._run_migrations()

            logger.info(f"Database initialized at {self.db_path}")

        except Exception as e:
            raise DatabaseError(f"Failed to initialize database: {e}") from e

    async def _run_migrations(self) -> None:
        """Run all pending database migrations.

        Migrations are SQL files in the migrations/ directory, executed in order.
        Each migration is idempotent and should use IF NOT EXISTS clauses.

        Raises:
            MigrationError: If migration execution fails.
        """
        migrations_dir = Path(__file__).parent / "migrations"

        if not migrations_dir.exists():
            raise MigrationError(f"Migrations directory not found: {migrations_dir}")

        # Get all migration files, sorted by name
        migration_files = sorted(migrations_dir.glob("*.sql"))

        if not migration_files:
            logger.warning("No migration files found")
            return

        try:
            async with aiosqlite.connect(self.db_path) as conn:
                await conn.execute("PRAGMA foreign_keys = ON")

                for migration_file in migration_files:
                    logger.info(f"Running migration: {migration_file.name}")

                    # Read migration SQL
                    sql = migration_file.read_text()

                    # Execute migration
                    await conn.executescript(sql)
                    await conn.commit()

                    logger.info(f"Migration {migration_file.name} completed")

        except Exception as e:
            raise MigrationError(f"Migration failed: {e}") from e

    async def connect(self) -> None:
        """Establish connection to the database.

        Raises:
            DatabaseConnectionError: If connection fails.
        """
        if self._conn is not None:
            logger.warning("Already connected to database")
            return

        try:
            self._conn = await aiosqlite.connect(self.db_path)
            self._conn.row_factory = aiosqlite.Row
            await self._conn.execute("PRAGMA foreign_keys = ON")
            logger.debug("Connected to database")
        except Exception as e:
            raise DatabaseConnectionError(f"Failed to connect to database: {e}") from e

    async def disconnect(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            await self._conn.close()
            self._conn = None
            logger.debug("Disconnected from database")

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[aiosqlite.Connection]:
        """Context manager for database connections.

        Provides a connection that is automatically closed when done.
        If a connection is already active, reuses it.

        Yields:
            Active database connection.

        Example:
            async with storage.connection() as conn:
                await conn.execute("SELECT * FROM entities")
        """
        if self._conn is not None:
            # Reuse existing connection
            yield self._conn
        else:
            # Create temporary connection
            async with aiosqlite.connect(self.db_path) as conn:
                conn.row_factory = aiosqlite.Row
                await conn.execute("PRAGMA foreign_keys = ON")
                yield conn

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[aiosqlite.Connection]:
        """Context manager for database transactions.

        Automatically commits on success or rolls back on error.

        Yields:
            Active database connection within a transaction.

        Example:
            async with storage.transaction() as conn:
                await conn.execute("INSERT INTO entities ...")
                await conn.execute("INSERT INTO facts ...")
                # Automatically committed if no exceptions
        """
        async with self.connection() as conn:
            try:
                yield conn
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise

    async def execute(self, query: str, params: tuple[Any, ...] = ()) -> None:
        """Execute a write query (INSERT, UPDATE, DELETE).

        Args:
            query: SQL query to execute.
            params: Query parameters.

        Raises:
            DatabaseError: If query execution fails.
        """
        try:
            async with self.connection() as conn:
                await conn.execute(query, params)
                await conn.commit()
        except Exception as e:
            raise DatabaseError(f"Query execution failed: {e}") from e

    async def execute_many(
        self, query: str, params_list: list[tuple[Any, ...]]
    ) -> None:
        """Execute a write query multiple times with different parameters.

        Args:
            query: SQL query to execute.
            params_list: List of parameter tuples.

        Raises:
            DatabaseError: If query execution fails.
        """
        try:
            async with self.connection() as conn:
                await conn.executemany(query, params_list)
                await conn.commit()
        except Exception as e:
            raise DatabaseError(f"Batch query execution failed: {e}") from e

    async def fetch_one(
        self, query: str, params: tuple[Any, ...] = ()
    ) -> dict[str, Any] | None:
        """Fetch a single row from the database.

        Args:
            query: SQL query to execute.
            params: Query parameters.

        Returns:
            Dictionary representation of the row, or None if no rows found.

        Raises:
            DatabaseError: If query execution fails.
        """
        try:
            async with self.connection() as conn:
                cursor = await conn.execute(query, params)
                row = await cursor.fetchone()

                if row is None:
                    return None

                return dict(row)

        except Exception as e:
            raise DatabaseError(f"Query failed: {e}") from e

    async def fetch_all(
        self, query: str, params: tuple[Any, ...] = ()
    ) -> list[dict[str, Any]]:
        """Fetch all rows from the database.

        Args:
            query: SQL query to execute.
            params: Query parameters.

        Returns:
            List of dictionaries, one per row.

        Raises:
            DatabaseError: If query execution fails.
        """
        try:
            async with self.connection() as conn:
                cursor = await conn.execute(query, params)
                rows = await cursor.fetchall()

                return [dict(row) for row in rows]

        except Exception as e:
            raise DatabaseError(f"Query failed: {e}") from e

    async def count(self, table: str, where: str = "", params: tuple[Any, ...] = ()) -> int:
        """Count rows in a table.

        Args:
            table: Table name.
            where: Optional WHERE clause (without 'WHERE' keyword).
            params: Query parameters for WHERE clause.

        Returns:
            Number of rows.

        Raises:
            DatabaseError: If query execution fails.
        """
        query = f"SELECT COUNT(*) as count FROM {table}"
        if where:
            query += f" WHERE {where}"

        result = await self.fetch_one(query, params)
        return result["count"] if result else 0

    async def table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database.

        Args:
            table_name: Name of the table to check.

        Returns:
            True if table exists, False otherwise.
        """
        query = """
            SELECT name FROM sqlite_master
            WHERE type='table' AND name=?
        """
        result = await self.fetch_one(query, (table_name,))
        return result is not None

    def _serialize_json_field(self, value: Any) -> str | None:
        """Serialize a Python object to JSON string for storage.

        Args:
            value: Value to serialize (dict, list, etc.).

        Returns:
            JSON string, or None if value is None.
        """
        if value is None:
            return None
        return json.dumps(value)

    def _deserialize_json_field(self, value: str | None) -> Any:
        """Deserialize a JSON string to Python object.

        Args:
            value: JSON string from database.

        Returns:
            Deserialized Python object, or None if value is None.
        """
        if value is None:
            return None
        return json.loads(value)
