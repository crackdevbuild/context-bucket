from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import Any

from context_bucket.models import ContextBucketRecord


def load_records(service: Any) -> list[ContextBucketRecord]:
    if service.settings.record_backend == "sqlite":
        return load_records_sqlite(service)
    if not service.records_root.exists():
        return []
    records: list[ContextBucketRecord] = []
    for path in sorted(service.records_root.glob("*.json")):
        try:
            records.append(ContextBucketRecord.model_validate_json(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    return records


def find_latest_source(
    service: Any,
    source_key: str,
    scope: str,
    user_id: str | None,
    session_id: str | None,
) -> ContextBucketRecord | None:
    matches = [
        record for record in service._load_record_index()
        if record.get("source_key") == source_key and record["scope"] == scope
    ]
    if scope == "session":
        matches = [record for record in matches if record.get("session_id") == session_id]
    elif scope == "user":
        matches = [record for record in matches if record.get("user_id") == user_id]
    matches.sort(key=lambda item: (int(item.get("source_version") or 1), item["created_at"]), reverse=True)
    if not matches:
        return None
    return service.get_record_sync(str(matches[0]["id"]))


def rebuild_indexes_locked(service: Any, records: list[ContextBucketRecord]) -> None:
    service.index_root.mkdir(parents=True, exist_ok=True)
    record_rows = [service._record_summary(record) for record in records]
    if service.settings.index_backend == "sqlite":
        rebuild_sqlite_indexes_locked(service, records, record_rows)
        return
    service.record_index_path.write_text(
        json.dumps([serialize_index_row(row) for row in record_rows], indent=2),
        encoding="utf-8",
    )
    with service.chunk_index_path.open("w", encoding="utf-8") as handle:
        for record in records:
            for chunk in record.chunks:
                handle.write(json.dumps(serialize_index_row(service._chunk_summary(record, chunk))) + "\n")


def load_record_index(service: Any) -> list[dict[str, Any]]:
    if service.settings.index_backend == "sqlite":
        return load_record_index_sqlite(service)
    if not service.record_index_path.exists():
        bootstrap_indexes(service)
    if not service.record_index_path.exists():
        return []
    try:
        payload = json.loads(service.record_index_path.read_text(encoding="utf-8"))
        return [deserialize_index_row(item) for item in payload]
    except Exception:
        bootstrap_indexes(service)
        if not service.record_index_path.exists():
            return []
        payload = json.loads(service.record_index_path.read_text(encoding="utf-8"))
        return [deserialize_index_row(item) for item in payload]


def load_chunk_index(service: Any) -> list[dict[str, Any]]:
    if service.settings.index_backend == "sqlite":
        return load_chunk_index_sqlite(service)
    if not service.chunk_index_path.exists():
        bootstrap_indexes(service)
    if not service.chunk_index_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with service.chunk_index_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                cleaned = line.strip()
                if not cleaned:
                    continue
                rows.append(deserialize_index_row(json.loads(cleaned)))
        return rows
    except Exception:
        bootstrap_indexes(service)
        if not service.chunk_index_path.exists():
            return []
        with service.chunk_index_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                cleaned = line.strip()
                if not cleaned:
                    continue
                rows.append(deserialize_index_row(json.loads(cleaned)))
        return rows


def bootstrap_indexes(service: Any) -> None:
    rebuild_indexes_locked(service, service._load_records())


def rebuild_sqlite_indexes_locked(
    service: Any,
    records: list[ContextBucketRecord],
    record_rows: list[dict[str, Any]],
) -> None:
    connection = sqlite_connection(service)
    try:
        cursor = connection.cursor()
        cursor.executescript(
            """
            CREATE TABLE IF NOT EXISTS record_index (
                record_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunk_index (
                chunk_id TEXT PRIMARY KEY,
                record_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                payload_json TEXT NOT NULL
            );
            DELETE FROM record_index;
            DELETE FROM chunk_index;
            """
        )
        cursor.executemany(
            "INSERT INTO record_index(record_id, created_at, payload_json) VALUES (?, ?, ?)",
            [
                (
                    str(row["id"]),
                    row["created_at"].isoformat() if isinstance(row.get("created_at"), datetime) else str(row.get("created_at")),
                    json.dumps(serialize_index_row(row)),
                )
                for row in record_rows
            ],
        )
        chunk_rows = [
            service._chunk_summary(record, chunk)
            for record in records
            for chunk in record.chunks
        ]
        cursor.executemany(
            "INSERT INTO chunk_index(chunk_id, record_id, chunk_index, payload_json) VALUES (?, ?, ?, ?)",
            [
                (
                    str(row["chunk_id"]),
                    str(row["record_id"]),
                    int(row["chunk_index"]),
                    json.dumps(serialize_index_row(row)),
                )
                for row in chunk_rows
            ],
        )
        connection.commit()
    finally:
        connection.close()


def load_record_index_sqlite(service: Any) -> list[dict[str, Any]]:
    if not service.sqlite_index_path.exists():
        bootstrap_indexes(service)
    if not service.sqlite_index_path.exists():
        return []
    connection = sqlite_connection(service)
    try:
        rows = connection.execute("SELECT payload_json FROM record_index ORDER BY created_at DESC").fetchall()
        return [deserialize_index_row(json.loads(row[0])) for row in rows]
    except sqlite3.OperationalError:
        bootstrap_indexes(service)
        if not service.sqlite_index_path.exists():
            return []
        rows = connection.execute("SELECT payload_json FROM record_index ORDER BY created_at DESC").fetchall()
        return [deserialize_index_row(json.loads(row[0])) for row in rows]
    finally:
        connection.close()


def load_chunk_index_sqlite(service: Any) -> list[dict[str, Any]]:
    if not service.sqlite_index_path.exists():
        bootstrap_indexes(service)
    if not service.sqlite_index_path.exists():
        return []
    connection = sqlite_connection(service)
    try:
        rows = connection.execute("SELECT payload_json FROM chunk_index ORDER BY record_id, chunk_index").fetchall()
        return [deserialize_index_row(json.loads(row[0])) for row in rows]
    except sqlite3.OperationalError:
        bootstrap_indexes(service)
        if not service.sqlite_index_path.exists():
            return []
        rows = connection.execute("SELECT payload_json FROM chunk_index ORDER BY record_id, chunk_index").fetchall()
        return [deserialize_index_row(json.loads(row[0])) for row in rows]
    finally:
        connection.close()


def sqlite_connection(service: Any) -> sqlite3.Connection:
    service.index_root.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(service.sqlite_index_path)


def sqlite_records_connection(service: Any) -> sqlite3.Connection:
    service.root.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(service.sqlite_records_path)


def get_record_sync(service: Any, record_id: str) -> ContextBucketRecord | None:
    if service.settings.record_backend == "sqlite":
        connection = sqlite_records_connection(service)
        try:
            ensure_sqlite_record_tables(connection)
            row = connection.execute("SELECT payload_json FROM records WHERE record_id = ?", (record_id,)).fetchone()
            if row is None:
                return None
            return ContextBucketRecord.model_validate_json(row[0])
        finally:
            connection.close()
    path = service.records_root / f"{record_id}.json"
    if not path.exists():
        return None
    return ContextBucketRecord.model_validate_json(path.read_text(encoding="utf-8"))


def persist_record_locked(service: Any, record: ContextBucketRecord) -> None:
    if service.settings.record_backend == "sqlite":
        connection = sqlite_records_connection(service)
        try:
            ensure_sqlite_record_tables(connection)
            connection.execute(
                "INSERT OR REPLACE INTO records(record_id, created_at, payload_json) VALUES (?, ?, ?)",
                (record.id, record.created_at.isoformat(), record.model_dump_json(indent=2)),
            )
            connection.commit()
            return
        finally:
            connection.close()
    service.records_root.mkdir(parents=True, exist_ok=True)
    (service.records_root / f"{record.id}.json").write_text(record.model_dump_json(indent=2), encoding="utf-8")


def delete_record_locked(service: Any, record_id: str) -> None:
    if service.settings.record_backend == "sqlite":
        connection = sqlite_records_connection(service)
        try:
            ensure_sqlite_record_tables(connection)
            connection.execute("DELETE FROM records WHERE record_id = ?", (record_id,))
            connection.commit()
            return
        finally:
            connection.close()
    (service.records_root / f"{record_id}.json").unlink(missing_ok=True)


def load_records_sqlite(service: Any) -> list[ContextBucketRecord]:
    if not service.sqlite_records_path.exists():
        return []
    connection = sqlite_records_connection(service)
    try:
        ensure_sqlite_record_tables(connection)
        rows = connection.execute("SELECT payload_json FROM records ORDER BY created_at ASC").fetchall()
        return [ContextBucketRecord.model_validate_json(row[0]) for row in rows]
    finally:
        connection.close()


def ensure_sqlite_record_tables(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS records (
            record_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            payload_json TEXT NOT NULL
        )
        """
    )
    connection.commit()


def serialize_index_row(row: dict[str, Any]) -> dict[str, Any]:
    serialized: dict[str, Any] = {}
    for key, value in row.items():
        serialized[key] = value.isoformat() if isinstance(value, datetime) else value
    return serialized


def deserialize_index_row(row: dict[str, Any]) -> dict[str, Any]:
    parsed = dict(row)
    for key in ("created_at", "source_updated_at", "source_last_synced_at", "deleted_at"):
        if parsed.get(key):
            parsed[key] = datetime.fromisoformat(parsed[key])
    return parsed


def read_state_locked(service: Any) -> dict[str, Any]:
    if not service.state_path.exists():
        return {}
    try:
        return json.loads(service.state_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_state_locked(service: Any, patch: dict[str, Any]) -> None:
    current = read_state_locked(service)
    for key, value in patch.items():
        if key.endswith("_delta"):
            base_key = key.removesuffix("_delta")
            current[base_key] = int(current.get(base_key) or 0) + int(value or 0)
        else:
            current[key] = value
    service.root.mkdir(parents=True, exist_ok=True)
    service.state_path.write_text(json.dumps(current, indent=2, sort_keys=True), encoding="utf-8")
