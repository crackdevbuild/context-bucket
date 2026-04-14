from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from context_bucket.models import (
    ContextBucketPolicy,
    ContextBucketRecord,
    ContextBucketRecordCreate,
    ContextBucketSourceCreate,
    ContextBucketSourceDelete,
    ContextBucketSourceUpsert,
)


async def store_record(service: Any, payload: ContextBucketRecordCreate) -> ContextBucketRecord | None:
    return await create_record(
        service,
        kind=payload.kind,
        text=payload.text,
        scope=payload.scope,
        title=payload.title,
        summary=payload.summary,
        user_id=payload.user_id,
        session_id=payload.session_id,
        source_id=payload.source_id,
        urls=payload.urls,
        tags=payload.tags,
        metadata=payload.metadata,
        structured_data=None,
        data_schema=None,
        content_class="normalized_memory",
        source_type="direct_text",
        source_key=None,
        external_id=None,
        source_updated_at=None,
        source_last_synced_at=None,
        policy=ContextBucketPolicy(),
        skip_duplicate_kinds={"training_example", "decision_outcome"},
    )


async def ingest_source(service: Any, payload: ContextBucketSourceCreate) -> ContextBucketRecord | None:
    return await create_record(
        service,
        kind=payload.kind,
        text=payload.text,
        scope=payload.scope,
        title=payload.title,
        summary=payload.summary,
        user_id=payload.user_id,
        session_id=payload.session_id,
        source_id=payload.source_key,
        urls=payload.urls,
        tags=payload.tags,
        metadata=payload.metadata,
        structured_data=payload.structured_data,
        data_schema=payload.data_schema,
        content_class=payload.content_class,
        source_type=payload.source_type,
        source_key=payload.source_key,
        external_id=payload.external_id,
        source_updated_at=payload.source_updated_at,
        source_last_synced_at=payload.source_last_synced_at,
        policy=payload.policy,
        skip_duplicate_kinds={"training_example", "decision_outcome"},
    )


async def upsert_source(service: Any, payload: ContextBucketSourceUpsert) -> ContextBucketRecord:
    previous = service._find_latest_source(payload.source_key, payload.scope, payload.user_id, payload.session_id)
    content = service._prepare_content(payload.text, payload.structured_data, payload.data_schema)
    checksum = service._content_checksum(content["text"])
    if previous is not None and previous.content_checksum == checksum and previous.source_status == "active":
        return previous
    source_version = (previous.source_version + 1) if previous else 1
    metadata = dict(payload.metadata)
    if previous is not None:
        metadata.setdefault("previous_record_id", previous.id)
    record = await create_record(
        service,
        kind=payload.kind,
        text=payload.text,
        scope=payload.scope,
        title=payload.title,
        summary=payload.summary,
        user_id=payload.user_id,
        session_id=payload.session_id,
        source_id=payload.source_key,
        urls=payload.urls,
        tags=payload.tags,
        metadata=metadata,
        structured_data=payload.structured_data,
        data_schema=payload.data_schema,
        content_class=payload.content_class,
        source_type=payload.source_type,
        source_key=payload.source_key,
        external_id=payload.external_id,
        source_updated_at=payload.source_updated_at,
        source_last_synced_at=payload.source_last_synced_at,
        policy=payload.policy,
        source_version=source_version,
        allow_duplicate=True,
    )
    if previous is not None:
        await service._update_record_lifecycle(previous.id, source_status="superseded")
    assert record is not None
    return record


async def delete_source(service: Any, payload: ContextBucketSourceDelete) -> ContextBucketRecord | None:
    previous = service._find_latest_source(payload.source_key, payload.scope, payload.user_id, payload.session_id)
    if previous is None or previous.source_status == "deleted":
        return None
    deleted_at = payload.deleted_at or datetime.now(timezone.utc)
    await service._update_record_lifecycle(previous.id, source_status="deleted", deleted_at=deleted_at)
    return await service.get_record(previous.id)


async def create_record(
    service: Any,
    *,
    kind: str,
    text: str | None,
    scope: str,
    title: str | None,
    summary: str | None,
    user_id: str | None,
    session_id: str | None,
    source_id: str | None,
    urls: list[str],
    tags: list[str],
    metadata: dict[str, Any],
    structured_data: Any | None,
    data_schema: Any | None,
    content_class: str,
    source_type: str,
    source_key: str | None,
    external_id: str | None,
    source_updated_at: datetime | None,
    source_last_synced_at: datetime | None,
    policy: ContextBucketPolicy,
    skip_duplicate_kinds: set[str] | None = None,
    source_version: int = 1,
    allow_duplicate: bool = False,
) -> ContextBucketRecord | None:
    content = service._prepare_content(text, structured_data, data_schema)
    cleaned = content["text"]
    if not cleaned:
        return None
    lexical_tokens = content["lexical_tokens"]
    semantic_terms = content["semantic_terms"]
    embedding = service._embed(semantic_terms)
    checksum = service._content_checksum(cleaned)
    if not allow_duplicate and kind not in (skip_duplicate_kinds or set()) and service._is_near_duplicate(
        kind=kind,
        scope=scope,
        lexical_tokens=lexical_tokens,
        session_id=session_id,
        user_id=user_id,
    ):
        return None
    now = datetime.now(timezone.utc)
    record_id = f"cb_{uuid4().hex}"
    record = ContextBucketRecord(
        id=record_id,
        kind=kind,  # type: ignore[arg-type]
        scope=scope,  # type: ignore[arg-type]
        owner_scope=scope,  # type: ignore[arg-type]
        source_type=source_type,  # type: ignore[arg-type]
        content_class=content_class,  # type: ignore[arg-type]
        title=(title or "").strip() or None,
        text=cleaned,
        summary=(summary or "").strip() or None,
        user_id=(user_id or "").strip() or None,
        session_id=(session_id or "").strip() or None,
        source_id=(source_id or "").strip() or None,
        source_key=(source_key or "").strip() or None,
        source_version=max(1, int(source_version)),
        source_status="active",
        external_id=(external_id or "").strip() or None,
        content_checksum=checksum,
        source_updated_at=source_updated_at,
        source_last_synced_at=source_last_synced_at or now,
        deleted_at=None,
        urls=service._dedupe_strings(urls),
        tags=service._dedupe_strings(tags),
        metadata=dict(metadata or {}),
        structured_data=content["structured_data"],
        data_schema=content["data_schema"],
        structured_fields=content["structured_fields"],
        policy=policy,
        token_count_estimate=service.token_estimate(cleaned),
        lexical_tokens=lexical_tokens,
        semantic_terms=semantic_terms,
        embedding=embedding,
        chunks=service._chunk_text(record_id, cleaned),
        created_at=now,
    )
    async with service._write_lock:
        service.training_root.mkdir(parents=True, exist_ok=True)
        service._persist_record_locked(record)
        if service.settings.training_export_enabled:
            service._append_training_line(record)
        pruned_records, _ = service._prune_records_locked(now)
        service._rebuild_indexes_locked(service._load_records())
        service._write_state_locked(
            {
                "last_success_at": now.isoformat(),
                "last_error_at": None,
                "last_error": None,
                "last_pruned_at": now.isoformat() if pruned_records else None,
                "pruned_records_total_delta": pruned_records,
            }
        )
    return record
