from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from context_bucket.models import ContextBucketRecord


def training_line(record: ContextBucketRecord) -> str:
    return json.dumps(
        {
            "id": record.id,
            "kind": record.kind,
            "scope": record.scope,
            "source_type": record.source_type,
            "content_class": record.content_class,
            "user_id": record.user_id,
            "session_id": record.session_id,
            "source_id": record.source_id,
            "source_key": record.source_key,
            "source_version": record.source_version,
            "source_status": record.source_status,
            "external_id": record.external_id,
            "content_checksum": record.content_checksum,
            "source_updated_at": record.source_updated_at.isoformat() if record.source_updated_at else None,
            "source_last_synced_at": record.source_last_synced_at.isoformat() if record.source_last_synced_at else None,
            "deleted_at": record.deleted_at.isoformat() if record.deleted_at else None,
            "title": record.title,
            "summary": record.summary,
            "text": record.text,
            "token_count_estimate": record.token_count_estimate,
            "lexical_tokens": record.lexical_tokens,
            "semantic_terms": record.semantic_terms,
            "embedding": record.embedding,
            "urls": record.urls,
            "tags": record.tags,
            "metadata": record.metadata,
            "structured_data": record.structured_data,
            "data_schema": record.data_schema.model_dump(mode="json") if record.data_schema else None,
            "structured_fields": [field.model_dump(mode="json") for field in record.structured_fields],
            "policy": record.policy.model_dump(mode="json"),
            "created_at": record.created_at.isoformat(),
        },
        ensure_ascii=True,
    )


def append_training_line(service: Any, record: ContextBucketRecord) -> None:
    training_path = service.training_root / f"{record.created_at.date().isoformat()}.jsonl"
    with training_path.open("a", encoding="utf-8") as handle:
        handle.write(training_line(record) + "\n")


async def export_training(service: Any) -> tuple[int, Path]:
    service.records_root.mkdir(parents=True, exist_ok=True)
    service.training_root.mkdir(parents=True, exist_ok=True)
    records = service._load_records()
    output_path = service.training_root / "export.jsonl"
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(training_line(record) + "\n")
    return len(records), output_path
