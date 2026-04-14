from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from context_bucket.models import ContextBucketRetrieveRequest, ContextBucketRetrieveResponse


def applied_filters(payload: ContextBucketRetrieveRequest) -> dict[str, list[str]]:
    filters: dict[str, list[str]] = {}
    filters["model_target"] = [payload.model_target]
    filters["enforce_policy"] = [str(bool(payload.enforce_policy)).lower()]
    if payload.kinds:
        filters["kinds"] = list(payload.kinds)
    if payload.source_types:
        filters["source_types"] = list(payload.source_types)
    if payload.content_classes:
        filters["content_classes"] = list(payload.content_classes)
    if payload.tags:
        filters["tags"] = list(payload.tags)
    if payload.source_keys:
        filters["source_keys"] = list(payload.source_keys)
    if payload.confidentialities:
        filters["confidentialities"] = list(payload.confidentialities)
    if payload.source_statuses:
        filters["source_statuses"] = list(payload.source_statuses)
    return filters


def write_audit_entry(
    service: Any,
    payload: ContextBucketRetrieveRequest,
    response: ContextBucketRetrieveResponse,
    visible_records: list[dict[str, Any]],
) -> str:
    audit_id = f"audit_{uuid4().hex}"
    service.audit_root.mkdir(parents=True, exist_ok=True)
    entry = {
        "audit_id": audit_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "query_text": payload.query_text,
        "user_id": payload.user_id,
        "session_id": payload.session_id,
        "model_target": payload.model_target,
        "enforce_policy": payload.enforce_policy,
        "applied_filters": response.applied_filters,
        "candidate_counts": response.candidate_counts,
        "excluded_counts": response.excluded_counts,
        "visible_record_ids": [record["id"] for record in visible_records],
        "selected_record_ids": [item.record_id for item in response.items],
        "selected_chunk_ids": [item.chunk_id for item in response.items],
        "retrieval_strategy": response.retrieval_strategy,
    }
    with service.audit_log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=True) + "\n")
    return audit_id
