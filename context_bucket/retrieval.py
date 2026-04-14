from __future__ import annotations

import math
import re
from collections import Counter
from datetime import datetime, timezone
from datetime import timedelta
from typing import Any, Callable

from context_bucket.audit import applied_filters, write_audit_entry
from context_bucket.models import (
    ContextBucketContextItem,
    ContextBucketRecord,
    ContextBucketRetrieveRequest,
    ContextBucketRetrieveResponse,
)


TOKEN_RE = re.compile(r"[a-z0-9]{4,}", re.IGNORECASE)
SEMANTIC_EXPANSIONS: dict[str, set[str]] = {
    "brief": {"concise", "short"},
    "concise": {"brief", "short"},
    "short": {"brief", "concise"},
    "update": {"status", "summary"},
    "status": {"update", "summary"},
    "summary": {"update", "status"},
    "email": {"mail", "draft"},
    "mail": {"email", "draft"},
    "draft": {"email", "mail"},
    "client": {"customer", "matter"},
    "matter": {"case", "client"},
    "case": {"matter"},
    "redline": {"revision", "edit"},
    "revision": {"redline", "edit"},
    "edit": {"revision", "redline"},
}


def lexical_tokens(text: str) -> list[str]:
    return sorted(set(token.lower() for token in TOKEN_RE.findall(text)))


def semantic_terms(text: str) -> list[str]:
    terms: set[str] = set()
    for token in lexical_tokens(text):
        lower = token.lower()
        terms.add(lower)
        stem = stem_token(lower)
        if stem:
            terms.add(stem)
        for synonym in SEMANTIC_EXPANSIONS.get(lower, set()):
            terms.add(synonym)
            synonym_stem = stem_token(synonym)
            if synonym_stem:
                terms.add(synonym_stem)
        for shingle in char_ngrams(lower, 4):
            terms.add(shingle)
    return sorted(terms)


def stem_token(token: str) -> str:
    stem = token.lower()
    for suffix in ("ingly", "edly", "ment", "tion", "ions", "ness", "able", "ible", "ing", "ers", "ies", "ied", "ed", "es", "s"):
        if len(stem) > (len(suffix) + 2) and stem.endswith(suffix):
            if suffix in {"ies", "ied"}:
                return stem[: -len(suffix)] + "y"
            return stem[: -len(suffix)]
    return stem


def char_ngrams(token: str, n: int) -> list[str]:
    padded = f"^{token}$"
    if len(padded) <= n:
        return [padded]
    return [padded[index:index + n] for index in range(len(padded) - n + 1)]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return max(0.0, sum(a * b for a, b in zip(left, right)))


def lexical_overlap_score(query_tokens: list[str], chunk_tokens: list[str]) -> float:
    return set_overlap_score(query_tokens, chunk_tokens)


def set_overlap_score(left: list[str], right: list[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set or not right_set:
        return 0.0
    overlap = len(left_set & right_set)
    if overlap == 0:
        return 0.0
    return overlap / math.sqrt(len(left_set) * len(right_set))


def rerank_score(
    *,
    record: ContextBucketRecord | dict[str, Any],
    semantic_score: float,
    lexical_score: float,
    keyword_bonus: float,
    metadata_bonus: float,
    settings: Any,
    query_max_age_days_fn: Callable[[ContextBucketRecord | dict[str, Any]], int],
) -> float:
    return (
        (semantic_score * float(settings.semantic_score_weight))
        + (lexical_score * float(settings.lexical_score_weight))
        + (keyword_bonus * float(settings.keyword_bonus_weight))
        + metadata_bonus
        + record_rank_bonus(record)
        + scope_priority_bonus(record["scope"] if isinstance(record, dict) else record.scope)
        + age_decay_factor(record, settings=settings, query_max_age_days_fn=query_max_age_days_fn)
    )


def scope_priority_bonus(scope: str) -> float:
    if scope == "session":
        return 0.12
    if scope == "user":
        return 0.08
    return 0.03


def selection_reason(scope: str, semantic_score: float, lexical_score: float, keyword_bonus: float) -> str:
    if semantic_score >= lexical_score and semantic_score > 0:
        mode = "semantic"
    elif lexical_score > 0:
        mode = "lexical"
    else:
        mode = "keyword"
    if keyword_bonus > 0 and mode not in {"semantic", "keyword"}:
        mode = f"{mode}_keyword"
    return f"{scope}_scope_{mode}_match"


def dedupe_context_items(items: list[ContextBucketContextItem]) -> list[ContextBucketContextItem]:
    seen: set[str] = set()
    deduped: list[ContextBucketContextItem] = []
    for item in items:
        key = f"{item.record_id}:{item.chunk_id}"
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def record_rank_bonus(record: ContextBucketRecord | dict[str, Any]) -> float:
    kind = record["kind"] if isinstance(record, dict) else record.kind
    if kind in {"research_report", "research_finding"}:
        return 0.15
    if kind in {"decision_outcome", "topic_pattern", "user_profile_note"}:
        return 0.08
    return 0.0


def age_decay_factor(
    record: ContextBucketRecord | dict[str, Any],
    *,
    settings: Any,
    query_max_age_days_fn: Callable[[ContextBucketRecord | dict[str, Any]], int],
) -> float:
    max_age_days = query_max_age_days_fn(record)
    if max_age_days <= 0:
        return 0.0
    created_at = record["created_at"] if isinstance(record, dict) else record.created_at
    age_days = max(0.0, (datetime.now(timezone.utc) - created_at).total_seconds() / 86400.0)
    start_pct = max(0.1, min(0.9, float(settings.decay_start_pct)))
    start_day = max_age_days * start_pct
    if age_days <= start_day:
        return 0.05
    if age_days >= max_age_days:
        return -0.25
    span = max(0.001, max_age_days - start_day)
    position = (age_days - start_day) / span
    return 0.05 - (0.30 * position)


def compress_context_items(
    items: list[ContextBucketContextItem],
    *,
    lexical_tokens_fn: Callable[[str], list[str]],
) -> list[ContextBucketContextItem]:
    compressed: list[ContextBucketContextItem] = []
    seen_signatures: list[set[str]] = []
    for item in items:
        signature = set(lexical_tokens_fn(item.text))
        duplicate = False
        for existing in seen_signatures:
            if token_set_similarity(signature, existing) >= 0.82:
                duplicate = True
                break
        if duplicate:
            continue
        seen_signatures.append(signature)
        compressed.append(item)
    return compressed


def token_set_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, len(left | right))


def record_visible(
    record: ContextBucketRecord | dict[str, Any],
    *,
    session_id: str | None,
    user_id: str | None,
    include_user_scope: bool,
    include_global_scope: bool,
) -> bool:
    scope = record["scope"] if isinstance(record, dict) else record.scope
    record_session_id = record.get("session_id") if isinstance(record, dict) else record.session_id
    record_user_id = record.get("user_id") if isinstance(record, dict) else record.user_id
    if scope == "session":
        return bool(session_id) and record_session_id == session_id
    if scope == "user":
        return include_user_scope and bool(user_id) and record_user_id == user_id
    return include_global_scope


def record_summary_stale_for_context(
    service: Any,
    record: dict[str, Any],
    max_age_days: int | None,
) -> bool:
    created_at = record["created_at"]
    if max_age_days is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=int(max_age_days))
        return created_at < cutoff
    return record_summary_stale_for_query(service, record)


def record_summary_stale_for_query(service: Any, record: dict[str, Any]) -> bool:
    max_age_days = service._query_max_age_days(record)
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    return record["created_at"] < cutoff


def policy_exclusion_reason(
    record: dict[str, Any],
    payload: ContextBucketRetrieveRequest,
) -> str | None:
    policy = dict(record.get("policy") or {})
    allowed_user_ids = {str(item).strip() for item in policy.get("allowed_user_ids", []) if str(item).strip()}
    allowed_session_ids = {str(item).strip() for item in policy.get("allowed_session_ids", []) if str(item).strip()}
    confidentiality = str(policy.get("confidentiality") or "private")

    if allowed_user_ids and (not payload.user_id or payload.user_id not in allowed_user_ids):
        return "policy_user_not_allowed"
    if allowed_session_ids and (not payload.session_id or payload.session_id not in allowed_session_ids):
        return "policy_session_not_allowed"

    if payload.model_target == "hosted" and not bool(policy.get("allow_remote_model_egress", False)):
        return "policy_remote_egress_denied"
    if payload.model_target == "local" and not bool(policy.get("allow_local_model_egress", True)):
        return "policy_local_egress_denied"

    if confidentiality == "shareable":
        return None
    if confidentiality == "restricted":
        if record["scope"] == "global" and not payload.user_id and not payload.session_id:
            return "policy_restricted_context_required"
        return None
    if confidentiality == "private":
        if record["scope"] == "global" and not (allowed_user_ids or allowed_session_ids):
            return "policy_private_global_denied"
        return None
    return None


def filter_record_summaries_for_retrieval(
    service: Any,
    records: list[dict[str, Any]],
    payload: ContextBucketRetrieveRequest,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    include_user_scope = (
        service.settings.include_user_scope_by_default
        if payload.include_user_scope is None
        else bool(payload.include_user_scope)
    )
    include_global_scope = (
        service.settings.include_global_scope_by_default
        if payload.include_global_scope is None
        else bool(payload.include_global_scope)
    )
    allowed_kinds = set(payload.kinds or [])
    allowed_source_types = set(payload.source_types or [])
    allowed_content_classes = set(payload.content_classes or [])
    allowed_tags = {tag.strip().lower() for tag in (payload.tags or []) if str(tag).strip()}
    allowed_source_keys = {item.strip() for item in (payload.source_keys or []) if str(item).strip()}
    allowed_confidentialities = set(payload.confidentialities or [])
    allowed_source_statuses = set(payload.source_statuses or [])

    filtered: list[dict[str, Any]] = []
    excluded_counts: Counter[str] = Counter()
    for record in records:
        if allowed_kinds and record["kind"] not in allowed_kinds:
            excluded_counts["kind_filter"] += 1
            continue
        if allowed_source_types and record["source_type"] not in allowed_source_types:
            excluded_counts["source_type_filter"] += 1
            continue
        if allowed_content_classes and record["content_class"] not in allowed_content_classes:
            excluded_counts["content_class_filter"] += 1
            continue
        if allowed_source_keys and (record.get("source_key") or "") not in allowed_source_keys:
            excluded_counts["source_key_filter"] += 1
            continue
        policy = record.get("policy") or {}
        if allowed_confidentialities and policy.get("confidentiality") not in allowed_confidentialities:
            excluded_counts["confidentiality_filter"] += 1
            continue
        if allowed_source_statuses:
            if record["source_status"] not in allowed_source_statuses:
                excluded_counts["source_status_filter"] += 1
                continue
        elif record["source_status"] != "active":
            excluded_counts["inactive_source"] += 1
            continue
        if allowed_tags and not (allowed_tags & {str(tag).lower() for tag in record.get("tags", [])}):
            excluded_counts["tag_filter"] += 1
            continue
        if record_summary_stale_for_context(service, record, payload.max_age_days):
            excluded_counts["stale"] += 1
            continue
        if not record_visible(
            record,
            session_id=payload.session_id,
            user_id=payload.user_id,
            include_user_scope=include_user_scope,
            include_global_scope=include_global_scope,
        ):
            excluded_counts["scope_visibility"] += 1
            continue
        if payload.enforce_policy:
            reason = policy_exclusion_reason(record, payload)
            if reason is not None:
                excluded_counts[reason] += 1
                continue
        filtered.append(record)
    return filtered, dict(excluded_counts)


def filter_records_for_retrieval(
    service: Any,
    records: list[ContextBucketRecord],
    payload: ContextBucketRetrieveRequest,
) -> list[ContextBucketRecord]:
    filtered, _ = filter_record_summaries_for_retrieval(
        service,
        [service._record_summary(record) for record in records],
        payload,
    )
    allowed_ids = {record["id"] for record in filtered}
    return [record for record in records if record.id in allowed_ids]


def keyword_bonus_from_index(
    service: Any,
    query_tokens: list[str],
    record: dict[str, Any],
    chunk: dict[str, Any],
) -> float:
    if not query_tokens:
        return 0.0
    bonus = 0.0
    title_tokens = service._lexical_tokens(str(record.get("title") or ""))
    tag_tokens = [str(tag).lower() for tag in record.get("tags", [])]
    field_path_tokens = service._lexical_tokens(" ".join(str(item) for item in record.get("schema_field_paths", [])))
    if set(query_tokens) & set(title_tokens):
        bonus += 0.04
    if set(query_tokens) & set(tag_tokens):
        bonus += 0.03
    if set(query_tokens) & set(field_path_tokens):
        bonus += 0.04
    source_key = str(record.get("source_key") or "")
    if source_key and any(token in source_key.lower() for token in query_tokens):
        bonus += 0.02
    if int(chunk.get("chunk_index") or 0) == 0:
        bonus += 0.01
    return bonus


def metadata_bonus_from_index(
    service: Any,
    payload: ContextBucketRetrieveRequest,
    record: dict[str, Any],
) -> float:
    bonus = 0.0
    if payload.source_keys and record.get("source_key") in set(payload.source_keys):
        bonus += float(service.settings.metadata_bonus_weight)
    if payload.tags and set(tag.lower() for tag in payload.tags) & {str(tag).lower() for tag in record.get("tags", [])}:
        bonus += float(service.settings.metadata_bonus_weight) / 2.0
    return bonus


def provenance_from_index(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "record_id": record["id"],
        "source_key": record.get("source_key"),
        "source_version": record.get("source_version"),
        "source_status": record.get("source_status"),
        "external_id": record.get("external_id"),
        "content_checksum": record.get("content_checksum"),
        "source_type": record["source_type"],
        "content_class": record["content_class"],
        "source_updated_at": record.get("source_updated_at").isoformat() if record.get("source_updated_at") else None,
        "source_last_synced_at": record.get("source_last_synced_at").isoformat() if record.get("source_last_synced_at") else None,
        "deleted_at": record.get("deleted_at").isoformat() if record.get("deleted_at") else None,
        "user_id": record.get("user_id"),
        "session_id": record.get("session_id"),
        "urls": list(record.get("urls", [])),
        "tags": list(record.get("tags", [])),
        "metadata": dict(record.get("metadata", {})),
        "structured_data": record.get("structured_data"),
        "data_schema": dict(record.get("data_schema", {})) if record.get("data_schema") else None,
        "structured_fields": list(record.get("structured_fields", [])),
        "policy": dict(record.get("policy", {})),
    }


def empty_retrieve_response(service: Any, payload: ContextBucketRetrieveRequest) -> ContextBucketRetrieveResponse:
    response = ContextBucketRetrieveResponse(
        query_text=payload.query_text,
        items=[],
        total_candidates=0,
        candidate_counts={
            "visible_records": 0,
            "eligible_chunks": 0,
            "semantic_candidates": 0,
            "lexical_candidates": 0,
            "reranked_candidates": 0,
        },
        excluded_counts={},
        applied_filters=applied_filters(payload),
    )
    response.audit_id = write_audit_entry(service, payload, response, [])
    return response


async def retrieve_context(service: Any, payload: ContextBucketRetrieveRequest) -> ContextBucketRetrieveResponse:
    query_text = payload.query_text.strip()
    if not query_text:
        return empty_retrieve_response(service, payload)
    record_index = service._load_record_index()
    if not record_index:
        return empty_retrieve_response(service, payload)

    query_content = service._prepare_content(query_text, payload.intent_data, payload.intent_schema)
    query_lexical = query_content["lexical_tokens"]
    query_semantic = query_content["semantic_terms"]
    query_embedding = service._embed(query_semantic)
    if not query_lexical and not query_semantic:
        return empty_retrieve_response(service, payload)

    filtered_records, excluded_counts = filter_record_summaries_for_retrieval(service, record_index, payload)
    if not filtered_records:
        response = ContextBucketRetrieveResponse(
            query_text=payload.query_text,
            items=[],
            total_candidates=0,
            candidate_counts={
                "visible_records": 0,
                "eligible_chunks": 0,
                "semantic_candidates": 0,
                "lexical_candidates": 0,
                "reranked_candidates": 0,
            },
            excluded_counts=excluded_counts,
            applied_filters=applied_filters(payload),
        )
        response.audit_id = write_audit_entry(service, payload, response, [])
        return response

    lexical_pool_size = max(
        max(1, int(payload.limit or service.settings.query_top_k)),
        max(1, int(payload.limit or service.settings.query_top_k)) * int(service.settings.lexical_candidate_multiplier),
    )
    semantic_pool_size = max(
        max(1, int(payload.limit or service.settings.query_top_k)),
        max(1, int(payload.limit or service.settings.query_top_k)) * int(service.settings.semantic_candidate_multiplier),
    )

    indexed_chunks = service._load_chunk_index()
    visible_record_ids = {record["id"] for record in filtered_records}
    visible_records = {record["id"]: record for record in filtered_records}

    chunk_rows: list[dict[str, Any]] = []
    for chunk in indexed_chunks:
        record_id = str(chunk.get("record_id") or "")
        if record_id not in visible_record_ids:
            continue
        record = visible_records[record_id]
        chunk_embedding = list(chunk.get("embedding") or [])
        if not chunk_embedding:
            chunk_embedding = service._embed(list(chunk.get("semantic_terms") or chunk.get("lexical_tokens") or []))
        semantic_score = service._cosine_similarity(query_embedding, chunk_embedding)
        chunk_tokens = list(chunk.get("lexical_tokens") or [])
        lexical_score = service._lexical_overlap_score(query_lexical, chunk_tokens)
        keyword_bonus = keyword_bonus_from_index(service, query_lexical, record, chunk)
        if semantic_score <= 0 and lexical_score <= 0 and keyword_bonus <= 0:
            continue
        metadata_bonus = metadata_bonus_from_index(service, payload, record)
        chunk_rows.append(
            {
                "record": record,
                "chunk": chunk,
                "semantic_score": semantic_score,
                "lexical_score": lexical_score,
                "keyword_bonus": keyword_bonus,
                "metadata_bonus": metadata_bonus,
            }
        )

    if not chunk_rows:
        response = ContextBucketRetrieveResponse(
            query_text=payload.query_text,
            items=[],
            total_candidates=0,
            candidate_counts={
                "visible_records": len(filtered_records),
                "eligible_chunks": 0,
                "semantic_candidates": 0,
                "lexical_candidates": 0,
                "reranked_candidates": 0,
            },
            excluded_counts=excluded_counts,
            applied_filters=applied_filters(payload),
        )
        response.audit_id = write_audit_entry(service, payload, response, filtered_records)
        return response

    semantic_candidates = sorted(
        chunk_rows,
        key=lambda row: (row["semantic_score"], row["keyword_bonus"], row["record"]["created_at"]),
        reverse=True,
    )[:semantic_pool_size]
    lexical_candidates = sorted(
        chunk_rows,
        key=lambda row: (row["lexical_score"], row["keyword_bonus"], row["record"]["created_at"]),
        reverse=True,
    )[:lexical_pool_size]

    candidate_map: dict[str, dict[str, Any]] = {}
    for row in semantic_candidates + lexical_candidates:
        chunk = row["chunk"]
        candidate_map[f"{chunk['record_id']}:{chunk['chunk_id']}"] = row

    reranked_items: list[ContextBucketContextItem] = []
    for row in candidate_map.values():
        record = row["record"]
        chunk = row["chunk"]
        semantic_score = float(row["semantic_score"])
        lexical_score = float(row["lexical_score"])
        keyword_bonus = float(row["keyword_bonus"])
        metadata_bonus = float(row["metadata_bonus"])
        final_score = service._rerank_score(
            record=record,
            semantic_score=semantic_score,
            lexical_score=lexical_score,
            keyword_bonus=keyword_bonus,
            metadata_bonus=metadata_bonus,
        )
        if final_score <= 0:
            continue
        reranked_items.append(
            ContextBucketContextItem(
                record_id=record["id"],
                chunk_id=chunk["chunk_id"],
                kind=record["kind"],
                scope=record["scope"],
                source_type=record["source_type"],
                content_class=record["content_class"],
                title=record.get("title"),
                text=chunk["text"],
                summary=record.get("summary"),
                score=final_score,
                semantic_score=semantic_score,
                lexical_score=lexical_score,
                provenance=provenance_from_index(record),
                created_at=record["created_at"],
                token_count_estimate=int(chunk.get("token_count_estimate") or 0),
                selection_reason=service._selection_reason(record["scope"], semantic_score, lexical_score, keyword_bonus),
            )
        )

    reranked_items.sort(key=lambda item: (item.score, item.created_at), reverse=True)
    deduped = service._dedupe_context_items(reranked_items)
    limit = max(1, int(payload.limit or service.settings.query_top_k))
    response = ContextBucketRetrieveResponse(
        query_text=payload.query_text,
        items=deduped[:limit],
        total_candidates=len(deduped),
        retrieval_strategy="embedding_lexical_rerank",
        candidate_counts={
            "visible_records": len(filtered_records),
            "eligible_chunks": len(chunk_rows),
            "semantic_candidates": len(semantic_candidates),
            "lexical_candidates": len(lexical_candidates),
            "reranked_candidates": len(deduped),
        },
        excluded_counts=excluded_counts,
        applied_filters=applied_filters(payload),
    )
    response.audit_id = write_audit_entry(service, payload, response, filtered_records)
    return response
