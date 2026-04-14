from __future__ import annotations

import asyncio
import hashlib
import math
import sqlite3
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from context_bucket.models import (
    ContextBucketAssembleRequest,
    ContextBucketAssembleResponse,
    ContextBucketChunk,
    ContextBucketContextBlock,
    ContextBucketContextItem,
    ContextBucketContextSection,
    ContextBucketDocumentImportItem,
    ContextBucketDocumentImportRequest,
    ContextBucketDocumentImportResponse,
    ContextBucketEvaluationCompareRequest,
    ContextBucketEvaluationCompareResponse,
    ContextBucketEvaluationGateRequest,
    ContextBucketEvaluationGateResponse,
    ContextBucketEvaluationRequest,
    ContextBucketEvaluationResponse,
    ContextBucketEvaluationRunRecord,
    ContextBucketEvaluationSuite,
    ContextBucketListResponse,
    ContextBucketDataSchema,
    ContextBucketPolicy,
    ContextBucketPrepareContextResponse,
    ContextBucketPruneResponse,
    ContextBucketQueryRequest,
    ContextBucketQueryResult,
    ContextBucketRecord,
    ContextBucketRecordCreate,
    ContextBucketRetrieveRequest,
    ContextBucketRetrieveResponse,
    ContextBucketSourceCreate,
    ContextBucketSourceDelete,
    ContextBucketSourceUpsert,
    ContextBucketStats,
    ContextBucketStructuredField,
    ContextBucketTaskEnvelopeResponse,
    ContextBucketWorkflowIntent,
    ContextBucketWorkflowPreference,
    ContextBucketWorkflowPreferenceUpdateRequest,
)
from context_bucket.assembly import (
    assemble_context as assemble_context_impl,
    build_context_sections,
    prepare_context as prepare_context_impl,
    render_context,
    section_for_item,
    section_order,
)
from context_bucket.audit import (
    applied_filters,
    write_audit_entry,
)
from context_bucket.evaluation import (
    compare_evaluation_runs as compare_evaluation_runs_impl,
    gate_evaluation_run as gate_evaluation_run_impl,
    list_evaluation_runs as list_evaluation_runs_impl,
    list_evaluation_suites as list_evaluation_suites_impl,
    run_evaluation_suite as run_evaluation_suite_impl,
    run_evaluations as run_evaluations_impl,
    save_evaluation_suite as save_evaluation_suite_impl,
)
from context_bucket.ingest import (
    create_record,
    delete_source as delete_source_impl,
    ingest_source as ingest_source_impl,
    store_record as store_record_impl,
    upsert_source as upsert_source_impl,
)
from context_bucket.importers import (
    document_source_key,
    expand_import_paths,
    import_path as import_path_impl,
    is_importable_file,
    parse_imported_content,
    read_importable_content,
)
from context_bucket.preferences import (
    latest_schema_record,
    merge_workflow_preference,
    summarize_preference_from_notes,
    workflow_defaults_from_preferences,
    workflow_preference_text,
)
from context_bucket.retrieval import (
    age_decay_factor,
    compress_context_items,
    cosine_similarity,
    dedupe_context_items,
    filter_records_for_retrieval as filter_records_for_retrieval_impl,
    lexical_overlap_score,
    lexical_tokens,
    provenance_from_index as provenance_from_index_impl,
    retrieve_context as retrieve_context_impl,
    rerank_score,
    scope_priority_bonus,
    selection_reason,
    semantic_terms,
    set_overlap_score,
    token_set_similarity,
)
from context_bucket.settings import Settings
from context_bucket.storage import (
    bootstrap_indexes,
    delete_record_locked,
    deserialize_index_row,
    find_latest_source,
    get_record_sync as get_record_sync_impl,
    load_chunk_index,
    load_chunk_index_sqlite,
    load_record_index,
    load_record_index_sqlite,
    load_records,
    load_records_sqlite,
    persist_record_locked,
    read_state_locked,
    rebuild_indexes_locked,
    rebuild_sqlite_indexes_locked,
    serialize_index_row,
    sqlite_connection,
    sqlite_records_connection,
    write_state_locked,
    ensure_sqlite_record_tables,
)
from context_bucket.structured import (
    collapse_whitespace,
    extract_structured_fields,
    flatten_json_text,
    infer_data_schema,
    infer_primary_text_paths,
    normalize_structured_data,
    prepare_content,
    resolve_data_schema,
    structured_text,
    value_type,
)
from context_bucket.task_envelope import (
    context_summary_from_blocks,
    derive_workflow_intent,
    intent_constraints,
    output_contract,
    prepare_task_envelope as prepare_task_envelope_impl,
    task_objective,
)
from context_bucket.training import (
    append_training_line,
    export_training as export_training_impl,
    training_line,
)

_TEXT_IMPORT_EXTENSIONS = {
    ".txt", ".md", ".rst", ".csv", ".json", ".yaml", ".yml", ".log", ".html", ".htm", ".xml", ".ndjson"
}


class _BaseEmbedder:
    backend_name = "base"

    def embed_terms(self, terms: list[str]) -> list[float]:
        raise NotImplementedError


class _HashingEmbedder(_BaseEmbedder):
    backend_name = "local_hashing"

    def __init__(self, dimensions: int) -> None:
        self.dimensions = max(8, int(dimensions))

    def embed_terms(self, terms: list[str]) -> list[float]:
        vector = [0.0] * self.dimensions
        if not terms:
            return vector
        for term in terms:
            digest = hashlib.sha256(term.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            weight = 1.0 + ((digest[5] % 11) / 50.0)
            vector[bucket] += sign * weight
        norm = math.sqrt(sum(value * value for value in vector))
        if norm <= 0:
            return [0.0] * self.dimensions
        return [value / norm for value in vector]


class ContextBucketService:
    def __init__(
        self,
        settings: Settings | None = None,
        embedder: _BaseEmbedder | None = None,
    ) -> None:
        self.settings = settings or Settings.from_env()
        self.root = self.settings.root_path
        self.records_root = self.root / "records"
        self.training_root = self.root / "training"
        self.index_root = self.root / "indexes"
        self.audit_root = self.root / "audit"
        self.evals_root = self.root / "evaluations"
        self.eval_suites_root = self.evals_root / "suites"
        self.eval_runs_root = self.evals_root / "runs"
        self.record_index_path = self.index_root / "records.json"
        self.chunk_index_path = self.index_root / "chunks.jsonl"
        self.sqlite_index_path = self.index_root / "context_bucket.sqlite3"
        self.sqlite_records_path = self.root / "context_bucket_records.sqlite3"
        self.audit_log_path = self.audit_root / "context_selection.jsonl"
        self.state_path = self.root / ".context_bucket-state.json"
        self._write_lock = asyncio.Lock()
        self._text_import_extensions = _TEXT_IMPORT_EXTENSIONS
        self._embedder = embedder or self._build_embedder()

    def _build_embedder(self) -> _BaseEmbedder:
        backend = getattr(self.settings, "embedding_backend", "local_hashing")
        if backend == "local_hashing":
            return _HashingEmbedder(self.settings.embedding_dimensions)
        raise ValueError(f"Unsupported embedding backend: {backend}")

    async def store_record(self, payload: ContextBucketRecordCreate) -> ContextBucketRecord | None:
        return await store_record_impl(self, payload)

    async def ingest_source(self, payload: ContextBucketSourceCreate) -> ContextBucketRecord | None:
        return await ingest_source_impl(self, payload)

    async def upsert_source(self, payload: ContextBucketSourceUpsert) -> ContextBucketRecord:
        return await upsert_source_impl(self, payload)

    async def delete_source(self, payload: ContextBucketSourceDelete) -> ContextBucketRecord | None:
        return await delete_source_impl(self, payload)

    async def import_path(self, payload: ContextBucketDocumentImportRequest) -> ContextBucketDocumentImportResponse:
        return await import_path_impl(self, payload)

    async def run_evaluations(self, payload: ContextBucketEvaluationRequest) -> ContextBucketEvaluationResponse:
        return await run_evaluations_impl(self, payload)

    async def save_evaluation_suite(
        self,
        suite_name: str,
        payload: ContextBucketEvaluationSuite,
    ) -> ContextBucketEvaluationSuite:
        return await save_evaluation_suite_impl(self, suite_name, payload)

    async def list_evaluation_suites(self) -> list[ContextBucketEvaluationSuite]:
        return await list_evaluation_suites_impl(self)

    async def run_evaluation_suite(self, suite_name: str) -> ContextBucketEvaluationResponse:
        return await run_evaluation_suite_impl(self, suite_name)

    async def list_evaluation_runs(self) -> list[ContextBucketEvaluationRunRecord]:
        return await list_evaluation_runs_impl(self)

    async def compare_evaluation_runs(
        self,
        payload: ContextBucketEvaluationCompareRequest,
    ) -> ContextBucketEvaluationCompareResponse:
        return await compare_evaluation_runs_impl(self, payload)

    async def gate_evaluation_run(
        self,
        payload: ContextBucketEvaluationGateRequest,
    ) -> ContextBucketEvaluationGateResponse:
        return await gate_evaluation_run_impl(self, payload)

    async def retrieve_context(self, payload: ContextBucketRetrieveRequest) -> ContextBucketRetrieveResponse:
        return await retrieve_context_impl(self, payload)

    async def assemble_context(self, payload: ContextBucketAssembleRequest) -> ContextBucketAssembleResponse:
        return await assemble_context_impl(self, payload)

    async def prepare_context(self, payload: ContextBucketAssembleRequest) -> ContextBucketPrepareContextResponse:
        return await prepare_context_impl(self, payload)

    async def prepare_task_envelope(self, payload: ContextBucketAssembleRequest) -> ContextBucketTaskEnvelopeResponse:
        return await prepare_task_envelope_impl(self, payload)

    async def update_workflow_preference(
        self,
        payload: ContextBucketWorkflowPreferenceUpdateRequest,
    ) -> ContextBucketRecord:
        existing = self._find_latest_source(
            payload.source_key,
            payload.scope,
            payload.user_id,
            payload.session_id,
        )
        base = ContextBucketWorkflowPreference()
        if existing and isinstance(existing.structured_data, dict):
            base = ContextBucketWorkflowPreference.model_validate(existing.structured_data)
        merged = self._merge_workflow_preference(
            base=base,
            preference_data=payload.preference_data,
            approved_text=payload.approved_text or "",
        )
        record = await self.upsert_source(
            ContextBucketSourceUpsert(
                source_key=payload.source_key,
                kind="user_profile_note",
                scope=payload.scope,
                title=payload.title or "User workflow preference",
                summary=payload.summary or "Aggregated workflow preference profile.",
                user_id=payload.user_id,
                session_id=payload.session_id,
                text=self._workflow_preference_text(merged),
                structured_data=merged.model_dump(mode="json"),
                data_schema=ContextBucketDataSchema(
                    schema_name="user_workflow_preference",
                    schema_mode="declared",
                    root_type="object",
                    primary_text_paths=[
                        "autonomy_level",
                        "clarification_preference",
                        "brevity_preference",
                        "structure_preference",
                        "initiative_preference",
                        "risk_tolerance",
                        "evidence_preference",
                        "style_notes",
                    ],
                ),
                tags=list(payload.tags),
                metadata=dict(payload.metadata),
            )
        )
        return record

    async def explain_context_selection(self, payload: ContextBucketRetrieveRequest) -> ContextBucketRetrieveResponse:
        return await self.retrieve_context(payload)

    async def query(self, payload: ContextBucketQueryRequest) -> list[ContextBucketQueryResult]:
        retrieved = await self.retrieve_context(
            ContextBucketRetrieveRequest(
                query_text=payload.query_text,
                session_id=payload.session_id,
                user_id=payload.user_id,
                include_user_scope=payload.include_user_scope,
                include_global_scope=payload.include_global_scope,
                kinds=payload.kinds,
                limit=payload.limit,
            )
        )
        return [
            ContextBucketQueryResult(
                record_id=item.record_id,
                chunk_id=item.chunk_id,
                kind=item.kind,
                scope=item.scope,
                title=item.title,
                text=item.text,
                summary=item.summary,
                urls=list(item.provenance.get("urls", [])),
                tags=list(item.provenance.get("tags", [])),
                metadata=dict(item.provenance.get("metadata", {})),
                created_at=item.created_at,
                score=item.score,
                user_id=item.provenance.get("user_id"),
                session_id=item.provenance.get("session_id"),
            )
            for item in retrieved.items
        ]

    async def get_record(self, record_id: str) -> ContextBucketRecord | None:
        return self.get_record_sync(record_id)

    async def list_records(
        self,
        *,
        kind: str | None = None,
        scope: str | None = None,
        limit: int = 100,
    ) -> ContextBucketListResponse:
        records = self._load_records()
        items = [
            record for record in records
            if (not kind or record.kind == kind) and (not scope or record.scope == scope)
        ]
        items.sort(key=lambda item: item.created_at, reverse=True)
        return ContextBucketListResponse(items=items[:limit], total=len(items))

    async def prune(self) -> ContextBucketPruneResponse:
        async with self._write_lock:
            pruned_records, pruned_chunk_ids = self._prune_records_locked(datetime.now(timezone.utc))
            state = self._read_state_locked()
        return ContextBucketPruneResponse(
            pruned_records=pruned_records,
            pruned_chunk_ids=pruned_chunk_ids,
            state=state,
        )

    async def export_training(self) -> tuple[int, Path]:
        return await export_training_impl(self)

    async def stats(self) -> ContextBucketStats:
        records = self._load_record_index()
        by_kind = Counter(record["kind"] for record in records)
        by_scope = Counter(record["scope"] for record in records)
        by_source_type = Counter(record["source_type"] for record in records)
        by_content_class = Counter(record["content_class"] for record in records)
        by_source_status = Counter(record["source_status"] for record in records)
        training_file_count = len(list(self.training_root.glob("*.jsonl"))) if self.training_root.exists() else 0
        return ContextBucketStats(
            root=str(self.root),
            record_count=len(records),
            training_file_count=training_file_count,
            records_by_kind=dict(by_kind),
            records_by_scope=dict(by_scope),
            records_by_source_type=dict(by_source_type),
            records_by_content_class=dict(by_content_class),
            records_by_source_status=dict(by_source_status),
            state=self._read_state_locked(),
        )

    def health(self) -> dict[str, Any]:
        return {"ok": True, "root": str(self.root)}

    @staticmethod
    def token_estimate(text: str) -> int:
        return max(1, math.ceil(len(text) / 4))

    async def _create_record(
        self,
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
        data_schema: ContextBucketDataSchema | None,
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
        return await create_record(
            self,
            kind=kind,
            text=text,
            scope=scope,
            title=title,
            summary=summary,
            user_id=user_id,
            session_id=session_id,
            source_id=source_id,
            urls=urls,
            tags=tags,
            metadata=metadata,
            structured_data=structured_data,
            data_schema=data_schema,
            content_class=content_class,
            source_type=source_type,
            source_key=source_key,
            external_id=external_id,
            source_updated_at=source_updated_at,
            source_last_synced_at=source_last_synced_at,
            policy=policy,
            skip_duplicate_kinds=skip_duplicate_kinds,
            source_version=source_version,
            allow_duplicate=allow_duplicate,
        )

    def _chunk_text(self, record_id: str, text: str) -> list[ContextBucketChunk]:
        max_chars = max(300, int(self.settings.chunk_chars))
        overlap = max(0, min(int(self.settings.chunk_overlap_chars), max_chars // 2))
        chunks: list[ContextBucketChunk] = []
        cursor = 0
        index = 0
        while cursor < len(text):
            end = min(len(text), cursor + max_chars)
            chunk_text = text[cursor:end].strip()
            if chunk_text:
                lexical_tokens = self._lexical_tokens(chunk_text)
                semantic_terms = self._semantic_terms(chunk_text)
                chunks.append(
                    ContextBucketChunk(
                        chunk_id=f"{record_id}:{index}",
                        record_id=record_id,
                        chunk_index=index,
                        text=chunk_text,
                        token_count_estimate=self.token_estimate(chunk_text),
                        lexical_tokens=lexical_tokens,
                        semantic_terms=semantic_terms,
                        embedding=self._embed(semantic_terms),
                    )
                )
                index += 1
            if end >= len(text):
                break
            cursor = max(0, end - overlap)
        return chunks

    @staticmethod
    def _lexical_tokens(text: str) -> list[str]:
        return lexical_tokens(text)

    def _semantic_terms(self, text: str) -> list[str]:
        return semantic_terms(text)

    @staticmethod
    def _stem_token(token: str) -> str:
        stem = token.lower()
        for suffix in ("ingly", "edly", "ment", "tion", "ions", "ness", "able", "ible", "ing", "ers", "ies", "ied", "ed", "es", "s"):
            if len(stem) > (len(suffix) + 2) and stem.endswith(suffix):
                if suffix in {"ies", "ied"}:
                    return stem[: -len(suffix)] + "y"
                return stem[: -len(suffix)]
        return stem

    @staticmethod
    def _char_ngrams(token: str, n: int) -> list[str]:
        padded = f"^{token}$"
        if len(padded) <= n:
            return [padded]
        return [padded[index:index + n] for index in range(len(padded) - n + 1)]

    def _embed(self, terms: list[str]) -> list[float]:
        return self._embedder.embed_terms(terms)

    def _expand_import_paths(self, target: Path, *, recursive: bool) -> list[Path]:
        return expand_import_paths(self, target, recursive=recursive)

    @staticmethod
    def _document_source_key(path: Path) -> str:
        return document_source_key(path)

    @staticmethod
    def _is_importable_file(path: Path) -> bool:
        return is_importable_file(path=path, text_extensions=_TEXT_IMPORT_EXTENSIONS)

    def _read_importable_content(self, path: Path) -> dict[str, Any] | None:
        return read_importable_content(self, path)

    def _parse_imported_content(self, suffix: str, raw_text: str) -> dict[str, Any]:
        return parse_imported_content(self, suffix, raw_text)

    def _flatten_json_text(self, payload: Any) -> str:
        return flatten_json_text(payload)

    def _prepare_content(
        self,
        text: str | None,
        structured_data: Any | None,
        data_schema: ContextBucketDataSchema | None,
    ) -> dict[str, Any]:
        return prepare_content(
            text,
            structured_data,
            data_schema,
            lexical_tokens_fn=self._lexical_tokens,
            semantic_terms_fn=self._semantic_terms,
        )

    def _resolve_data_schema(
        self,
        structured_data: Any | None,
        data_schema: ContextBucketDataSchema | None,
    ) -> ContextBucketDataSchema | None:
        return resolve_data_schema(structured_data, data_schema)

    def _infer_data_schema(self, structured_data: Any | None) -> ContextBucketDataSchema | None:
        return infer_data_schema(structured_data)

    def _extract_structured_fields(self, payload: Any, prefix: str = "") -> list[ContextBucketStructuredField]:
        return extract_structured_fields(payload, prefix)

    def _infer_primary_text_paths(self, fields: list[ContextBucketStructuredField]) -> list[str]:
        return infer_primary_text_paths(fields)

    def _structured_text(
        self,
        fields: list[ContextBucketStructuredField],
        primary_paths: list[str],
        *,
        include_all_fields: bool = True,
    ) -> str:
        return structured_text(fields, primary_paths, include_all_fields=include_all_fields)

    @staticmethod
    def _declared_user_intent_schema(data_schema: ContextBucketDataSchema | None) -> bool:
        if data_schema is None:
            return False
        return data_schema.schema_mode == "declared" and (data_schema.schema_name or "").strip().lower() == "user_intent"

    def _normalize_structured_data(self, payload: Any | None) -> Any | None:
        return normalize_structured_data(payload)

    @staticmethod
    def _value_type(value: Any) -> str:
        return value_type(value)

    @staticmethod
    def _collapse_whitespace(text: str) -> str:
        return collapse_whitespace(text)

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        return cosine_similarity(left, right)

    def _lexical_overlap_score(self, query_tokens: list[str], chunk_tokens: list[str]) -> float:
        return lexical_overlap_score(query_tokens, chunk_tokens)

    @staticmethod
    def _set_overlap_score(left: list[str], right: list[str]) -> float:
        return set_overlap_score(left, right)

    def _filter_records_for_retrieval(
        self,
        records: list[ContextBucketRecord],
        payload: ContextBucketRetrieveRequest,
    ) -> list[ContextBucketRecord]:
        return filter_records_for_retrieval_impl(self, records, payload)

    def _rerank_score(
        self,
        *,
        record: ContextBucketRecord | dict[str, Any],
        semantic_score: float,
        lexical_score: float,
        keyword_bonus: float,
        metadata_bonus: float,
    ) -> float:
        return rerank_score(
            record=record,
            semantic_score=semantic_score,
            lexical_score=lexical_score,
            keyword_bonus=keyword_bonus,
            metadata_bonus=metadata_bonus,
            settings=self.settings,
            query_max_age_days_fn=self._query_max_age_days,
        )

    @staticmethod
    def _scope_priority_bonus(scope: str) -> float:
        return scope_priority_bonus(scope)

    def _selection_reason(
        self,
        scope: str,
        semantic_score: float,
        lexical_score: float,
        keyword_bonus: float,
    ) -> str:
        return selection_reason(scope, semantic_score, lexical_score, keyword_bonus)

    @staticmethod
    def _dedupe_context_items(items: list[ContextBucketContextItem]) -> list[ContextBucketContextItem]:
        return dedupe_context_items(items)

    def _record_rank_bonus(self, record: ContextBucketRecord | dict[str, Any]) -> float:
        from context_bucket.retrieval import record_rank_bonus

        return record_rank_bonus(record)

    def _age_decay_factor(self, record: ContextBucketRecord | dict[str, Any]) -> float:
        return age_decay_factor(record, settings=self.settings, query_max_age_days_fn=self._query_max_age_days)

    def _query_max_age_days(self, record: ContextBucketRecord | dict[str, Any]) -> int:
        policy = record.get("policy") if isinstance(record, dict) else None
        if isinstance(record, dict):
            if policy and policy.get("freshness_days") is not None:
                return int(policy["freshness_days"])
            kind = record["kind"]
        else:
            if record.policy.freshness_days is not None:
                return int(record.policy.freshness_days)
            kind = record.kind
        if kind == "research_report":
            return int(self.settings.stale_research_report_days)
        if kind == "assistant_answer":
            return int(self.settings.stale_assistant_answer_days)
        if kind == "topic_pattern":
            return int(self.settings.stale_topic_pattern_days)
        if kind == "user_profile_note":
            return int(self.settings.stale_user_profile_note_days)
        return int(self.settings.retention_days)

    def _group_limit(self, kind: str) -> int:
        return max(4, int(self.settings.max_records_per_scope_kind))

    def _is_near_duplicate(
        self,
        *,
        kind: str,
        scope: str,
        lexical_tokens: list[str],
        session_id: str | None,
        user_id: str | None,
    ) -> bool:
        threshold = float(self.settings.dedup_threshold)
        lookback = max(1, int(self.settings.dedup_lookback))
        recent = [
            record for record in self._load_record_index()
            if record["kind"] == kind and record["scope"] == scope
        ]
        if scope == "session":
            recent = [record for record in recent if record.get("session_id") == session_id]
        elif scope == "user":
            recent = [record for record in recent if record.get("user_id") == user_id]
        recent.sort(key=lambda item: item["created_at"], reverse=True)
        needle = set(lexical_tokens)
        for record in recent[:lookback]:
            hay = set(record.get("lexical_tokens", []))
            if not needle or not hay:
                continue
            jaccard = len(needle & hay) / max(1, len(needle | hay))
            if jaccard >= threshold:
                return True
        return False

    def _prune_records_locked(self, now: datetime) -> tuple[int, list[str]]:
        if not self.records_root.exists():
            return 0, []
        records = self._load_records()
        grouped: dict[tuple[str, str], list[ContextBucketRecord]] = {}
        remove_ids: set[str] = set()
        for record in records:
            retention_days = record.policy.retention_days or max(1, int(self.settings.retention_days))
            expired_cutoff = now - timedelta(days=retention_days)
            if record.created_at < expired_cutoff:
                remove_ids.add(record.id)
            grouped.setdefault((record.scope, record.kind), []).append(record)
        for (_, kind), items in grouped.items():
            items.sort(key=lambda item: item.created_at, reverse=True)
            for record in items[self._group_limit(kind):]:
                remove_ids.add(record.id)
        pruned_chunk_ids: list[str] = []
        pruned_records = 0
        for record_id in sorted(remove_ids):
            record = self.get_record_sync(record_id)
            if record is None:
                continue
            pruned_chunk_ids.extend(chunk.chunk_id for chunk in record.chunks)
            self._delete_record_locked(record_id)
            pruned_records += 1
        self._rebuild_indexes_locked(self._load_records())
        return pruned_records, pruned_chunk_ids

    def _load_records(self) -> list[ContextBucketRecord]:
        return load_records(self)

    def _find_latest_source(
        self,
        source_key: str,
        scope: str,
        user_id: str | None,
        session_id: str | None,
    ) -> ContextBucketRecord | None:
        return find_latest_source(self, source_key, scope, user_id, session_id)

    async def _update_record_lifecycle(
        self,
        record_id: str,
        *,
        source_status: str,
        deleted_at: datetime | None = None,
    ) -> None:
        async with self._write_lock:
            record = self.get_record_sync(record_id)
            if record is None:
                return
            updated = record.model_copy(
                update={
                    "source_status": source_status,
                    "deleted_at": deleted_at,
                    "source_last_synced_at": datetime.now(timezone.utc),
                }
            )
            self._persist_record_locked(updated)
            self._rebuild_indexes_locked(self._load_records())

    @staticmethod
    def _dedupe_strings(values: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            cleaned = str(value).strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            result.append(cleaned)
        return result

    def _provenance_from_index(self, record: dict[str, Any]) -> dict[str, Any]:
        return provenance_from_index_impl(record)

    def _compress_context_items(self, items: list[ContextBucketContextItem]) -> list[ContextBucketContextItem]:
        return compress_context_items(items, lexical_tokens_fn=self._lexical_tokens)

    def _build_context_sections(
        self,
        assembly_mode: str,
        items: list[ContextBucketContextItem],
    ) -> list[ContextBucketContextSection]:
        return build_context_sections(self, assembly_mode, items)

    def _section_order(self, assembly_mode: str) -> list[str]:
        return section_order(assembly_mode)

    def _section_for_item(self, assembly_mode: str, item: ContextBucketContextItem) -> str:
        return section_for_item(assembly_mode, item)

    def _render_context(
        self,
        query_text: str,
        assembly_mode: str,
        sections: list[ContextBucketContextSection],
    ) -> str:
        return render_context(query_text, assembly_mode, sections)

    def _derive_workflow_intent(
        self,
        payload: ContextBucketAssembleRequest,
        context_blocks: list[ContextBucketContextBlock],
    ) -> ContextBucketWorkflowIntent:
        return derive_workflow_intent(payload, context_blocks)

    def _derive_user_workflow_preference(self, payload: ContextBucketRetrieveRequest) -> ContextBucketWorkflowPreference:
        records = self._filter_records_for_retrieval(self._load_records(), payload)
        preference_record = self._latest_schema_record(records, "user_workflow_preference")
        if preference_record and isinstance(preference_record.structured_data, dict):
            data = dict(preference_record.structured_data)
            return ContextBucketWorkflowPreference.model_validate(
                {
                    "autonomy_level": data.get("autonomy_level", "medium"),
                    "clarification_preference": data.get("clarification_preference", "medium"),
                    "brevity_preference": data.get("brevity_preference", "medium"),
                    "structure_preference": data.get("structure_preference", "medium"),
                    "initiative_preference": data.get("initiative_preference", "medium"),
                    "risk_tolerance": data.get("risk_tolerance", "moderate"),
                    "evidence_preference": data.get("evidence_preference", "medium"),
                    "style_preferences": data.get("style_preferences", {}),
                    "workflow_defaults": data.get("workflow_defaults", {}),
                    "style_notes": data.get("style_notes", []),
                    "evidence_count": int(data.get("evidence_count") or 1),
                }
            )

        notes = [
            record for record in records
            if record.kind in {"user_profile_note", "decision_outcome", "training_example"}
            and record.source_status == "active"
        ]
        return self._summarize_preference_from_notes(notes)

    def _merge_workflow_preference(
        self,
        *,
        base: ContextBucketWorkflowPreference,
        preference_data: dict[str, Any],
        approved_text: str,
    ) -> ContextBucketWorkflowPreference:
        return merge_workflow_preference(
            base=base,
            preference_data=preference_data,
            approved_text=approved_text,
        )

    @staticmethod
    def _workflow_preference_text(preference: ContextBucketWorkflowPreference) -> str:
        return workflow_preference_text(preference)

    def _latest_schema_record(self, records: list[ContextBucketRecord], schema_name: str) -> ContextBucketRecord | None:
        return latest_schema_record(records, schema_name)

    def _summarize_preference_from_notes(self, records: list[ContextBucketRecord]) -> ContextBucketWorkflowPreference:
        return summarize_preference_from_notes(records)

    @staticmethod
    def _workflow_defaults_from_preferences(
        *,
        prefer_best_effort: bool,
        prefer_next_step: bool,
        prefer_evidence: bool,
    ) -> dict[str, dict[str, Any]]:
        return workflow_defaults_from_preferences(
            prefer_best_effort=prefer_best_effort,
            prefer_next_step=prefer_next_step,
            prefer_evidence=prefer_evidence,
        )

    @staticmethod
    def _intent_constraints(query_text: str, workflow_type: str) -> list[str]:
        return intent_constraints(query_text, workflow_type)

    @staticmethod
    def _context_summary(context_blocks: list[ContextBucketContextBlock]) -> str:
        return context_summary_from_blocks(context_blocks)

    @staticmethod
    def _task_objective(
        workflow_intent: ContextBucketWorkflowIntent,
        context_blocks: list[ContextBucketContextBlock],
    ) -> str:
        return task_objective(workflow_intent, context_blocks)

    @staticmethod
    def _output_contract(
        workflow_intent: ContextBucketWorkflowIntent,
        workflow_preference: ContextBucketWorkflowPreference,
    ) -> dict[str, Any]:
        return output_contract(workflow_intent, workflow_preference)

    @staticmethod
    def _token_set_similarity(left: set[str], right: set[str]) -> float:
        return token_set_similarity(left, right)

    def _empty_retrieve_response(self, payload: ContextBucketRetrieveRequest) -> ContextBucketRetrieveResponse:
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
            applied_filters=self._applied_filters(payload),
        )
        response.audit_id = self._write_audit_entry(payload, response, [])
        return response

    def _record_summary(self, record: ContextBucketRecord) -> dict[str, Any]:
        return {
            "id": record.id,
            "kind": record.kind,
            "scope": record.scope,
            "source_type": record.source_type,
            "content_class": record.content_class,
            "title": record.title,
            "summary": record.summary,
            "user_id": record.user_id,
            "session_id": record.session_id,
            "source_id": record.source_id,
            "source_key": record.source_key,
            "source_version": record.source_version,
            "source_status": record.source_status,
            "external_id": record.external_id,
            "content_checksum": record.content_checksum,
            "source_updated_at": record.source_updated_at,
            "source_last_synced_at": record.source_last_synced_at,
            "deleted_at": record.deleted_at,
            "urls": list(record.urls),
            "tags": list(record.tags),
            "metadata": dict(record.metadata),
            "structured_data": record.structured_data,
            "data_schema": record.data_schema.model_dump(mode="json") if record.data_schema else None,
            "structured_fields": [field.model_dump(mode="json") for field in record.structured_fields],
            "policy": record.policy.model_dump(mode="json"),
            "created_at": record.created_at,
            "lexical_tokens": list(record.lexical_tokens),
            "schema_field_paths": list(record.data_schema.field_paths if record.data_schema else []),
        }

    def _chunk_summary(self, record: ContextBucketRecord, chunk: ContextBucketChunk) -> dict[str, Any]:
        return {
            "chunk_id": chunk.chunk_id,
            "record_id": record.id,
            "chunk_index": chunk.chunk_index,
            "text": chunk.text,
            "token_count_estimate": chunk.token_count_estimate,
            "lexical_tokens": list(chunk.lexical_tokens),
            "semantic_terms": list(chunk.semantic_terms),
            "embedding": list(chunk.embedding),
        }

    def _rebuild_indexes_locked(self, records: list[ContextBucketRecord]) -> None:
        rebuild_indexes_locked(self, records)

    def _load_record_index(self) -> list[dict[str, Any]]:
        return load_record_index(self)

    def _load_chunk_index(self) -> list[dict[str, Any]]:
        return load_chunk_index(self)

    def _bootstrap_indexes(self) -> None:
        bootstrap_indexes(self)

    def _rebuild_sqlite_indexes_locked(
        self,
        records: list[ContextBucketRecord],
        record_rows: list[dict[str, Any]],
    ) -> None:
        rebuild_sqlite_indexes_locked(self, records, record_rows)

    def _load_record_index_sqlite(self) -> list[dict[str, Any]]:
        return load_record_index_sqlite(self)

    def _load_chunk_index_sqlite(self) -> list[dict[str, Any]]:
        return load_chunk_index_sqlite(self)

    def _sqlite_connection(self) -> sqlite3.Connection:
        return sqlite_connection(self)

    def _sqlite_records_connection(self) -> sqlite3.Connection:
        return sqlite_records_connection(self)

    def get_record_sync(self, record_id: str) -> ContextBucketRecord | None:
        return get_record_sync_impl(self, record_id)

    def _persist_record_locked(self, record: ContextBucketRecord) -> None:
        persist_record_locked(self, record)

    def _delete_record_locked(self, record_id: str) -> None:
        delete_record_locked(self, record_id)

    def _load_records_sqlite(self) -> list[ContextBucketRecord]:
        return load_records_sqlite(self)

    @staticmethod
    def _ensure_sqlite_record_tables(connection: sqlite3.Connection) -> None:
        ensure_sqlite_record_tables(connection)

    @staticmethod
    def _serialize_index_row(row: dict[str, Any]) -> dict[str, Any]:
        return serialize_index_row(row)

    @staticmethod
    def _deserialize_index_row(row: dict[str, Any]) -> dict[str, Any]:
        return deserialize_index_row(row)

    def _applied_filters(self, payload: ContextBucketRetrieveRequest) -> dict[str, list[str]]:
        return applied_filters(payload)

    def _write_audit_entry(
        self,
        payload: ContextBucketRetrieveRequest,
        response: ContextBucketRetrieveResponse,
        visible_records: list[dict[str, Any]],
    ) -> str:
        return write_audit_entry(self, payload, response, visible_records)

    def _training_line(self, record: ContextBucketRecord) -> str:
        return training_line(record)

    def _append_training_line(self, record: ContextBucketRecord) -> None:
        append_training_line(self, record)

    def _read_state_locked(self) -> dict[str, Any]:
        return read_state_locked(self)

    def _write_state_locked(self, patch: dict[str, Any]) -> None:
        write_state_locked(self, patch)

    @staticmethod
    def _content_checksum(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
