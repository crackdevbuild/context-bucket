from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class Settings:
    data_root: str = ".local/context_bucket"
    record_backend: str = "file"
    retention_days: int = 45
    max_records_per_scope_kind: int = 24
    stale_research_report_days: int = 60
    stale_assistant_answer_days: int = 14
    stale_topic_pattern_days: int = 30
    stale_user_profile_note_days: int = 120
    query_top_k: int = 6
    chunk_chars: int = 900
    chunk_overlap_chars: int = 120
    include_global_scope_by_default: bool = True
    include_user_scope_by_default: bool = True
    dedup_threshold: float = 0.85
    dedup_lookback: int = 10
    decay_start_pct: float = 0.5
    training_export_enabled: bool = True
    index_backend: str = "json"
    embedding_backend: str = "local_hashing"
    embedding_dimensions: int = 64
    semantic_candidate_multiplier: int = 4
    lexical_candidate_multiplier: int = 4
    semantic_score_weight: float = 0.6
    lexical_score_weight: float = 0.25
    keyword_bonus_weight: float = 0.08
    metadata_bonus_weight: float = 0.07

    @property
    def root_path(self) -> Path:
        return Path(self.data_root).resolve()

    @classmethod
    def from_env(cls) -> "Settings":
        defaults = cls()

        def _get_bool(name: str, default: bool) -> bool:
            raw = os.getenv(name)
            if raw is None:
                return default
            return raw.strip().lower() in {"1", "true", "yes", "on"}

        def _get_int(name: str, default: int) -> int:
            raw = os.getenv(name)
            return int(raw) if raw is not None and raw.strip() else default

        def _get_float(name: str, default: float) -> float:
            raw = os.getenv(name)
            return float(raw) if raw is not None and raw.strip() else default

        return cls(
            data_root=os.getenv("CONTEXT_BUCKET_ROOT", defaults.data_root),
            record_backend=os.getenv("CONTEXT_BUCKET_RECORD_BACKEND", defaults.record_backend),
            retention_days=_get_int("CONTEXT_BUCKET_RETENTION_DAYS", defaults.retention_days),
            max_records_per_scope_kind=_get_int(
                "CONTEXT_BUCKET_MAX_RECORDS_PER_SCOPE_KIND",
                defaults.max_records_per_scope_kind,
            ),
            stale_research_report_days=_get_int(
                "CONTEXT_BUCKET_STALE_RESEARCH_REPORT_DAYS",
                defaults.stale_research_report_days,
            ),
            stale_assistant_answer_days=_get_int(
                "CONTEXT_BUCKET_STALE_ASSISTANT_ANSWER_DAYS",
                defaults.stale_assistant_answer_days,
            ),
            stale_topic_pattern_days=_get_int(
                "CONTEXT_BUCKET_STALE_TOPIC_PATTERN_DAYS",
                defaults.stale_topic_pattern_days,
            ),
            stale_user_profile_note_days=_get_int(
                "CONTEXT_BUCKET_STALE_USER_PROFILE_NOTE_DAYS",
                defaults.stale_user_profile_note_days,
            ),
            query_top_k=_get_int("CONTEXT_BUCKET_QUERY_TOP_K", defaults.query_top_k),
            chunk_chars=_get_int("CONTEXT_BUCKET_CHUNK_CHARS", defaults.chunk_chars),
            chunk_overlap_chars=_get_int(
                "CONTEXT_BUCKET_CHUNK_OVERLAP_CHARS",
                defaults.chunk_overlap_chars,
            ),
            include_global_scope_by_default=_get_bool(
                "CONTEXT_BUCKET_INCLUDE_GLOBAL_SCOPE_BY_DEFAULT",
                defaults.include_global_scope_by_default,
            ),
            include_user_scope_by_default=_get_bool(
                "CONTEXT_BUCKET_INCLUDE_USER_SCOPE_BY_DEFAULT",
                defaults.include_user_scope_by_default,
            ),
            dedup_threshold=_get_float("CONTEXT_BUCKET_DEDUP_THRESHOLD", defaults.dedup_threshold),
            dedup_lookback=_get_int("CONTEXT_BUCKET_DEDUP_LOOKBACK", defaults.dedup_lookback),
            decay_start_pct=_get_float("CONTEXT_BUCKET_DECAY_START_PCT", defaults.decay_start_pct),
            training_export_enabled=_get_bool(
                "CONTEXT_BUCKET_TRAINING_EXPORT_ENABLED",
                defaults.training_export_enabled,
            ),
            index_backend=os.getenv(
                "CONTEXT_BUCKET_INDEX_BACKEND",
                defaults.index_backend,
            ),
            embedding_backend=os.getenv(
                "CONTEXT_BUCKET_EMBEDDING_BACKEND",
                defaults.embedding_backend,
            ),
            embedding_dimensions=_get_int(
                "CONTEXT_BUCKET_EMBEDDING_DIMENSIONS",
                defaults.embedding_dimensions,
            ),
            semantic_candidate_multiplier=_get_int(
                "CONTEXT_BUCKET_SEMANTIC_CANDIDATE_MULTIPLIER",
                defaults.semantic_candidate_multiplier,
            ),
            lexical_candidate_multiplier=_get_int(
                "CONTEXT_BUCKET_LEXICAL_CANDIDATE_MULTIPLIER",
                defaults.lexical_candidate_multiplier,
            ),
            semantic_score_weight=_get_float(
                "CONTEXT_BUCKET_SEMANTIC_SCORE_WEIGHT",
                defaults.semantic_score_weight,
            ),
            lexical_score_weight=_get_float(
                "CONTEXT_BUCKET_LEXICAL_SCORE_WEIGHT",
                defaults.lexical_score_weight,
            ),
            keyword_bonus_weight=_get_float(
                "CONTEXT_BUCKET_KEYWORD_BONUS_WEIGHT",
                defaults.keyword_bonus_weight,
            ),
            metadata_bonus_weight=_get_float(
                "CONTEXT_BUCKET_METADATA_BONUS_WEIGHT",
                defaults.metadata_bonus_weight,
            ),
        )
