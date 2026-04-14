from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


ContextBucketRecordKind = Literal[
    "user_prompt",
    "normalized_query",
    "assistant_answer",
    "routing_decision",
    "decision_trace",
    "topic_pattern",
    "research_plan",
    "search_query",
    "evidence_summary",
    "research_report",
    "research_finding",
    "user_profile_note",
    "decision_outcome",
    "training_example",
]
ContextBucketScope = Literal["session", "user", "global"]
ContextBucketSourceType = Literal["direct_text", "imported_document"]
ContextBucketContentClass = Literal[
    "raw_source",
    "normalized_memory",
    "retrieval_chunk",
    "assembled_context",
]
ContextBucketConfidentiality = Literal["private", "restricted", "shareable"]
ContextBucketAssemblyMode = Literal["assistant", "planner", "research", "drafting"]
ContextBucketSourceStatus = Literal["active", "superseded", "deleted"]
ContextBucketModelTarget = Literal["local", "hosted"]
ContextBucketImportMode = Literal["upsert", "create"]
ContextBucketSchemaMode = Literal["declared", "inferred"]


class ContextBucketPolicy(BaseModel):
    confidentiality: ContextBucketConfidentiality = "private"
    freshness_days: int | None = Field(default=None, ge=1)
    retention_days: int | None = Field(default=None, ge=1)
    allowed_user_ids: list[str] = Field(default_factory=list)
    allowed_session_ids: list[str] = Field(default_factory=list)
    allow_local_model_egress: bool = True
    allow_remote_model_egress: bool = False


class ContextBucketStructuredField(BaseModel):
    path: str
    value_text: str
    value_type: str


class ContextBucketDataSchema(BaseModel):
    schema_name: str | None = None
    schema_mode: ContextBucketSchemaMode = "inferred"
    root_type: str = "object"
    field_paths: list[str] = Field(default_factory=list)
    primary_text_paths: list[str] = Field(default_factory=list)


class ContextBucketChunk(BaseModel):
    chunk_id: str
    record_id: str
    chunk_index: int
    text: str
    token_count_estimate: int = 0
    lexical_tokens: list[str] = Field(default_factory=list)
    semantic_terms: list[str] = Field(default_factory=list)
    embedding: list[float] = Field(default_factory=list)


class ContextBucketRecord(BaseModel):
    id: str
    kind: ContextBucketRecordKind
    scope: ContextBucketScope = "session"
    owner_scope: ContextBucketScope = "session"
    source_type: ContextBucketSourceType = "direct_text"
    content_class: ContextBucketContentClass = "normalized_memory"
    title: str | None = None
    text: str
    summary: str | None = None
    user_id: str | None = None
    session_id: str | None = None
    source_id: str | None = None
    source_key: str | None = None
    source_version: int = 1
    source_status: ContextBucketSourceStatus = "active"
    external_id: str | None = None
    content_checksum: str | None = None
    source_updated_at: datetime | None = None
    source_last_synced_at: datetime | None = None
    deleted_at: datetime | None = None
    urls: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    structured_data: Any | None = None
    data_schema: ContextBucketDataSchema | None = None
    structured_fields: list[ContextBucketStructuredField] = Field(default_factory=list)
    policy: ContextBucketPolicy = Field(default_factory=ContextBucketPolicy)
    token_count_estimate: int = 0
    lexical_tokens: list[str] = Field(default_factory=list)
    semantic_terms: list[str] = Field(default_factory=list)
    embedding: list[float] = Field(default_factory=list)
    chunks: list[ContextBucketChunk] = Field(default_factory=list)
    created_at: datetime


class ContextBucketRecordCreate(BaseModel):
    kind: ContextBucketRecordKind
    text: str = Field(min_length=1)
    scope: ContextBucketScope = "session"
    title: str | None = None
    summary: str | None = None
    user_id: str | None = None
    session_id: str | None = None
    source_id: str | None = None
    urls: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContextBucketSourceCreate(BaseModel):
    text: str | None = None
    kind: ContextBucketRecordKind = "research_finding"
    scope: ContextBucketScope = "session"
    title: str | None = None
    summary: str | None = None
    user_id: str | None = None
    session_id: str | None = None
    source_type: ContextBucketSourceType = "direct_text"
    content_class: ContextBucketContentClass = "raw_source"
    source_key: str | None = None
    external_id: str | None = None
    source_updated_at: datetime | None = None
    source_last_synced_at: datetime | None = None
    urls: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    structured_data: Any | None = None
    data_schema: ContextBucketDataSchema | None = None
    policy: ContextBucketPolicy = Field(default_factory=ContextBucketPolicy)


class ContextBucketSourceUpsert(ContextBucketSourceCreate):
    source_key: str = Field(min_length=1)


class ContextBucketSourceDelete(BaseModel):
    source_key: str = Field(min_length=1)
    scope: ContextBucketScope = "session"
    user_id: str | None = None
    session_id: str | None = None
    deleted_at: datetime | None = None


class ContextBucketDocumentImportRequest(BaseModel):
    path: str = Field(min_length=1)
    kind: ContextBucketRecordKind = "research_finding"
    scope: ContextBucketScope = "session"
    user_id: str | None = None
    session_id: str | None = None
    recursive: bool = False
    import_mode: ContextBucketImportMode = "upsert"
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    data_schema: ContextBucketDataSchema | None = None
    policy: ContextBucketPolicy = Field(default_factory=ContextBucketPolicy)


class ContextBucketDocumentImportItem(BaseModel):
    path: str
    source_key: str
    status: Literal["imported", "skipped"]
    reason: str | None = None
    record_id: str | None = None


class ContextBucketDocumentImportResponse(BaseModel):
    imported: int = 0
    skipped: int = 0
    items: list[ContextBucketDocumentImportItem] = Field(default_factory=list)


class ContextBucketQueryRequest(BaseModel):
    query_text: str = Field(min_length=1)
    session_id: str | None = None
    user_id: str | None = None
    include_user_scope: bool | None = None
    include_global_scope: bool | None = None
    kinds: list[ContextBucketRecordKind] | None = None
    limit: int | None = Field(default=None, ge=1, le=100)


class ContextBucketQueryResult(BaseModel):
    record_id: str
    chunk_id: str
    kind: ContextBucketRecordKind
    scope: ContextBucketScope
    title: str | None = None
    text: str
    summary: str | None = None
    user_id: str | None = None
    session_id: str | None = None
    urls: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    score: float = 0.0


class ContextBucketRetrieveRequest(BaseModel):
    query_text: str = Field(min_length=1)
    intent_data: Any | None = None
    intent_schema: ContextBucketDataSchema | None = None
    session_id: str | None = None
    user_id: str | None = None
    model_target: ContextBucketModelTarget = "local"
    enforce_policy: bool = True
    include_user_scope: bool | None = None
    include_global_scope: bool | None = None
    source_types: list[ContextBucketSourceType] | None = None
    content_classes: list[ContextBucketContentClass] | None = None
    kinds: list[ContextBucketRecordKind] | None = None
    tags: list[str] | None = None
    source_keys: list[str] | None = None
    confidentialities: list[ContextBucketConfidentiality] | None = None
    source_statuses: list[ContextBucketSourceStatus] | None = None
    limit: int | None = Field(default=None, ge=1, le=100)
    max_age_days: int | None = Field(default=None, ge=1)
    token_budget: int | None = Field(default=None, ge=50, le=12000)


class ContextBucketContextItem(BaseModel):
    record_id: str
    chunk_id: str
    kind: ContextBucketRecordKind
    scope: ContextBucketScope
    source_type: ContextBucketSourceType = "direct_text"
    content_class: ContextBucketContentClass = "normalized_memory"
    title: str | None = None
    text: str
    summary: str | None = None
    score: float = 0.0
    semantic_score: float = 0.0
    lexical_score: float = 0.0
    provenance: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    token_count_estimate: int = 0
    selection_reason: str | None = None


class ContextBucketRetrieveResponse(BaseModel):
    query_text: str
    items: list[ContextBucketContextItem] = Field(default_factory=list)
    total_candidates: int = 0
    retrieval_strategy: str = "semantic_lexical_hybrid"
    candidate_counts: dict[str, int] = Field(default_factory=dict)
    excluded_counts: dict[str, int] = Field(default_factory=dict)
    applied_filters: dict[str, list[str]] = Field(default_factory=dict)
    audit_id: str | None = None


class ContextBucketAssembleRequest(ContextBucketRetrieveRequest):
    assembly_mode: ContextBucketAssemblyMode = "assistant"
    token_budget: int = Field(default=1200, ge=50, le=12000)


class ContextBucketContextSection(BaseModel):
    name: str
    items: list[ContextBucketContextItem] = Field(default_factory=list)
    token_count_estimate: int = 0


class ContextBucketAssembleResponse(BaseModel):
    query_text: str
    context_text: str
    items: list[ContextBucketContextItem] = Field(default_factory=list)
    sections: list[ContextBucketContextSection] = Field(default_factory=list)
    assembly_mode: ContextBucketAssemblyMode = "assistant"
    token_budget: int
    token_count_estimate: int = 0
    omitted_items: int = 0
    truncation_reason: str | None = None
    retrieval_strategy: str = "semantic_lexical_hybrid"
    audit_id: str | None = None


class ContextBucketContextBlock(BaseModel):
    name: str
    text: str
    items: list[ContextBucketContextItem] = Field(default_factory=list)
    token_count_estimate: int = 0


class ContextBucketPrepareContextResponse(BaseModel):
    query_text: str
    request_summary: str
    context_blocks: list[ContextBucketContextBlock] = Field(default_factory=list)
    provenance: list[dict[str, Any]] = Field(default_factory=list)
    token_estimate: int = 0
    retrieval_strategy: str = "semantic_lexical_hybrid"
    audit_id: str | None = None


class ContextBucketWorkflowIntent(BaseModel):
    workflow_type: str = "analyze"
    action: str = "respond"
    target_type: str = "context"
    goal: str = "produce_helpful_result"
    constraints: list[str] = Field(default_factory=list)
    output_type: str = "text"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    source_refs: list[str] = Field(default_factory=list)


class ContextBucketWorkflowPreference(BaseModel):
    autonomy_level: str = "medium"
    clarification_preference: str = "medium"
    brevity_preference: str = "medium"
    structure_preference: str = "medium"
    initiative_preference: str = "medium"
    risk_tolerance: str = "moderate"
    evidence_preference: str = "medium"
    style_preferences: dict[str, str] = Field(default_factory=dict)
    workflow_defaults: dict[str, dict[str, Any]] = Field(default_factory=dict)
    style_notes: list[str] = Field(default_factory=list)
    evidence_count: int = 0


class ContextBucketWorkflowPreferenceUpdateRequest(BaseModel):
    scope: ContextBucketScope = "user"
    user_id: str | None = None
    session_id: str | None = None
    source_key: str = "user_workflow_preference"
    approved_text: str | None = None
    preference_data: dict[str, Any] = Field(default_factory=dict)
    title: str | None = None
    summary: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContextBucketTaskEnvelopeResponse(BaseModel):
    query_text: str
    objective: str
    workflow_intent: ContextBucketWorkflowIntent
    user_workflow_preference: ContextBucketWorkflowPreference
    context_blocks: list[ContextBucketContextBlock] = Field(default_factory=list)
    context_summary: str = ""
    output_contract: dict[str, Any] = Field(default_factory=dict)
    retrieval_strategy: str = "semantic_lexical_hybrid"
    token_estimate: int = 0
    audit_id: str | None = None


class ContextBucketListResponse(BaseModel):
    items: list[ContextBucketRecord] = Field(default_factory=list)
    total: int = 0


class ContextBucketStats(BaseModel):
    root: str
    record_count: int = 0
    training_file_count: int = 0
    records_by_kind: dict[str, int] = Field(default_factory=dict)
    records_by_scope: dict[str, int] = Field(default_factory=dict)
    records_by_source_type: dict[str, int] = Field(default_factory=dict)
    records_by_content_class: dict[str, int] = Field(default_factory=dict)
    records_by_source_status: dict[str, int] = Field(default_factory=dict)
    state: dict[str, Any] = Field(default_factory=dict)


class ContextBucketPruneResponse(BaseModel):
    pruned_records: int = 0
    pruned_chunk_ids: list[str] = Field(default_factory=list)
    state: dict[str, Any] = Field(default_factory=dict)


class ContextBucketExportResponse(BaseModel):
    exported_records: int = 0
    training_path: str


class ContextBucketHealth(BaseModel):
    ok: bool = True
    root: str


class ContextBucketEvaluationCase(BaseModel):
    name: str = Field(min_length=1)
    query_text: str = Field(min_length=1)
    session_id: str | None = None
    user_id: str | None = None
    model_target: ContextBucketModelTarget = "local"
    include_user_scope: bool | None = None
    include_global_scope: bool | None = None
    expected_source_keys: list[str] = Field(default_factory=list)
    expected_terms: list[str] = Field(default_factory=list)
    expected_terms_scope: Literal["assembled_context", "retrieved_records"] = "assembled_context"
    token_budget: int = Field(default=1200, ge=50, le=12000)


class ContextBucketEvaluationRequest(BaseModel):
    cases: list[ContextBucketEvaluationCase] = Field(default_factory=list)
    limit: int = Field(default=6, ge=1, le=100)


class ContextBucketEvaluationResult(BaseModel):
    name: str
    query_text: str
    retrieved_source_keys: list[str] = Field(default_factory=list)
    matched_expected_source_keys: list[str] = Field(default_factory=list)
    matched_expected_terms: list[str] = Field(default_factory=list)
    expected_terms_scope: str = "assembled_context"
    retrieval_hit: bool = False
    assembly_hit: bool = False
    retrieval_count: int = 0
    assembly_token_count_estimate: int = 0
    audit_id: str | None = None


class ContextBucketEvaluationResponse(BaseModel):
    total_cases: int = 0
    retrieval_hits: int = 0
    assembly_hits: int = 0
    results: list[ContextBucketEvaluationResult] = Field(default_factory=list)


class ContextBucketEvaluationSuite(BaseModel):
    name: str = Field(min_length=1)
    cases: list[ContextBucketEvaluationCase] = Field(default_factory=list)


class ContextBucketEvaluationRunRecord(BaseModel):
    run_id: str
    suite_name: str
    created_at: datetime
    total_cases: int = 0
    retrieval_hits: int = 0
    assembly_hits: int = 0


class ContextBucketEvaluationCompareRequest(BaseModel):
    baseline_run_id: str = Field(min_length=1)
    candidate_run_id: str = Field(min_length=1)


class ContextBucketEvaluationCompareItem(BaseModel):
    name: str
    retrieval_changed: bool = False
    assembly_changed: bool = False
    baseline_retrieval_hit: bool = False
    candidate_retrieval_hit: bool = False
    baseline_assembly_hit: bool = False
    candidate_assembly_hit: bool = False


class ContextBucketEvaluationCompareResponse(BaseModel):
    baseline_run_id: str
    candidate_run_id: str
    retrieval_regressions: int = 0
    assembly_regressions: int = 0
    items: list[ContextBucketEvaluationCompareItem] = Field(default_factory=list)


class ContextBucketEvaluationThresholds(BaseModel):
    min_retrieval_hit_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    min_assembly_hit_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    max_retrieval_regressions: int = Field(default=0, ge=0)
    max_assembly_regressions: int = Field(default=0, ge=0)


class ContextBucketEvaluationGateRequest(BaseModel):
    candidate_run_id: str = Field(min_length=1)
    baseline_run_id: str | None = None
    thresholds: ContextBucketEvaluationThresholds = Field(default_factory=ContextBucketEvaluationThresholds)


class ContextBucketEvaluationGateResponse(BaseModel):
    passed: bool = False
    candidate_run_id: str
    baseline_run_id: str | None = None
    total_cases: int = 0
    retrieval_hit_rate: float = 0.0
    assembly_hit_rate: float = 0.0
    retrieval_regressions: int = 0
    assembly_regressions: int = 0
    failures: list[str] = Field(default_factory=list)
