"""Unit tests for context_bucket/models.py"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone
from pydantic import ValidationError

from context_bucket.models import (
    ContextBucketRecord,
    ContextBucketRecordCreate,
    ContextBucketSourceCreate,
    ContextBucketSourceUpsert,
    ContextBucketSourceDelete,
    ContextBucketDataSchema,
    ContextBucketPolicy,
    ContextBucketChunk,
    ContextBucketRetrieveRequest,
    ContextBucketAssembleRequest,
    ContextBucketEvaluationCase,
    ContextBucketEvaluationRequest,
    ContextBucketWorkflowIntent,
    ContextBucketWorkflowPreference,
    ContextBucketWorkflowPreferenceUpdateRequest,
    ContextBucketDocumentImportRequest,
    ContextBucketEvaluationCompareRequest,
    ContextBucketEvaluationGateRequest,
    ContextBucketEvaluationThresholds,
)


class TestContextBucketPolicy:
    """Tests for ContextBucketPolicy model."""

    def test_default_values(self) -> None:
        """Test that default values are set correctly."""
        policy = ContextBucketPolicy()
        assert policy.confidentiality == "private"
        assert policy.freshness_days is None
        assert policy.retention_days is None
        assert policy.allowed_user_ids == []
        assert policy.allowed_session_ids == []
        assert policy.allow_local_model_egress is True
        assert policy.allow_remote_model_egress is False

    def test_custom_values(self) -> None:
        """Test that custom values are accepted."""
        policy = ContextBucketPolicy(
            confidentiality="restricted",
            freshness_days=30,
            retention_days=90,
            allowed_user_ids=["u1", "u2"],
            allowed_session_ids=["s1"],
            allow_local_model_egress=False,
            allow_remote_model_egress=True,
        )
        assert policy.confidentiality == "restricted"
        assert policy.freshness_days == 30
        assert policy.retention_days == 90
        assert policy.allowed_user_ids == ["u1", "u2"]
        assert policy.allow_local_model_egress is False
        assert policy.allow_remote_model_egress is True

    def test_freshness_days_validation(self) -> None:
        """Test that freshness_days must be >= 1."""
        with pytest.raises(ValidationError):
            ContextBucketPolicy(freshness_days=0)
        
        with pytest.raises(ValidationError):
            ContextBucketPolicy(freshness_days=-1)

    def test_retention_days_validation(self) -> None:
        """Test that retention_days must be >= 1."""
        with pytest.raises(ValidationError):
            ContextBucketPolicy(retention_days=0)


class TestContextBucketDataSchema:
    """Tests for ContextBucketDataSchema model."""

    def test_default_values(self) -> None:
        """Test default schema values."""
        schema = ContextBucketDataSchema()
        assert schema.schema_name is None
        assert schema.schema_mode == "inferred"
        assert schema.root_type == "object"
        assert schema.field_paths == []
        assert schema.primary_text_paths == []

    def test_declared_schema(self) -> None:
        """Test declared schema with custom paths."""
        schema = ContextBucketDataSchema(
            schema_name="user_intent",
            schema_mode="declared",
            root_type="object",
            primary_text_paths=["goal", "tone", "channel"],
        )
        assert schema.schema_name == "user_intent"
        assert schema.schema_mode == "declared"
        assert schema.primary_text_paths == ["goal", "tone", "channel"]


class TestContextBucketChunk:
    """Tests for ContextBucketChunk model."""

    def test_required_fields(self) -> None:
        """Test that required fields must be provided."""
        chunk = ContextBucketChunk(
            chunk_id="c1",
            record_id="r1",
            chunk_index=0,
            text="Sample text",
        )
        assert chunk.chunk_id == "c1"
        assert chunk.record_id == "r1"
        assert chunk.chunk_index == 0
        assert chunk.text == "Sample text"
        assert chunk.token_count_estimate == 0
        assert chunk.lexical_tokens == []

    def test_all_fields(self) -> None:
        """Test chunk with all fields populated."""
        chunk = ContextBucketChunk(
            chunk_id="c1",
            record_id="r1",
            chunk_index=0,
            text="Sample text",
            token_count_estimate=10,
            lexical_tokens=["sample", "text"],
            semantic_terms=["sample", "text", "example"],
            embedding=[0.1, 0.2, 0.3],
        )
        assert chunk.token_count_estimate == 10
        assert chunk.embedding == [0.1, 0.2, 0.3]


class TestContextBucketRecord:
    """Tests for ContextBucketRecord model."""

    def test_minimal_record(self) -> None:
        """Test creating a record with minimal required fields."""
        record = ContextBucketRecord(
            id="r1",
            kind="research_finding",
            text="Sample text",
            created_at=datetime.now(timezone.utc),
        )
        assert record.id == "r1"
        assert record.kind == "research_finding"
        assert record.scope == "session"
        assert record.source_type == "direct_text"
        assert record.content_class == "normalized_memory"
        assert record.source_status == "active"

    def test_full_record(self) -> None:
        """Test creating a record with all fields."""
        now = datetime.now(timezone.utc)
        record = ContextBucketRecord(
            id="r1",
            kind="research_report",
            scope="user",
            user_id="u1",
            source_key="test_key",
            source_version=2,
            text="Sample text",
            title="Test Title",
            summary="Test summary",
            tags=["tag1", "tag2"],
            metadata={"key": "value"},
            created_at=now,
        )
        assert record.scope == "user"
        assert record.user_id == "u1"
        assert record.source_version == 2


class TestContextBucketRecordCreate:
    """Tests for ContextBucketRecordCreate model."""

    def test_minimal_create(self) -> None:
        """Test minimal record creation request."""
        create = ContextBucketRecordCreate(kind="search_query", text="test query")
        assert create.kind == "search_query"
        assert create.text == "test query"
        assert create.scope == "session"

    def test_text_validation_empty(self) -> None:
        """Test that empty text is rejected."""
        with pytest.raises(ValidationError):
            ContextBucketRecordCreate(kind="search_query", text="")

    def test_text_validation_min_length(self) -> None:
        """Test that text must have min_length=1."""
        create = ContextBucketRecordCreate(kind="search_query", text="a")
        assert create.text == "a"


class TestContextBucketSourceCreate:
    """Tests for ContextBucketSourceCreate model."""

    def test_text_only_source(self) -> None:
        """Test creating source with text only."""
        source = ContextBucketSourceCreate(text="Sample text")
        assert source.text == "Sample text"
        assert source.kind == "research_finding"
        assert source.scope == "session"

    def test_structured_data_source(self) -> None:
        """Test creating source with structured data."""
        source = ContextBucketSourceCreate(
            structured_data={"goal": "draft update", "tone": "concise"},
            data_schema=ContextBucketDataSchema(
                schema_name="user_intent",
                schema_mode="declared",
                primary_text_paths=["goal", "tone"],
            ),
        )
        assert source.text is None
        assert source.structured_data == {"goal": "draft update", "tone": "concise"}

    def test_default_kind(self) -> None:
        """Test default kind value."""
        source = ContextBucketSourceCreate(text="test")
        assert source.kind == "research_finding"


class TestContextBucketSourceUpsert:
    """Tests for ContextBucketSourceUpsert model."""

    def test_source_key_required(self) -> None:
        """Test that source_key is required for upsert."""
        upsert = ContextBucketSourceUpsert(source_key="test_key", text="test")
        assert upsert.source_key == "test_key"

    def test_source_key_validation(self) -> None:
        """Test that source_key must have min_length=1."""
        with pytest.raises(ValidationError):
            ContextBucketSourceUpsert(source_key="", text="test")


class TestContextBucketSourceDelete:
    """Tests for ContextBucketSourceDelete model."""

    def test_minimal_delete(self) -> None:
        """Test minimal delete request."""
        delete = ContextBucketSourceDelete(source_key="test_key")
        assert delete.source_key == "test_key"
        assert delete.scope == "session"

    def test_source_key_required(self) -> None:
        """Test that source_key is required."""
        with pytest.raises(ValidationError):
            ContextBucketSourceDelete()


class TestContextBucketRetrieveRequest:
    """Tests for ContextBucketRetrieveRequest model."""

    def test_minimal_request(self) -> None:
        """Test minimal retrieve request."""
        req = ContextBucketRetrieveRequest(query_text="test query")
        assert req.query_text == "test query"
        assert req.model_target == "local"
        assert req.enforce_policy is True

    def test_query_text_validation(self) -> None:
        """Test that query_text must have min_length=1."""
        with pytest.raises(ValidationError):
            ContextBucketRetrieveRequest(query_text="")

    def test_limit_validation(self) -> None:
        """Test limit bounds."""
        # Valid limits
        req = ContextBucketRetrieveRequest(query_text="test", limit=1)
        assert req.limit == 1
        req = ContextBucketRetrieveRequest(query_text="test", limit=100)
        assert req.limit == 100
        
        # Invalid limits
        with pytest.raises(ValidationError):
            ContextBucketRetrieveRequest(query_text="test", limit=0)
        with pytest.raises(ValidationError):
            ContextBucketRetrieveRequest(query_text="test", limit=101)

    def test_token_budget_validation(self) -> None:
        """Test token_budget bounds."""
        req = ContextBucketRetrieveRequest(query_text="test", token_budget=50)
        assert req.token_budget == 50
        req = ContextBucketRetrieveRequest(query_text="test", token_budget=12000)
        assert req.token_budget == 12000
        
        with pytest.raises(ValidationError):
            ContextBucketRetrieveRequest(query_text="test", token_budget=49)
        with pytest.raises(ValidationError):
            ContextBucketRetrieveRequest(query_text="test", token_budget=12001)


class TestContextBucketAssembleRequest:
    """Tests for ContextBucketAssembleRequest model."""

    def test_defaults(self) -> None:
        """Test default values for assemble request."""
        req = ContextBucketAssembleRequest(query_text="test")
        assert req.assembly_mode == "assistant"
        assert req.token_budget == 1200

    def test_inherits_from_retrieve(self) -> None:
        """Test that assemble request inherits retrieve request fields."""
        req = ContextBucketAssembleRequest(
            query_text="test",
            session_id="s1",
            user_id="u1",
            assembly_mode="drafting",
            token_budget=500,
        )
        assert req.session_id == "s1"
        assert req.user_id == "u1"
        assert req.assembly_mode == "drafting"
        assert req.token_budget == 500


class TestContextBucketWorkflowIntent:
    """Tests for ContextBucketWorkflowIntent model."""

    def test_defaults(self) -> None:
        """Test default workflow intent values."""
        intent = ContextBucketWorkflowIntent()
        assert intent.workflow_type == "analyze"
        assert intent.action == "respond"
        assert intent.target_type == "context"
        assert intent.goal == "produce_helpful_result"
        assert intent.confidence == 0.0

    def test_confidence_bounds(self) -> None:
        """Test confidence validation (0.0 to 1.0)."""
        intent = ContextBucketWorkflowIntent(confidence=0.5)
        assert intent.confidence == 0.5
        
        intent = ContextBucketWorkflowIntent(confidence=1.0)
        assert intent.confidence == 1.0
        
        with pytest.raises(ValidationError):
            ContextBucketWorkflowIntent(confidence=-0.1)
        with pytest.raises(ValidationError):
            ContextBucketWorkflowIntent(confidence=1.1)


class TestContextBucketWorkflowPreference:
    """Tests for ContextBucketWorkflowPreference model."""

    def test_defaults(self) -> None:
        """Test default preference values."""
        pref = ContextBucketWorkflowPreference()
        assert pref.autonomy_level == "medium"
        assert pref.clarification_preference == "medium"
        assert pref.brevity_preference == "medium"
        assert pref.evidence_count == 0

    def test_custom_values(self) -> None:
        """Test custom preference values."""
        pref = ContextBucketWorkflowPreference(
            autonomy_level="high",
            clarification_preference="low",
            brevity_preference="high",
            style_preferences={"directness": "high"},
            workflow_defaults={"reply": {"draft_full_response": True}},
            style_notes=["prefer concise outputs"],
            evidence_count=5,
        )
        assert pref.autonomy_level == "high"
        assert pref.clarification_preference == "low"
        assert pref.style_preferences == {"directness": "high"}
        assert pref.evidence_count == 5


class TestContextBucketWorkflowPreferenceUpdateRequest:
    """Tests for ContextBucketWorkflowPreferenceUpdateRequest model."""

    def test_defaults(self) -> None:
        """Test default update request values."""
        req = ContextBucketWorkflowPreferenceUpdateRequest()
        assert req.scope == "user"
        assert req.source_key == "user_workflow_preference"

    def test_with_data(self) -> None:
        """Test update request with data."""
        req = ContextBucketWorkflowPreferenceUpdateRequest(
            user_id="u1",
            approved_text="Prefer concise outputs",
            preference_data={"clarification_preference": "low"},
        )
        assert req.user_id == "u1"
        assert req.approved_text == "Prefer concise outputs"
        assert req.preference_data == {"clarification_preference": "low"}


class TestContextBucketDocumentImportRequest:
    """Tests for ContextBucketDocumentImportRequest model."""

    def test_path_required(self) -> None:
        """Test that path is required."""
        req = ContextBucketDocumentImportRequest(path="/tmp/test.json")
        assert req.path == "/tmp/test.json"
        assert req.kind == "research_finding"
        assert req.scope == "session"

    def test_path_validation(self) -> None:
        """Test path validation."""
        with pytest.raises(ValidationError):
            ContextBucketDocumentImportRequest(path="")


class TestContextBucketEvaluationCase:
    """Tests for ContextBucketEvaluationCase model."""

    def test_required_fields(self) -> None:
        """Test required fields for evaluation case."""
        case = ContextBucketEvaluationCase(name="test_case", query_text="test query")
        assert case.name == "test_case"
        assert case.query_text == "test query"
        assert case.token_budget == 1200

    def test_name_validation(self) -> None:
        """Test that name must have min_length=1."""
        with pytest.raises(ValidationError):
            ContextBucketEvaluationCase(name="", query_text="test")

    def test_token_budget_bounds(self) -> None:
        """Test token_budget validation."""
        case = ContextBucketEvaluationCase(name="test", query_text="test", token_budget=50)
        assert case.token_budget == 50
        
        with pytest.raises(ValidationError):
            ContextBucketEvaluationCase(name="test", query_text="test", token_budget=49)


class TestContextBucketEvaluationRequest:
    """Tests for ContextBucketEvaluationRequest model."""

    def test_defaults(self) -> None:
        """Test default evaluation request values."""
        req = ContextBucketEvaluationRequest()
        assert req.cases == []
        assert req.limit == 6

    def test_limit_validation(self) -> None:
        """Test limit bounds (1-100)."""
        req = ContextBucketEvaluationRequest(limit=1)
        assert req.limit == 1
        req = ContextBucketEvaluationRequest(limit=100)
        assert req.limit == 100
        
        with pytest.raises(ValidationError):
            ContextBucketEvaluationRequest(limit=0)
        with pytest.raises(ValidationError):
            ContextBucketEvaluationRequest(limit=101)


class TestContextBucketEvaluationCompareRequest:
    """Tests for ContextBucketEvaluationCompareRequest model."""

    def test_required_fields(self) -> None:
        """Test required fields."""
        req = ContextBucketEvaluationCompareRequest(
            baseline_run_id="run1",
            candidate_run_id="run2",
        )
        assert req.baseline_run_id == "run1"
        assert req.candidate_run_id == "run2"

    def test_validation(self) -> None:
        """Test that run_ids are required."""
        with pytest.raises(ValidationError):
            ContextBucketEvaluationCompareRequest()


class TestContextBucketEvaluationGateRequest:
    """Tests for ContextBucketEvaluationGateRequest model."""

    def test_required_fields(self) -> None:
        """Test that candidate_run_id is required."""
        req = ContextBucketEvaluationGateRequest(candidate_run_id="run1")
        assert req.candidate_run_id == "run1"
        assert req.baseline_run_id is None

    def test_default_thresholds(self) -> None:
        """Test default threshold values."""
        req = ContextBucketEvaluationGateRequest(candidate_run_id="run1")
        assert req.thresholds.min_retrieval_hit_rate == 1.0
        assert req.thresholds.min_assembly_hit_rate == 1.0
        assert req.thresholds.max_retrieval_regressions == 0


class TestContextBucketEvaluationThresholds:
    """Tests for ContextBucketEvaluationThresholds model."""

    def test_defaults(self) -> None:
        """Test default threshold values."""
        thresholds = ContextBucketEvaluationThresholds()
        assert thresholds.min_retrieval_hit_rate == 1.0
        assert thresholds.min_assembly_hit_rate == 1.0
        assert thresholds.max_retrieval_regressions == 0
        assert thresholds.max_assembly_regressions == 0

    def test_bounds_validation(self) -> None:
        """Test threshold bounds."""
        # Valid values
        thresholds = ContextBucketEvaluationThresholds(
            min_retrieval_hit_rate=0.0,
            min_assembly_hit_rate=1.0,
            max_retrieval_regressions=5,
        )
        assert thresholds.min_retrieval_hit_rate == 0.0
        
        # Invalid values
        with pytest.raises(ValidationError):
            ContextBucketEvaluationThresholds(min_retrieval_hit_rate=-0.1)
        with pytest.raises(ValidationError):
            ContextBucketEvaluationThresholds(min_retrieval_hit_rate=1.1)