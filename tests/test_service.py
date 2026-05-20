from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone

from context_bucket.models import (
    ContextBucketAssembleRequest,
    ContextBucketDataSchema,
    ContextBucketDocumentImportRequest,
    ContextBucketEvaluationCase,
    ContextBucketEvaluationCompareRequest,
    ContextBucketEvaluationGateRequest,
    ContextBucketEvaluationRequest,
    ContextBucketEvaluationSuite,
    ContextBucketRecordCreate,
    ContextBucketRetrieveRequest,
    ContextBucketSourceCreate,
    ContextBucketSourceDelete,
    ContextBucketSourceUpsert,
    ContextBucketWorkflowPreferenceUpdateRequest,
)
from context_bucket.service import ContextBucketService
from context_bucket.settings import Settings
from context_bucket.task_envelope import derive_workflow_intent
import context_bucket
from workflow_fixtures import WORKFLOW_FIXTURES


def _service(tmp_path) -> ContextBucketService:  # type: ignore[no-untyped-def]
    return ContextBucketService(Settings(data_root=str(tmp_path / "bucket")))


def test_workflow_intent_prioritizes_code_over_generic_write() -> None:
    intent = derive_workflow_intent(
        ContextBucketAssembleRequest(query_text="write code to fix the parser"),
        [],
    )

    assert intent.workflow_type == "code"
    assert intent.action == "produce_code"
    assert intent.output_type == "code"
    assert intent.target_type == "codebase"
    assert intent.goal == "produce_code_change"


def test_workflow_intent_keeps_generic_write_as_draft() -> None:
    intent = derive_workflow_intent(
        ContextBucketAssembleRequest(query_text="write a client update"),
        [],
    )

    assert intent.workflow_type == "draft"
    assert intent.output_type == "draft"
    assert intent.target_type == "context"


def test_store_and_retrieve_scoped_records(tmp_path) -> None:  # type: ignore[no-untyped-def]
    service = _service(tmp_path)

    async def _scenario() -> None:
        await service.store_record(
            ContextBucketRecordCreate(
                kind="research_report",
                scope="session",
                session_id="s1",
                text="Complex products and limited attention shape consumer choice.",
            )
        )
        await service.store_record(
            ContextBucketRecordCreate(
                kind="research_finding",
                scope="user",
                user_id="u1",
                text="Choice architecture can improve welfare in complex markets.",
            )
        )
        response = await service.retrieve_context(
            ContextBucketRetrieveRequest(
                query_text="consumer choice complex products welfare",
                session_id="s1",
                user_id="u1",
                include_user_scope=True,
            )
        )
        assert len(response.items) == 2

    asyncio.run(_scenario())


def test_upsert_delete_and_scope_preference(tmp_path) -> None:  # type: ignore[no-untyped-def]
    service = _service(tmp_path)

    async def _scenario() -> None:
        first = await service.upsert_source(
            ContextBucketSourceUpsert(
                source_key="client_profile",
                scope="user",
                user_id="u1",
                kind="user_profile_note",
                text="Client prefers weekly status updates and concise summaries.",
            )
        )
        second = await service.upsert_source(
            ContextBucketSourceUpsert(
                source_key="client_profile",
                scope="user",
                user_id="u1",
                kind="user_profile_note",
                text="Client prefers concise status updates and email drafts.",
            )
        )
        assert first.source_version == 1
        assert second.source_version == 2

        hidden = await service.delete_source(
            ContextBucketSourceDelete(source_key="client_profile", scope="user", user_id="u1")
        )
        assert hidden is not None

        response = await service.retrieve_context(
            ContextBucketRetrieveRequest(
                query_text="email update",
                user_id="u1",
                include_user_scope=True,
            )
        )
        assert response.items == []

    asyncio.run(_scenario())


def test_structured_data_schema_and_retrieval(tmp_path) -> None:  # type: ignore[no-untyped-def]
    service = _service(tmp_path)

    async def _scenario() -> None:
        record = await service.ingest_source(
            ContextBucketSourceCreate(
                scope="user",
                user_id="u1",
                source_key="client_card",
                structured_data={
                    "client": {
                        "name": "Acme Legal",
                        "preferences": {"tone": "concise", "channel": "email"},
                    }
                },
                data_schema=ContextBucketDataSchema(
                    schema_name="client_profile",
                    schema_mode="declared",
                    root_type="object",
                    primary_text_paths=["client.name", "client.preferences.tone", "client.preferences.channel"],
                ),
            )
        )
        assert record is not None
        assert record.data_schema is not None

        retrieved = await service.retrieve_context(
            ContextBucketRetrieveRequest(
                query_text="client preferences tone concise email",
                user_id="u1",
                include_user_scope=True,
            )
        )
        assert retrieved.items
        assert retrieved.items[0].provenance["data_schema"]["schema_name"] == "client_profile"

    asyncio.run(_scenario())


def test_declared_user_intent_schema_drives_retrieval(tmp_path) -> None:  # type: ignore[no-untyped-def]
    service = _service(tmp_path)

    async def _scenario() -> None:
        await service.ingest_source(
            ContextBucketSourceCreate(
                scope="user",
                user_id="u1",
                source_key="client_card",
                structured_data={
                    "client": {
                        "name": "Acme Legal",
                        "preferences": {"tone": "concise", "channel": "email"},
                    }
                },
                data_schema=ContextBucketDataSchema(
                    schema_name="client_profile",
                    schema_mode="declared",
                    root_type="object",
                    primary_text_paths=["client.name", "client.preferences.tone", "client.preferences.channel"],
                ),
            )
        )

        retrieved = await service.retrieve_context(
            ContextBucketRetrieveRequest(
                query_text="help me with this request",
                intent_data={
                    "goal": "draft a client update",
                    "tone": "concise",
                    "channel": "email",
                    "debug_trace": "ignore internal routing notes",
                },
                intent_schema=ContextBucketDataSchema(
                    schema_name="user_intent",
                    schema_mode="declared",
                    root_type="object",
                    primary_text_paths=["goal", "tone", "channel"],
                ),
                user_id="u1",
                include_user_scope=True,
            )
        )
        assert retrieved.items
        assert retrieved.items[0].provenance["data_schema"]["schema_name"] == "client_profile"

    asyncio.run(_scenario())


def test_import_path_parses_json_and_prepare_context(tmp_path) -> None:  # type: ignore[no-untyped-def]
    service = _service(tmp_path)
    json_path = tmp_path / "profile.json"
    json_path.write_text(
        json.dumps({"client": {"preference": "concise email drafts", "status": "active"}}),
        encoding="utf-8",
    )

    async def _scenario() -> None:
        response = await service.import_path(
            ContextBucketDocumentImportRequest(
                path=str(json_path),
                scope="session",
                session_id="s1",
            )
        )
        assert response.imported == 1

        prepared = await service.prepare_context(
            ContextBucketAssembleRequest(
                query_text="draft a client update",
                session_id="s1",
                assembly_mode="drafting",
                token_budget=200,
            )
        )
        assert prepared.context_blocks
        assert prepared.provenance
        assert prepared.token_estimate > 0

    asyncio.run(_scenario())


def test_import_path_accepts_declared_data_schema(tmp_path) -> None:  # type: ignore[no-untyped-def]
    service = _service(tmp_path)
    json_path = tmp_path / "intent.json"
    json_path.write_text(
        json.dumps({"goal": "draft a client update", "tone": "concise", "channel": "email"}),
        encoding="utf-8",
    )

    async def _scenario() -> None:
        response = await service.import_path(
            ContextBucketDocumentImportRequest(
                path=str(json_path),
                scope="user",
                user_id="u1",
                data_schema=ContextBucketDataSchema(
                    schema_name="user_intent",
                    schema_mode="declared",
                    root_type="object",
                    primary_text_paths=["goal", "tone", "channel"],
                ),
            )
        )
        assert response.imported == 1
        assert response.items[0].record_id is not None

        record = await service.get_record(response.items[0].record_id or "")
        assert record is not None
        assert record.data_schema is not None
        assert record.data_schema.schema_name == "user_intent"
        assert record.data_schema.schema_mode == "declared"

    asyncio.run(_scenario())


def test_prepare_task_envelope_derives_workflow_and_preferences(tmp_path) -> None:  # type: ignore[no-untyped-def]
    service = _service(tmp_path)

    async def _scenario() -> None:
        await service.ingest_source(
            ContextBucketSourceCreate(
                scope="user",
                user_id="u1",
                source_key="workflow_prefs",
                kind="user_profile_note",
                structured_data={
                    "autonomy_level": "medium_high",
                    "clarification_preference": "low",
                    "brevity_preference": "high",
                    "initiative_preference": "high",
                    "workflow_defaults": {"reply": {"draft_full_response": True}},
                    "style_preferences": {"directness": "high"},
                    "style_notes": ["prefer concise outputs"],
                    "evidence_count": 4,
                },
                data_schema=ContextBucketDataSchema(
                    schema_name="user_workflow_preference",
                    schema_mode="declared",
                    root_type="object",
                    primary_text_paths=[
                        "autonomy_level",
                        "clarification_preference",
                        "brevity_preference",
                        "initiative_preference",
                    ],
                ),
            )
        )
        await service.ingest_source(
            ContextBucketSourceCreate(
                scope="user",
                user_id="u1",
                source_key="email_1",
                structured_data={
                    "subject": "Interview invitation",
                    "from_name": "Recruiter",
                    "body_text": "We would like to invite you to interview next week.",
                },
                data_schema=ContextBucketDataSchema(
                    schema_name="email_source",
                    schema_mode="declared",
                    root_type="object",
                    primary_text_paths=["subject", "from_name", "body_text"],
                ),
            )
        )

        envelope = await service.prepare_task_envelope(
            ContextBucketAssembleRequest(
                query_text="answer the email",
                user_id="u1",
                include_user_scope=True,
                assembly_mode="drafting",
                token_budget=300,
            )
        )
        assert envelope.workflow_intent.workflow_type == "reply"
        assert envelope.workflow_intent.target_type == "email"
        assert envelope.workflow_intent.output_type == "email_reply"
        assert envelope.user_workflow_preference.clarification_preference == "low"
        assert envelope.user_workflow_preference.brevity_preference == "high"
        assert envelope.output_contract["must_include"]
        assert envelope.context_blocks

    asyncio.run(_scenario())


def test_update_workflow_preference_merges_explicit_and_approved_text(tmp_path) -> None:  # type: ignore[no-untyped-def]
    service = _service(tmp_path)

    async def _scenario() -> None:
        record = await service.update_workflow_preference(
            ContextBucketWorkflowPreferenceUpdateRequest(
                user_id="u1",
                preference_data={
                    "clarification_preference": "low",
                    "workflow_defaults": {"reply": {"draft_full_response": True}},
                },
                approved_text="Prefer concise outputs. Include clear next steps. Keep responses direct.",
            )
        )
        assert record.data_schema is not None
        assert record.data_schema.schema_name == "user_workflow_preference"
        assert record.structured_data["clarification_preference"] == "low"
        assert record.structured_data["brevity_preference"] == "high"
        assert record.structured_data["initiative_preference"] == "high"
        assert record.structured_data["style_preferences"]["directness"] == "high"
        assert record.structured_data["workflow_defaults"]["reply"]["draft_full_response"] is True
        assert record.structured_data["evidence_count"] == 1

        updated = await service.update_workflow_preference(
            ContextBucketWorkflowPreferenceUpdateRequest(
                user_id="u1",
                approved_text="Use source-backed answers when possible.",
            )
        )
        assert updated.structured_data["evidence_preference"] == "medium_high"
        assert updated.structured_data["evidence_count"] == 2

    asyncio.run(_scenario())


def test_retrieval_edge_cases(tmp_path) -> None:  # type: ignore[no-untyped-def]
    service = _service(tmp_path)

    async def _scenario() -> None:
        paraphrase = await service.ingest_source(
            ContextBucketSourceCreate(
                kind="user_profile_note",
                scope="user",
                user_id="u1",
                source_key="profile",
                text="The client prefers brief mail drafts for status updates.",
            )
        )
        assert paraphrase is not None

        duplicate_a = await service.upsert_source(
            ContextBucketSourceUpsert(
                source_key="playbook_a",
                kind="research_finding",
                scope="session",
                session_id="s1",
                text="The client wants concise weekly updates by email and fast turnaround on revisions.",
            )
        )
        duplicate_b = await service.upsert_source(
            ContextBucketSourceUpsert(
                source_key="playbook_b",
                kind="research_finding",
                scope="session",
                session_id="s1",
                text="The client wants concise weekly updates by email and fast turnaround on revisions.",
            )
        )
        assert duplicate_a is not None and duplicate_b is not None

        stale = await service.ingest_source(
            ContextBucketSourceCreate(
                kind="research_report",
                scope="session",
                session_id="s1",
                source_key="stale_report",
                text="Outdated matter update that should not be returned.",
                policy={"freshness_days": 1},
            )
        )
        assert stale is not None
        path = service.records_root / f"{stale.id}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["created_at"] = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        service._bootstrap_indexes()

        paraphrased = await service.retrieve_context(
            ContextBucketRetrieveRequest(
                query_text="write a concise email draft update",
                user_id="u1",
                include_user_scope=True,
            )
        )
        assert paraphrased.items
        assert paraphrased.items[0].provenance["source_key"] == "profile"

        assembled = await service.assemble_context(
            ContextBucketAssembleRequest(
                query_text="prepare a weekly client email update",
                session_id="s1",
                token_budget=400,
                limit=10,
            )
        )
        assert len(assembled.items) == 1

        stale_filtered = await service.retrieve_context(
            ContextBucketRetrieveRequest(
                query_text="matter update",
                session_id="s1",
            )
        )
        assert stale_filtered.excluded_counts["stale"] >= 1

    asyncio.run(_scenario())


def test_sqlite_backends_work(tmp_path) -> None:  # type: ignore[no-untyped-def]
    service = ContextBucketService(
        Settings(data_root=str(tmp_path / "bucket"), record_backend="sqlite", index_backend="sqlite")
    )

    async def _scenario() -> None:
        record = await service.ingest_source(
            ContextBucketSourceCreate(
                kind="research_finding",
                scope="session",
                session_id="s1",
                source_key="sqlite_source",
                text="SQLite-backed local persistence works.",
            )
        )
        assert record is not None
        assert service.sqlite_records_path.exists()
        assert service.sqlite_index_path.exists()

        loaded = await service.get_record(record.id)
        assert loaded is not None

        listed = await service.list_records(scope="session")
        assert listed.total == 1

    asyncio.run(_scenario())


def test_stats_prune_and_export(tmp_path) -> None:  # type: ignore[no-untyped-def]
    service = _service(tmp_path)

    async def _scenario() -> None:
        record = await service.store_record(ContextBucketRecordCreate(kind="assistant_answer", text="Old answer"))
        assert record is not None
        path = service.records_root / f"{record.id}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["created_at"] = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        pruned = await service.prune()
        assert pruned.pruned_records == 1

        await service.store_record(ContextBucketRecordCreate(kind="search_query", text="bounded rationality welfare"))
        exported_records, output_path = await service.export_training()
        assert exported_records == 1
        assert output_path.exists()

        stats = await service.stats()
        assert stats.record_count == 1

    asyncio.run(_scenario())


def test_evaluation_surface_runs_and_compares(tmp_path) -> None:  # type: ignore[no-untyped-def]
    service = _service(tmp_path)

    async def _scenario() -> None:
        await service.ingest_source(
            ContextBucketSourceCreate(
                kind="research_finding",
                scope="session",
                session_id="s1",
                source_key="matter_note",
                text="Matter ACME-17 is in discovery and needs a concise client update.",
            )
        )

        request = ContextBucketEvaluationRequest(
            cases=[
                ContextBucketEvaluationCase(
                    name="client_update",
                    query_text="draft a client update for ACME-17",
                    session_id="s1",
                    expected_source_keys=["matter_note"],
                    expected_terms=["ACME-17"],
                )
            ]
        )
        result = await service.run_evaluations(request)
        assert result.total_cases == 1
        assert result.retrieval_hits == 1
        assert result.assembly_hits == 1

        suite = await service.save_evaluation_suite(
            "smoke",
            ContextBucketEvaluationSuite(name="ignored", cases=request.cases),
        )
        assert suite.name == "smoke"

        suite_result = await service.run_evaluation_suite("smoke")
        assert suite_result.total_cases == 1

        runs = await service.list_evaluation_runs()
        assert len(runs) >= 2

        comparison = await service.compare_evaluation_runs(
            ContextBucketEvaluationCompareRequest(
                baseline_run_id=runs[-1].run_id,
                candidate_run_id=runs[0].run_id,
            )
        )
        assert comparison.baseline_run_id
        assert comparison.candidate_run_id

        gate = await service.gate_evaluation_run(
            ContextBucketEvaluationGateRequest(candidate_run_id=runs[0].run_id)
        )
        assert gate.passed is True

    asyncio.run(_scenario())


def test_evaluation_can_match_expected_terms_against_retrieved_records(tmp_path) -> None:  # type: ignore[no-untyped-def]
    service = _service(tmp_path)

    async def _scenario() -> None:
        await service.ingest_source(
            ContextBucketSourceCreate(
                kind="research_report",
                scope="session",
                session_id="s1",
                source_key="purpose_doc",
                text=(
                    "Purpose is whatever helps you contribute to the world. "
                    "Later sections discuss routines and execution."
                ),
            )
        )
        result = await service.run_evaluations(
            ContextBucketEvaluationRequest(
                cases=[
                    ContextBucketEvaluationCase(
                        name="full_doc_terms",
                        query_text="summarize purpose routines",
                        session_id="s1",
                        expected_source_keys=["purpose_doc"],
                        expected_terms=["contribute to the world"],
                        expected_terms_scope="retrieved_records",
                        token_budget=50,
                    )
                ],
                limit=1,
            )
        )

        assert result.assembly_hits == 1
        assert result.results[0].expected_terms_scope == "retrieved_records"
        assert result.results[0].matched_expected_terms == ["contribute to the world"]

    asyncio.run(_scenario())


def test_connector_and_worker_surfaces_are_removed() -> None:
    assert not hasattr(ContextBucketService, "sync_connector_items")
    assert not hasattr(ContextBucketService, "run_connector_adapter")
    assert not hasattr(ContextBucketService, "save_connector_definition")
    assert not hasattr(ContextBucketService, "enqueue_connector_run")
    assert not hasattr(ContextBucketService, "run_pending_tasks")
    assert not hasattr(context_bucket, "ContextBucketConnectorSyncRequest")
    assert not hasattr(context_bucket, "ContextBucketConnectorRunResponse")
    assert not hasattr(context_bucket, "ContextBucketWorkerTask")


def test_end_to_end_workflow_fixtures(tmp_path) -> None:  # type: ignore[no-untyped-def]
    async def _scenario() -> None:
        for index, fixture in enumerate(WORKFLOW_FIXTURES):
            service = _service(tmp_path / f"workflow_{index}")
            for record_payload in fixture["records"]:
                payload = dict(record_payload)
                data_schema = payload.get("data_schema")
                if data_schema is not None:
                    payload["data_schema"] = ContextBucketDataSchema.model_validate(data_schema)
                await service.ingest_source(ContextBucketSourceCreate(**payload))

            envelope = await service.prepare_task_envelope(
                ContextBucketAssembleRequest(
                    query_text=fixture["query_text"],
                    user_id="u1",
                    session_id="s1",
                    include_user_scope=True,
                    assembly_mode=fixture["assembly_mode"],
                    token_budget=400,
                )
            )
            assert envelope.workflow_intent.workflow_type == fixture["expected_workflow_type"]
            assert envelope.workflow_intent.output_type == fixture["expected_output_type"]
            assert envelope.workflow_intent.target_type == fixture["expected_target_type"]
            assert envelope.context_blocks

            prepared = await service.prepare_context(
                ContextBucketAssembleRequest(
                    query_text=fixture["query_text"],
                    user_id="u1",
                    session_id="s1",
                    include_user_scope=True,
                    assembly_mode=fixture["assembly_mode"],
                    token_budget=400,
                )
            )
            assert prepared.context_blocks
            assert prepared.token_estimate > 0

    asyncio.run(_scenario())
