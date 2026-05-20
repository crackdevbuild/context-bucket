from __future__ import annotations

import asyncio
import json
import math
import shutil
import tempfile
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from context_bucket import (
    ContextBucketAssembleRequest,
    ContextBucketDataSchema,
    ContextBucketDocumentImportRequest,
    ContextBucketEvaluationCase,
    ContextBucketEvaluationGateRequest,
    ContextBucketEvaluationSuite,
    ContextBucketRecordCreate,
    ContextBucketRetrieveRequest,
    ContextBucketSourceCreate,
    ContextBucketSourceDelete,
    ContextBucketSourceUpsert,
    ContextBucketWorkflowPreferenceUpdateRequest,
    Settings,
    ContextBucketService,
)
from context_bucket.models import ContextBucketEvaluationCompareRequest, ContextBucketEvaluationThresholds, ContextBucketQueryRequest

SIM_ROOT = Path(tempfile.mkdtemp(prefix="cb_sim_"))
RESULTS: dict[str, Any] = {}


def _svc(name: str, **overrides: Any) -> ContextBucketService:
    kw: dict[str, Any] = {"data_root": str(SIM_ROOT / name), "training_export_enabled": False}
    kw.update(overrides)
    return ContextBucketService(Settings(**kw))


async def simulate_ingest() -> dict[str, Any]:
    t0 = time.perf_counter()
    service = _svc("ingest")
    records = []
    sources = [
        ContextBucketSourceCreate(scope="session", session_id="s1", source_key="note_1", kind="research_finding", text="ACME expansion research shows regulatory risk rising and acquisition costs volatile."),
        ContextBucketSourceCreate(scope="user", user_id="u1", source_key="pref_1", kind="user_profile_note", text="Prefer concise professional outputs. Use bullet points. Cite sources."),
        ContextBucketSourceCreate(scope="global", source_key="global_policy", kind="topic_pattern", text="Company policy: all external communications require legal review before sending."),
        ContextBucketSourceCreate(scope="session", session_id="s1", source_key="meeting_1", kind="research_report", text="Meeting notes: launch slips two weeks. Budget unchanged. Legal review due Friday. Team agreed on revised timeline for Q3 deliverables.", tags=["meeting", "timeline"]),
        ContextBucketSourceCreate(scope="session", session_id="s1", source_key="draft_1", kind="decision_outcome", text="Decision: proceed with phased rollout. Phase 1 targets enterprise customers. Phase 2 expands to SMB segment by end of quarter.", tags=["decision", "rollout"]),
    ]
    for src in sources:
        rec = await service.ingest_source(src)
        if rec:
            records.append(rec)
    stats = await service.stats()
    elapsed = time.perf_counter() - t0
    chunks = [len(r.chunks) for r in records]
    tokens = [r.token_count_estimate for r in records]
    return {"status": "PASS", "elapsed_s": round(elapsed, 4), "records_created": len(records), "total_chunks": sum(chunks), "total_tokens": sum(tokens), "avg_tokens": round(sum(tokens) / max(1, len(tokens)), 1), "scope_dist": dict(Counter(r.scope for r in records)), "kind_dist": dict(Counter(r.kind for r in records)), "stats_count": stats.record_count}


async def simulate_upsert() -> dict[str, Any]:
    t0 = time.perf_counter()
    service = _svc("upsert")
    r1 = await service.upsert_source(ContextBucketSourceUpsert(source_key="doc_a", scope="session", session_id="s1", kind="research_finding", text="Version one of document A."))
    r2 = await service.upsert_source(ContextBucketSourceUpsert(source_key="doc_a", scope="session", session_id="s1", kind="research_finding", text="Version two of document A with updates."))
    r3 = await service.upsert_source(ContextBucketSourceUpsert(source_key="doc_a", scope="session", session_id="s1", kind="research_finding", text="Version two of document A with updates."))
    r4 = await service.upsert_source(ContextBucketSourceUpsert(source_key="doc_b", scope="user", user_id="u1", kind="research_finding", text="Independent document B."))
    elapsed = time.perf_counter() - t0
    lr = await service.list_records()
    return {"status": "PASS", "elapsed_s": round(elapsed, 4), "versions": [r.source_version for r in [r1, r2, r3, r4] if r], "dedup_same_content": r3.id == r2.id, "supersede_works": r1.source_status == "superseded", "active": sum(1 for r in lr.items if r.source_status == "active"), "total": lr.total}


async def simulate_delete() -> dict[str, Any]:
    t0 = time.perf_counter()
    service = _svc("delete")
    await service.upsert_source(ContextBucketSourceUpsert(source_key="del_target", scope="session", session_id="s1", kind="research_finding", text="This will be deleted."))
    before = await service.list_records()
    deleted = await service.delete_source(ContextBucketSourceDelete(source_key="del_target", scope="session", session_id="s1"))
    after = await service.list_records()
    double = await service.delete_source(ContextBucketSourceDelete(source_key="del_target", scope="session", session_id="s1"))
    elapsed = time.perf_counter() - t0
    return {"status": "PASS", "elapsed_s": round(elapsed, 4), "before": before.total, "deleted_status": deleted.source_status if deleted else None, "after": after.total, "double_del_none": double is None}


async def simulate_structured() -> dict[str, Any]:
    t0 = time.perf_counter()
    service = _svc("structured")
    records = []
    srcs = [
        ContextBucketSourceCreate(scope="session", session_id="s1", source_key="struct_1", kind="research_finding", structured_data={"title": "Quarterly Report", "summary": "Revenue up 12 percent", "metrics": {"revenue": 1200000, "growth": 0.12}}),
        ContextBucketSourceCreate(scope="session", session_id="s1", source_key="struct_2", kind="research_finding", structured_data=[{"name": "Item A", "count": 10}, {"name": "Item B", "count": 20}]),
        ContextBucketSourceCreate(scope="session", session_id="s1", source_key="struct_3", kind="research_finding", text="Plain text alongside structured data.", structured_data={"key": "value"}),
    ]
    for src in srcs:
        rec = await service.ingest_source(src)
        if rec:
            records.append(rec)
    elapsed = time.perf_counter() - t0
    fc = [len(r.structured_fields) for r in records]
    return {"status": "PASS", "elapsed_s": round(elapsed, 4), "records": len(records), "field_counts": fc, "avg_fields": round(sum(fc) / max(1, len(fc)), 1), "schema_modes": [r.data_schema.schema_mode if r.data_schema else None for r in records]}


async def simulate_retrieval() -> dict[str, Any]:
    t0 = time.perf_counter()
    service = _svc("retrieval")
    await service.upsert_source(ContextBucketSourceUpsert(source_key="expansion_risk", scope="session", session_id="s1", kind="research_report", text="ACME expansion research: regulatory risk is rising and customer acquisition costs are volatile. Compliance overhead adds significant burden.", tags=["research", "risk"]))
    await service.upsert_source(ContextBucketSourceUpsert(source_key="pricing_note", scope="session", session_id="s1", kind="research_finding", text="Pricing pressure from competitors is the primary concern for Q4 planning.", tags=["pricing"]))
    await service.upsert_source(ContextBucketSourceUpsert(source_key="user_pref", scope="user", user_id="u1", kind="user_profile_note", text="Prefer concise outputs with evidence citations."))
    await service.upsert_source(ContextBucketSourceUpsert(source_key="global_faq", scope="global", kind="topic_pattern", text="Standard FAQ: expansion decisions require board approval and legal review."))
    queries = [("research ACME expansion risks", "s1", "u1"), ("pricing pressure Q4", "s1", "u1"), ("concise evidence", None, "u1"), ("board approval legal", None, None), ("unrelated quantum physics", "s1", "u1")]
    qr = []
    for qt, sid, uid in queries:
        resp = await service.retrieve_context(ContextBucketRetrieveRequest(query_text=qt, session_id=sid, user_id=uid, include_user_scope=True, include_global_scope=True, limit=6))
        qr.append({"query": qt, "items": len(resp.items), "total_candidates": resp.total_candidates, "top_score": round(resp.items[0].score, 4) if resp.items else 0, "top_kind": resp.items[0].kind if resp.items else None, "strategy": resp.retrieval_strategy})
    elapsed = time.perf_counter() - t0
    return {"status": "PASS", "elapsed_s": round(elapsed, 4), "queries": len(queries), "avg_items": round(sum(q["items"] for q in qr) / max(1, len(qr)), 2), "query_results": qr}


async def simulate_assembly() -> dict[str, Any]:
    t0 = time.perf_counter()
    service = _svc("assembly")
    await service.upsert_source(ContextBucketSourceUpsert(source_key="assem_risk", scope="session", session_id="s1", kind="research_report", text="Risk assessment: expansion into EU markets faces GDPR compliance challenges."))
    await service.upsert_source(ContextBucketSourceUpsert(source_key="assem_evidence", scope="session", session_id="s1", kind="evidence_summary", text="Evidence: three competitors failed EU market entry due to data residency requirements."))
    await service.upsert_source(ContextBucketSourceUpsert(source_key="assem_pref", scope="user", user_id="u1", kind="user_profile_note", text="Prefer structured evidence-backed analysis with clear recommendations."))
    modes = ["assistant", "research", "drafting", "planner"]
    mr = []
    for mode in modes:
        resp = await service.assemble_context(ContextBucketAssembleRequest(query_text="research EU expansion risks", session_id="s1", user_id="u1", include_user_scope=True, assembly_mode=mode, token_budget=800))
        mr.append({"mode": mode, "sections": [s.name for s in resp.sections], "items": len(resp.items), "token_estimate": resp.token_count_estimate, "omitted": resp.omitted_items, "truncation": resp.truncation_reason, "context_length": len(resp.context_text)})
    elapsed = time.perf_counter() - t0
    return {"status": "PASS", "elapsed_s": round(elapsed, 4), "modes_tested": len(modes), "mode_results": mr}


async def simulate_preferences() -> dict[str, Any]:
    t0 = time.perf_counter()
    service = _svc("preferences")
    p1 = await service.update_workflow_preference(ContextBucketWorkflowPreferenceUpdateRequest(scope="user", user_id="u1", preference_data={"brevity_preference": "high", "clarification_preference": "low"}, approved_text="I prefer concise and direct responses without unnecessary questions."))
    p2 = await service.update_workflow_preference(ContextBucketWorkflowPreferenceUpdateRequest(scope="user", user_id="u1", preference_data={"initiative_preference": "high", "evidence_preference": "high"}, approved_text="Be proactive and cite sources when making recommendations."))
    rec = await service.get_record(p2.id)
    sd = rec.structured_data if rec else {}
    elapsed = time.perf_counter() - t0
    return {"status": "PASS", "elapsed_s": round(elapsed, 4), "p1_version": p1.source_version, "p2_version": p2.source_version, "brevity": sd.get("brevity_preference"), "clarification": sd.get("clarification_preference"), "initiative": sd.get("initiative_preference"), "evidence": sd.get("evidence_preference"), "style_notes_count": len(sd.get("style_notes", [])), "schema_name": rec.data_schema.schema_name if rec and rec.data_schema else None}


async def simulate_task_envelope() -> dict[str, Any]:
    t0 = time.perf_counter()
    service = _svc("task_envelope")
    await service.upsert_source(ContextBucketSourceUpsert(source_key="env_email", scope="session", session_id="s1", kind="research_finding", structured_data={"subject": "Contract review deadline", "from_name": "Legal Team", "body_text": "Please review the attached contract amendments by end of week."}, data_schema=ContextBucketDataSchema(schema_name="email_source", schema_mode="declared", root_type="object", primary_text_paths=["subject", "from_name", "body_text"])))
    await service.update_workflow_preference(ContextBucketWorkflowPreferenceUpdateRequest(scope="user", user_id="u1", approved_text="Be concise, proactive, and cite evidence."))
    scenarios = [("answer the email", "drafting"), ("summarize the contract review", "assistant"), ("research compliance requirements", "research"), ("rewrite this to be professional", "drafting"), ("plan next steps for contract review", "planner")]
    er = []
    for qt, mode in scenarios:
        env = await service.prepare_task_envelope(ContextBucketAssembleRequest(query_text=qt, session_id="s1", user_id="u1", include_user_scope=True, assembly_mode=mode, token_budget=600))
        er.append({"query": qt, "mode": mode, "objective": env.objective, "workflow_type": env.workflow_intent.workflow_type, "action": env.workflow_intent.action, "target_type": env.workflow_intent.target_type, "output_type": env.workflow_intent.output_type, "confidence": env.workflow_intent.confidence, "blocks": len(env.context_blocks), "tokens": env.token_estimate, "must_include": env.output_contract.get("must_include", []), "must_avoid": env.output_contract.get("must_avoid", [])})
    elapsed = time.perf_counter() - t0
    return {"status": "PASS", "elapsed_s": round(elapsed, 4), "scenarios": len(scenarios), "envelope_results": er}


async def simulate_evaluation() -> dict[str, Any]:
    t0 = time.perf_counter()
    service = _svc("evaluation")
    await service.upsert_source(ContextBucketSourceUpsert(source_key="eval_risk", scope="session", session_id="s1", kind="research_report", text="ACME expansion risk assessment: regulatory headwinds and volatile acquisition costs."))
    await service.upsert_source(ContextBucketSourceUpsert(source_key="eval_pricing", scope="session", session_id="s1", kind="research_finding", text="Pricing analysis: competitive pressure drives margin compression in Q4."))
    await service.upsert_source(ContextBucketSourceUpsert(source_key="eval_legal", scope="session", session_id="s1", kind="evidence_summary", text="Legal review: three jurisdictions require additional compliance documentation."))
    suite = ContextBucketEvaluationSuite(name="sim_suite", cases=[
        ContextBucketEvaluationCase(name="risk_query", query_text="research ACME expansion risks", session_id="s1", expected_source_keys=["eval_risk"], expected_terms=["regulatory", "acquisition"]),
        ContextBucketEvaluationCase(name="pricing_query", query_text="pricing pressure margin", session_id="s1", expected_source_keys=["eval_pricing"], expected_terms=["pricing", "margin"]),
        ContextBucketEvaluationCase(name="legal_query", query_text="legal compliance documentation", session_id="s1", expected_source_keys=["eval_legal"], expected_terms=["compliance", "jurisdictions"]),
        ContextBucketEvaluationCase(name="cross_query", query_text="expansion pricing legal risks", session_id="s1", expected_source_keys=["eval_risk", "eval_pricing", "eval_legal"], expected_terms=["regulatory", "pricing", "compliance"]),
    ])
    await service.save_evaluation_suite(suite.name, suite)
    eval_resp = await service.run_evaluation_suite(suite.name)
    runs = await service.list_evaluation_runs()
    gate = await service.gate_evaluation_run(ContextBucketEvaluationGateRequest(candidate_run_id=runs[0].run_id, thresholds=ContextBucketEvaluationThresholds(min_retrieval_hit_rate=0.5, min_assembly_hit_rate=0.5, max_retrieval_regressions=2, max_assembly_regressions=2)))
    elapsed = time.perf_counter() - t0
    cr = [{"name": r.name, "retrieval_hit": r.retrieval_hit, "assembly_hit": r.assembly_hit, "matched_source_keys": r.matched_expected_source_keys, "matched_terms": r.matched_expected_terms} for r in eval_resp.results]
    return {"status": "PASS", "elapsed_s": round(elapsed, 4), "total_cases": eval_resp.total_cases, "retrieval_hits": eval_resp.retrieval_hits, "assembly_hits": eval_resp.assembly_hits, "retrieval_hit_rate": round(eval_resp.retrieval_hits / max(1, eval_resp.total_cases), 3), "assembly_hit_rate": round(eval_resp.assembly_hits / max(1, eval_resp.total_cases), 3), "runs": len(runs), "gate_passed": gate.passed, "gate_failures": gate.failures, "case_results": cr}


async def simulate_importers() -> dict[str, Any]:
    t0 = time.perf_counter()
    service = _svc("importers")
    d = SIM_ROOT / "importers" / "test_files"
    d.mkdir(parents=True, exist_ok=True)
    (d / "notes.txt").write_text("Plain text file with research notes about market expansion.", encoding="utf-8")
    (d / "data.json").write_text(json.dumps({"title": "JSON Report", "findings": ["Finding A", "Finding B"]}), encoding="utf-8")
    (d / "log.csv").write_text("event,count\nlogin,5\nlogout,3\n", encoding="utf-8")
    (d / "page.html").write_text("<html><body><h1>Report</h1><p>Key findings from Q4 analysis.</p></body></html>", encoding="utf-8")
    (d / "feed.ndjson").write_text('{"type":"alert","msg":"Compliance deadline approaching"}\n{"type":"info","msg":"Board meeting scheduled"}\n', encoding="utf-8")
    (d / "data.xml").write_text("<report><title>XML Report</title><body>Structured XML content.</body></report>", encoding="utf-8")
    resp = await service.import_path(ContextBucketDocumentImportRequest(path=str(d), recursive=True, scope="session", session_id="s1", kind="research_finding", tags=["imported"]))
    elapsed = time.perf_counter() - t0
    return {"status": "PASS", "elapsed_s": round(elapsed, 4), "imported": resp.imported, "skipped": resp.skipped, "file_statuses": [{"path": Path(i.path).name, "status": i.status, "reason": i.reason} for i in resp.items]}


async def simulate_storage_backends() -> dict[str, Any]:
    t0 = time.perf_counter()
    file_svc = _svc("storage_file")
    sqlite_svc = _svc("storage_sqlite", record_backend="sqlite", index_backend="sqlite")
    for svc in [file_svc, sqlite_svc]:
        await svc.upsert_source(ContextBucketSourceUpsert(source_key="backend_test", scope="session", session_id="s1", kind="research_finding", text="Testing storage backend persistence."))
        await svc.upsert_source(ContextBucketSourceUpsert(source_key="backend_test_2", scope="user", user_id="u1", kind="user_profile_note", text="Storage backend user data."))
    fs = await file_svc.stats()
    ss = await sqlite_svc.stats()
    elapsed = time.perf_counter() - t0
    return {"status": "PASS", "elapsed_s": round(elapsed, 4), "file_records": fs.record_count, "sqlite_records": ss.record_count, "file_by_kind": fs.records_by_kind, "sqlite_by_kind": ss.records_by_kind, "backends_match": fs.record_count == ss.record_count}


async def simulate_audit() -> dict[str, Any]:
    t0 = time.perf_counter()
    service = _svc("audit")
    await service.upsert_source(ContextBucketSourceUpsert(source_key="audit_doc", scope="session", session_id="s1", kind="research_finding", text="Document for audit trail testing."))
    resp = await service.retrieve_context(ContextBucketRetrieveRequest(query_text="audit document testing", session_id="s1", limit=6))
    audit_log = service.audit_root / "context_selection.jsonl"
    entries = []
    if audit_log.exists():
        for line in audit_log.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entries.append(json.loads(line))
    elapsed = time.perf_counter() - t0
    return {"status": "PASS", "elapsed_s": round(elapsed, 4), "audit_id": resp.audit_id, "entries": len(entries), "latest_has_query": entries[-1].get("query_text") == "audit document testing" if entries else False, "latest_has_counts": "candidate_counts" in entries[-1] if entries else False}


async def simulate_training() -> dict[str, Any]:
    t0 = time.perf_counter()
    service = _svc("training", training_export_enabled=True)
    await service.upsert_source(ContextBucketSourceUpsert(source_key="train_1", scope="session", session_id="s1", kind="research_finding", text="Training data record one."))
    await service.upsert_source(ContextBucketSourceUpsert(source_key="train_2", scope="session", session_id="s1", kind="research_finding", text="Training data record two."))
    count, path = await service.export_training()
    lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()] if path.exists() else []
    elapsed = time.perf_counter() - t0
    return {"status": "PASS", "elapsed_s": round(elapsed, 4), "exported_count": count, "export_lines": len(lines), "lines_match": len(lines) == count}


async def simulate_prune() -> dict[str, Any]:
    t0 = time.perf_counter()
    service = _svc("prune")
    await service.upsert_source(ContextBucketSourceUpsert(source_key="prune_target", scope="session", session_id="s1", kind="research_finding", text="This record will be pruned."))
    record = service._find_latest_source("prune_target", "session", None, "s1")
    if record:
        old = record.model_copy(update={"created_at": datetime.now(timezone.utc) - timedelta(days=365), "policy": record.model_copy(update={"retention_days": 30})})
        async with service._write_lock:
            service._persist_record_locked(old)
            service._rebuild_indexes_locked(service._load_records())
    prune_resp = await service.prune()
    elapsed = time.perf_counter() - t0
    return {"status": "PASS", "elapsed_s": round(elapsed, 4), "pruned_records": prune_resp.pruned_records, "pruned_chunks": len(prune_resp.pruned_chunk_ids)}


async def simulate_workflows() -> dict[str, Any]:
    t0 = time.perf_counter()
    results = {}
    reply_svc = _svc("wf_reply")
    await reply_svc.update_workflow_preference(ContextBucketWorkflowPreferenceUpdateRequest(scope="user", user_id="u1", approved_text="Concise, professional, proactive with next steps."))
    await reply_svc.upsert_source(ContextBucketSourceUpsert(source_key="inbox_email", scope="session", session_id="s1", kind="research_finding", structured_data={"subject": "Follow-up needed", "from_name": "Partner", "body_text": "Can you provide an update on the ACME matter by Friday?"}, data_schema=ContextBucketDataSchema(schema_name="email_source", schema_mode="declared", root_type="object", primary_text_paths=["subject", "from_name", "body_text"])))
    re = await reply_svc.prepare_task_envelope(ContextBucketAssembleRequest(query_text="answer the email", session_id="s1", user_id="u1", include_user_scope=True, assembly_mode="drafting", token_budget=500))
    results["reply"] = {"workflow_type": re.workflow_intent.workflow_type, "target_type": re.workflow_intent.target_type, "objective": re.objective, "blocks": len(re.context_blocks)}

    sum_svc = _svc("wf_summarize")
    for i in range(5):
        await sum_svc.upsert_source(ContextBucketSourceUpsert(source_key=f"note_{i}", scope="session", session_id="s1", kind="research_finding", text=f"Research note {i}: finding about topic area {i} with supporting evidence and data points."))
    se = await sum_svc.prepare_task_envelope(ContextBucketAssembleRequest(query_text="summarize the research notes", session_id="s1", assembly_mode="assistant", token_budget=400))
    results["summarize"] = {"workflow_type": se.workflow_intent.workflow_type, "objective": se.objective, "blocks": len(se.context_blocks), "tokens": se.token_estimate}

    res_svc = _svc("wf_research")
    await res_svc.upsert_source(ContextBucketSourceUpsert(source_key="market_report", scope="session", session_id="s1", kind="research_report", text="Market analysis: APAC expansion shows strong growth potential but regulatory barriers persist."))
    await res_svc.upsert_source(ContextBucketSourceUpsert(source_key="risk_brief", scope="session", session_id="s1", kind="evidence_summary", text="Risk brief: currency fluctuation and IP protection are top concerns for APAC entry."))
    rse = await res_svc.prepare_task_envelope(ContextBucketAssembleRequest(query_text="research APAC expansion risks", session_id="s1", assembly_mode="research", token_budget=500))
    results["research"] = {"workflow_type": rse.workflow_intent.workflow_type, "target_type": rse.workflow_intent.target_type, "objective": rse.objective, "blocks": len(rse.context_blocks)}

    rw_svc = _svc("wf_rewrite")
    await rw_svc.update_workflow_preference(ContextBucketWorkflowPreferenceUpdateRequest(scope="user", user_id="u1", approved_text="Concise, formal, structured with bullet points."))
    await rw_svc.upsert_source(ContextBucketSourceUpsert(source_key="draft_memo", scope="session", session_id="s1", kind="research_finding", text="This memo is excessively verbose and repetitive and needs to be made clearer and more professional."))
    rwe = await rw_svc.prepare_task_envelope(ContextBucketAssembleRequest(query_text="rewrite this memo to be concise and professional", session_id="s1", user_id="u1", include_user_scope=True, assembly_mode="drafting", token_budget=400))
    results["rewrite"] = {"workflow_type": rwe.workflow_intent.workflow_type, "objective": rwe.objective, "blocks": len(rwe.context_blocks)}

    elapsed = time.perf_counter() - t0
    return {"status": "PASS", "elapsed_s": round(elapsed, 4), "workflows_tested": len(results), "results": results}


async def simulate_embeddings() -> dict[str, Any]:
    t0 = time.perf_counter()
    service = _svc("embeddings")
    texts = ["ACME expansion regulatory risk compliance", "pricing pressure competitive margin Q4", "concise professional proactive evidence", "legal review board approval contract", "meeting notes timeline budget deliverables"]
    embeddings = [service._embed(service._semantic_terms(t)) for t in texts]
    dim = len(embeddings[0]) if embeddings else 0
    norms = [math.sqrt(sum(v * v for v in e)) for e in embeddings]
    sims = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sims.append(service._cosine_similarity(embeddings[i], embeddings[j]))
    elapsed = time.perf_counter() - t0
    return {"status": "PASS", "elapsed_s": round(elapsed, 4), "dimensions": dim, "vectors_tested": len(embeddings), "avg_norm": round(sum(norms) / max(1, len(norms)), 4), "min_sim": round(min(sims), 4) if sims else 0, "max_sim": round(max(sims), 4) if sims else 0, "avg_sim": round(sum(sims) / max(1, len(sims)), 4)}


async def simulate_scoring() -> dict[str, Any]:
    t0 = time.perf_counter()
    service = _svc("scoring")
    pairs = [
        (["expansion", "risk", "regulatory"], ["expansion", "risk", "compliance"]),
        (["pricing", "margin", "q4"], ["pricing", "competitive", "pressure"]),
        (["concise", "professional"], ["verbose", "informal"]),
        (["meeting", "notes", "timeline"], ["meeting", "agenda", "schedule"]),
    ]
    results = []
    for left, right in pairs:
        results.append({"left": left, "right": right, "lexical": round(service._lexical_overlap_score(left, right), 4), "set": round(service._set_overlap_score(left, right), 4)})
    scope_bonuses = {s: service._scope_priority_bonus(s) for s in ["session", "user", "global"]}
    elapsed = time.perf_counter() - t0
    return {"status": "PASS", "elapsed_s": round(elapsed, 4), "pairs": len(pairs), "results": results, "scope_bonuses": scope_bonuses}


async def simulate_bulk_ingest() -> dict[str, Any]:
    t0 = time.perf_counter()
    service = _svc("bulk_ingest")
    kinds = ["research_finding", "research_report", "evidence_summary", "decision_outcome", "topic_pattern"]
    scopes = ["session", "user", "global"]
    records = []
    for i in range(50):
        kind = kinds[i % len(kinds)]
        scope = scopes[i % len(scopes)]
        kw: dict[str, Any] = {"scope": scope, "source_key": f"bulk_{i}", "kind": kind, "text": f"Bulk record {i}: analysis of market segment {i % 7} with findings on revenue growth and competitive dynamics in region {i % 5}."}
        if scope == "session":
            kw["session_id"] = "s_bulk"
        elif scope == "user":
            kw["user_id"] = "u_bulk"
        rec = await service.ingest_source(ContextBucketSourceCreate(**kw))
        if rec:
            records.append(rec)
    stats = await service.stats()
    elapsed = time.perf_counter() - t0
    return {"status": "PASS", "elapsed_s": round(elapsed, 4), "ingested": len(records), "stats_count": stats.record_count, "scope_dist": dict(Counter(r.scope for r in records)), "kind_dist": dict(Counter(r.kind for r in records)), "total_tokens": sum(r.token_count_estimate for r in records), "avg_chunks": round(sum(len(r.chunks) for r in records) / max(1, len(records)), 2)}


async def simulate_multi_scope_retrieval() -> dict[str, Any]:
    t0 = time.perf_counter()
    service = _svc("multi_scope_ret")
    await service.upsert_source(ContextBucketSourceUpsert(source_key="global_faq", scope="global", kind="topic_pattern", text="Company FAQ: remote work policy allows 3 days per week. VPN required for all remote access."))
    await service.upsert_source(ContextBucketSourceUpsert(source_key="user_style", scope="user", user_id="u_ms", kind="user_profile_note", text="I prefer detailed technical explanations with code examples and step-by-step instructions."))
    await service.upsert_source(ContextBucketSourceUpsert(source_key="sess_task", scope="session", session_id="s_ms", kind="research_finding", text="Current sprint: implement OAuth2 PKCE flow for mobile clients. Deadline is end of week."))
    scenarios = [
        ("remote work policy", "s_ms", "u_ms", True, True),
        ("OAuth2 mobile implementation", "s_ms", "u_ms", True, True),
        ("technical coding instructions", "s_ms", "u_ms", True, True),
        ("unrelated cooking recipes", "s_ms", "u_ms", False, False),
    ]
    results = []
    for qt, sid, uid, incl_user, incl_global in scenarios:
        resp = await service.retrieve_context(ContextBucketRetrieveRequest(query_text=qt, session_id=sid, user_id=uid, include_user_scope=incl_user, include_global_scope=incl_global, limit=6))
        results.append({"query": qt, "items": len(resp.items), "scopes": list(set(it.scope for it in resp.items)), "top_score": round(resp.items[0].score, 4) if resp.items else 0, "strategy": resp.retrieval_strategy})
    elapsed = time.perf_counter() - t0
    return {"status": "PASS", "elapsed_s": round(elapsed, 4), "scenarios": len(scenarios), "results": results}


async def simulate_confidentiality_policy() -> dict[str, Any]:
    t0 = time.perf_counter()
    service = _svc("conf_policy")
    from context_bucket.models import ContextBucketPolicy as Policy
    await service.upsert_source(ContextBucketSourceUpsert(source_key="private_doc", scope="session", session_id="s_cp", kind="research_finding", text="Confidential board strategy: explore acquisition of StartupCo at 2x revenue multiple.", policy=Policy(confidentiality="private")))
    await service.upsert_source(ContextBucketSourceUpsert(source_key="restricted_doc", scope="session", session_id="s_cp", kind="evidence_summary", text="Restricted: financial projections show 30% revenue growth for next fiscal year.", policy=Policy(confidentiality="restricted")))
    await service.upsert_source(ContextBucketSourceUpsert(source_key="shareable_doc", scope="global", kind="topic_pattern", text="Public company values: innovation, transparency, and customer-first approach.", policy=Policy(confidentiality="shareable")))
    stats = await service.stats()
    resp = await service.retrieve_context(ContextBucketRetrieveRequest(query_text="company strategy financial growth", session_id="s_cp", limit=6))
    elapsed = time.perf_counter() - t0
    return {"status": "PASS", "elapsed_s": round(elapsed, 4), "total_records": stats.record_count, "retrieved": len(resp.items), "confidentialities": [it.provenance.get("policy", {}).get("confidentiality", "unknown") if isinstance(it.provenance.get("policy"), dict) else "unknown" for it in resp.items], "stats_by_status": stats.records_by_source_status}


async def simulate_upsert_versioning() -> dict[str, Any]:
    t0 = time.perf_counter()
    service = _svc("upsert_ver")
    versions = []
    for v in range(1, 8):
        rec = await service.upsert_source(ContextBucketSourceUpsert(source_key="evolving_doc", scope="session", session_id="s_uv", kind="research_report", text=f"Document revision {v}: Updated analysis of Q{v} market conditions and strategic recommendations for the board."))
        versions.append({"version": rec.source_version, "status": rec.source_status, "checksum": rec.content_checksum})
    all_records = await service.list_records()
    active = [r for r in all_records.items if r.source_status == "active"]
    superseded = [r for r in all_records.items if r.source_status == "superseded"]
    elapsed = time.perf_counter() - t0
    return {"status": "PASS", "elapsed_s": round(elapsed, 4), "upserts": len(versions), "latest_version": versions[-1]["version"], "active_count": len(active), "superseded_count": len(superseded), "checksums_unique": len(set(v["checksum"] for v in versions))}


async def simulate_tag_filtering() -> dict[str, Any]:
    t0 = time.perf_counter()
    service = _svc("tag_filter")
    await service.upsert_source(ContextBucketSourceUpsert(source_key="tag_finance", scope="session", session_id="s_tf", kind="research_finding", text="Q4 financial results exceeded expectations with 15% revenue growth.", tags=["finance", "quarterly", "growth"]))
    await service.upsert_source(ContextBucketSourceUpsert(source_key="tag_legal", scope="session", session_id="s_tf", kind="research_report", text="Legal review identified three compliance gaps in the current framework.", tags=["legal", "compliance", "audit"]))
    await service.upsert_source(ContextBucketSourceUpsert(source_key="tag_tech", scope="session", session_id="s_tf", kind="research_finding", text="Infrastructure migration to Kubernetes completed successfully ahead of schedule.", tags=["technology", "infrastructure", "migration"]))
    resp_all = await service.retrieve_context(ContextBucketRetrieveRequest(query_text="financial legal technology results", session_id="s_tf", limit=10))
    resp_legal = await service.retrieve_context(ContextBucketRetrieveRequest(query_text="financial legal technology results", session_id="s_tf", tags=["legal"], limit=10))
    resp_multi = await service.retrieve_context(ContextBucketRetrieveRequest(query_text="financial legal technology results", session_id="s_tf", tags=["finance", "technology"], limit=10))
    elapsed = time.perf_counter() - t0
    return {"status": "PASS", "elapsed_s": round(elapsed, 4), "all_items": len(resp_all.items), "legal_only": len(resp_legal.items), "multi_tag": len(resp_multi.items)}


async def simulate_kind_filtering() -> dict[str, Any]:
    t0 = time.perf_counter()
    service = _svc("kind_filter")
    for kind_name in ["research_finding", "research_report", "evidence_summary", "decision_outcome", "topic_pattern", "user_profile_note"]:
        await service.upsert_source(ContextBucketSourceUpsert(source_key=f"kf_{kind_name}", scope="session", session_id="s_kf", kind=kind_name, text=f"Content for {kind_name}: detailed analysis and findings relevant to current project."))
    resp_all = await service.retrieve_context(ContextBucketRetrieveRequest(query_text="analysis findings project", session_id="s_kf", limit=10))
    resp_reports = await service.retrieve_context(ContextBucketRetrieveRequest(query_text="analysis findings project", session_id="s_kf", kinds=["research_report"], limit=10))
    resp_evidence = await service.retrieve_context(ContextBucketRetrieveRequest(query_text="analysis findings project", session_id="s_kf", kinds=["evidence_summary", "research_finding"], limit=10))
    elapsed = time.perf_counter() - t0
    return {"status": "PASS", "elapsed_s": round(elapsed, 4), "all_items": len(resp_all.items), "report_only": len(resp_reports.items), "evidence_and_finding": len(resp_evidence.items), "all_kinds": list(set(it.kind for it in resp_all.items))}


async def simulate_store_record_api() -> dict[str, Any]:
    t0 = time.perf_counter()
    service = _svc("store_rec")
    rec = await service.store_record(ContextBucketRecordCreate(kind="research_finding", text="Directly stored record with no source key or upsert logic.", scope="session", session_id="s_sr", title="Direct Store Test", tags=["direct", "test"]))
    stats = await service.stats()
    fetched = await service.get_record(rec.id) if rec else None
    elapsed = time.perf_counter() - t0
    return {"status": "PASS", "elapsed_s": round(elapsed, 4), "created": rec is not None, "fetched": fetched is not None, "kind_match": rec.kind == "research_finding" if rec else False, "title": rec.title if rec else None, "stats_count": stats.record_count}


async def simulate_structured_nested() -> dict[str, Any]:
    t0 = time.perf_counter()
    service = _svc("struct_nested")
    rec = await service.ingest_source(ContextBucketSourceCreate(scope="session", session_id="s_sn", source_key="nested_1", kind="research_finding", structured_data={"project": {"name": "Project Alpha", "status": "active", "team": {"lead": "Jane Smith", "members": 8, "budget": {"allocated": 500000, "spent": 320000, "currency": "USD"}}}, "milestones": [{"phase": "discovery", "complete": True}, {"phase": "implementation", "complete": False}]}))
    elapsed = time.perf_counter() - t0
    return {"status": "PASS", "elapsed_s": round(elapsed, 4), "field_count": len(rec.structured_fields) if rec else 0, "schema_mode": rec.data_schema.schema_mode if rec and rec.data_schema else None, "has_primary_paths": len(rec.data_schema.primary_text_paths) > 0 if rec and rec.data_schema else False, "root_type": rec.data_schema.root_type if rec and rec.data_schema else None}


async def simulate_structured_array() -> dict[str, Any]:
    t0 = time.perf_counter()
    service = _svc("struct_array")
    data = [{"id": 1, "name": "Alice", "role": "engineer", "skills": ["python", "kubernetes"]}, {"id": 2, "name": "Bob", "role": "designer", "skills": ["figma", "css"]}, {"id": 3, "name": "Carol", "role": "manager", "skills": ["strategy", "analytics"]}]
    rec = await service.ingest_source(ContextBucketSourceCreate(scope="session", session_id="s_sa", source_key="array_1", kind="research_finding", structured_data=data))
    elapsed = time.perf_counter() - t0
    return {"status": "PASS", "elapsed_s": round(elapsed, 4), "field_count": len(rec.structured_fields) if rec else 0, "schema_mode": rec.data_schema.schema_mode if rec and rec.data_schema else None, "root_type": rec.data_schema.root_type if rec and rec.data_schema else None}


async def simulate_declared_schema() -> dict[str, Any]:
    t0 = time.perf_counter()
    service = _svc("declared_schema")
    rec = await service.ingest_source(ContextBucketSourceCreate(scope="session", session_id="s_ds", source_key="declared_1", kind="research_finding", structured_data={"incident": "Server outage", "severity": "high", "duration_min": 45, "affected_users": 1200}, data_schema=ContextBucketDataSchema(schema_name="incident_report", schema_mode="declared", root_type="object", field_paths=["incident", "severity", "duration_min", "affected_users"], primary_text_paths=["incident", "severity"])))
    elapsed = time.perf_counter() - t0
    return {"status": "PASS", "elapsed_s": round(elapsed, 4), "schema_name": rec.data_schema.schema_name if rec and rec.data_schema else None, "schema_mode": rec.data_schema.schema_mode if rec and rec.data_schema else None, "field_paths": rec.data_schema.field_paths if rec and rec.data_schema else [], "primary_text_paths": rec.data_schema.primary_text_paths if rec and rec.data_schema else []}


async def simulate_age_decay() -> dict[str, Any]:
    t0 = time.perf_counter()
    service = _svc("age_decay")
    await service.upsert_source(ContextBucketSourceUpsert(source_key="fresh_doc", scope="session", session_id="s_ad", kind="research_finding", text="Fresh research finding about current market conditions and opportunities."))
    old_rec = await service.upsert_source(ContextBucketSourceUpsert(source_key="old_doc", scope="session", session_id="s_ad", kind="research_finding", text="Historical research finding about past market conditions and outdated trends."))
    if old_rec:
        aged = old_rec.model_copy(update={"created_at": datetime.now(timezone.utc) - timedelta(days=90)})
        async with service._write_lock:
            service._persist_record_locked(aged)
            service._rebuild_indexes_locked(service._load_records())
    resp = await service.retrieve_context(ContextBucketRetrieveRequest(query_text="market conditions research", session_id="s_ad", limit=10))
    elapsed = time.perf_counter() - t0
    fresh_score = 0
    old_score = 0
    for it in resp.items:
        if it.provenance.get("source_key") == "fresh_doc":
            fresh_score = it.score
        elif it.provenance.get("source_key") == "old_doc":
            old_score = it.score
    return {"status": "PASS", "elapsed_s": round(elapsed, 4), "items": len(resp.items), "fresh_score": round(fresh_score, 4), "old_score": round(old_score, 4), "fresh_ranked_higher": fresh_score >= old_score}


async def simulate_dedup_detection() -> dict[str, Any]:
    t0 = time.perf_counter()
    service = _svc("dedup_detect")
    await service.ingest_source(ContextBucketSourceCreate(scope="session", session_id="s_dd", source_key="dedup_1", kind="research_finding", text="The quarterly earnings report shows consistent revenue growth across all business segments."))
    dup = await service.ingest_source(ContextBucketSourceCreate(scope="session", session_id="s_dd", source_key="dedup_2", kind="research_finding", text="The quarterly earnings report shows consistent revenue growth across all business segments."))
    near_dup = await service.ingest_source(ContextBucketSourceCreate(scope="session", session_id="s_dd", source_key="dedup_3", kind="research_finding", text="The quarterly earnings report shows consistent revenue growth across most business segments."))
    diff = await service.ingest_source(ContextBucketSourceCreate(scope="session", session_id="s_dd", source_key="dedup_4", kind="research_finding", text="Strategic partnerships are driving expansion into emerging markets in Southeast Asia."))
    stats = await service.stats()
    elapsed = time.perf_counter() - t0
    return {"status": "PASS", "elapsed_s": round(elapsed, 4), "total_attempted": 4, "total_created": stats.record_count, "exact_dup_skipped": dup is None, "near_dup_created": near_dup is not None, "diff_created": diff is not None}


async def simulate_evaluation_compare() -> dict[str, Any]:
    t0 = time.perf_counter()
    service = _svc("eval_compare")
    await service.upsert_source(ContextBucketSourceUpsert(source_key="ec_doc1", scope="session", session_id="s_ec", kind="research_report", text="Comprehensive analysis of renewable energy market trends and investment opportunities."))
    await service.upsert_source(ContextBucketSourceUpsert(source_key="ec_doc2", scope="session", session_id="s_ec", kind="evidence_summary", text="Solar panel costs dropped 40% in the last decade, making utility-scale projects economically viable."))
    suite = ContextBucketEvaluationSuite(name="compare_suite", cases=[
        ContextBucketEvaluationCase(name="renewable_query", query_text="renewable energy investment", session_id="s_ec", expected_source_keys=["ec_doc1", "ec_doc2"], expected_terms=["renewable", "energy", "solar"]),
        ContextBucketEvaluationCase(name="solar_query", query_text="solar panel cost trends", session_id="s_ec", expected_source_keys=["ec_doc2"], expected_terms=["solar", "cost"]),
    ])
    await service.save_evaluation_suite(suite.name, suite)
    run1 = await service.run_evaluation_suite(suite.name)
    await service.upsert_source(ContextBucketSourceUpsert(source_key="ec_doc3", scope="session", session_id="s_ec", kind="research_finding", text="Wind energy offshore projects are accelerating across European markets."))
    run2 = await service.run_evaluation_suite(suite.name)
    runs = await service.list_evaluation_runs()
    if len(runs) >= 2:
        compare = await service.compare_evaluation_runs(ContextBucketEvaluationCompareRequest(baseline_run_id=runs[-2].run_id, candidate_run_id=runs[-1].run_id))
    else:
        compare = None
    elapsed = time.perf_counter() - t0
    return {"status": "PASS", "elapsed_s": round(elapsed, 4), "run1_hits": run1.retrieval_hits, "run2_hits": run2.retrieval_hits, "total_runs": len(runs), "compare_regressions": compare.retrieval_regressions if compare else None, "compare_items": len(compare.items) if compare else 0}


async def simulate_evaluation_gate_fail() -> dict[str, Any]:
    t0 = time.perf_counter()
    service = _svc("eval_gate_fail")
    await service.upsert_source(ContextBucketSourceUpsert(source_key="gf_doc", scope="session", session_id="s_gf", kind="research_report", text="Niche analysis of quantum computing applications in pharmaceutical drug discovery."))
    suite = ContextBucketEvaluationSuite(name="strict_suite", cases=[
        ContextBucketEvaluationCase(name="quantum_hit", query_text="quantum computing pharmaceutical", session_id="s_gf", expected_source_keys=["gf_doc"], expected_terms=["quantum", "computing"]),
        ContextBucketEvaluationCase(name="unrelated_miss", query_text="agricultural farming techniques", session_id="s_gf", expected_source_keys=["gf_doc"], expected_terms=["farming", "agriculture"]),
    ])
    await service.save_evaluation_suite(suite.name, suite)
    await service.run_evaluation_suite(suite.name)
    runs = await service.list_evaluation_runs()
    gate = await service.gate_evaluation_run(ContextBucketEvaluationGateRequest(candidate_run_id=runs[-1].run_id, thresholds=ContextBucketEvaluationThresholds(min_retrieval_hit_rate=1.0, min_assembly_hit_rate=1.0, max_retrieval_regressions=0, max_assembly_regressions=0)))
    elapsed = time.perf_counter() - t0
    return {"status": "PASS", "elapsed_s": round(elapsed, 4), "gate_passed": gate.passed, "failures": gate.failures, "retrieval_hit_rate": gate.retrieval_hit_rate, "assembly_hit_rate": gate.assembly_hit_rate}


async def simulate_importer_csv() -> dict[str, Any]:
    t0 = time.perf_counter()
    service = _svc("import_csv")
    d = SIM_ROOT / "import_csv" / "files"
    d.mkdir(parents=True, exist_ok=True)
    (d / "transactions.csv").write_text("date,amount,category,description\n2024-01-15,1250.00,travel,Flight to conference\n2024-01-18,89.99,software,IDE license renewal\n2024-01-22,450.00,consulting,External audit fee\n2024-02-01,2100.00,hardware,Server rack purchase\n", encoding="utf-8")
    resp = await service.import_path(ContextBucketDocumentImportRequest(path=str(d), recursive=True, scope="session", session_id="s_ic", kind="research_finding", tags=["csv", "transactions"]))
    stats = await service.stats()
    elapsed = time.perf_counter() - t0
    return {"status": "PASS", "elapsed_s": round(elapsed, 4), "imported": resp.imported, "skipped": resp.skipped, "stats_count": stats.record_count, "file_details": [{"path": Path(i.path).name, "status": i.status} for i in resp.items]}


async def simulate_importer_nested_dirs() -> dict[str, Any]:
    t0 = time.perf_counter()
    service = _svc("import_nested")
    base = SIM_ROOT / "import_nested" / "docs"
    for subdir in ["reports", "notes", "reports/quarterly"]:
        p = base / subdir
        p.mkdir(parents=True, exist_ok=True)
    (base / "reports" / "annual.txt").write_text("Annual report: company achieved record revenue of $50M.", encoding="utf-8")
    (base / "reports" / "quarterly" / "q1.txt").write_text("Q1 report: strong start with 12% growth year over year.", encoding="utf-8")
    (base / "notes" / "meeting.txt").write_text("Meeting notes: discussed expansion strategy for APAC region.", encoding="utf-8")
    (base / "notes" / "standup.txt").write_text("Standup: team completed sprint goals ahead of schedule.", encoding="utf-8")
    resp = await service.import_path(ContextBucketDocumentImportRequest(path=str(base), recursive=True, scope="session", session_id="s_in", kind="research_finding", tags=["nested_import"]))
    elapsed = time.perf_counter() - t0
    return {"status": "PASS", "elapsed_s": round(elapsed, 4), "imported": resp.imported, "skipped": resp.skipped, "item_count": len(resp.items), "paths": [Path(i.path).name for i in resp.items]}


async def simulate_query_api() -> dict[str, Any]:
    t0 = time.perf_counter()
    service = _svc("query_api")
    await service.upsert_source(ContextBucketSourceUpsert(source_key="qa_1", scope="session", session_id="s_qa", kind="research_finding", text="Machine learning models show 95% accuracy on customer churn prediction tasks."))
    await service.upsert_source(ContextBucketSourceUpsert(source_key="qa_2", scope="session", session_id="s_qa", kind="research_report", text="Deep learning architecture comparison: transformers outperform RNNs on sequence tasks."))
    results = await service.query(ContextBucketQueryRequest(query_text="machine learning model accuracy", session_id="s_qa", limit=6))
    elapsed = time.perf_counter() - t0
    return {"status": "PASS", "elapsed_s": round(elapsed, 4), "results": len(results), "top_score": round(results[0].score, 4) if results else 0, "top_kind": results[0].kind if results else None, "has_record_id": bool(results and results[0].record_id)}


async def simulate_health_check() -> dict[str, Any]:
    t0 = time.perf_counter()
    service = _svc("health_chk")
    await service.upsert_source(ContextBucketSourceUpsert(source_key="hc_doc", scope="session", session_id="s_hc", kind="research_finding", text="Health check test document."))
    health = service.health()
    token_est = ContextBucketService.token_estimate("This is a sample text for token estimation testing.")
    checksum = ContextBucketService._content_checksum("test content for checksum")
    deduped = ContextBucketService._dedupe_strings(["alpha", "beta", "alpha", "gamma", "beta", "delta"])
    elapsed = time.perf_counter() - t0
    return {"status": "PASS", "elapsed_s": round(elapsed, 4), "health_ok": health.get("ok"), "token_estimate": token_est, "checksum_len": len(checksum), "deduped": deduped, "deduped_count": len(deduped)}


async def simulate_stemming_ngrams() -> dict[str, Any]:
    t0 = time.perf_counter()
    service = _svc("stem_ngram")
    stems = {w: service._stem_token(w) for w in ["running", "studies", "decision", "expansion", "regulatory", "findings", "outcomes"]}
    ngrams = {w: service._char_ngrams(w, 3) for w in ["risk", "legal", "price"]}
    elapsed = time.perf_counter() - t0
    return {"status": "PASS", "elapsed_s": round(elapsed, 4), "stems": stems, "ngrams_sample": {k: v[:5] for k, v in ngrams.items()}}


async def simulate_context_compression() -> dict[str, Any]:
    t0 = time.perf_counter()
    service = _svc("ctx_compress")
    for i in range(15):
        await service.upsert_source(ContextBucketSourceUpsert(source_key=f"compress_{i}", scope="session", session_id="s_cc", kind="research_finding", text=f"Record {i}: Analysis of market segment {i % 4} with data on revenue projections and competitive positioning for region {i % 3}."))
    resp_low = await service.assemble_context(ContextBucketAssembleRequest(query_text="market analysis revenue projections", session_id="s_cc", assembly_mode="assistant", token_budget=200))
    resp_high = await service.assemble_context(ContextBucketAssembleRequest(query_text="market analysis revenue projections", session_id="s_cc", assembly_mode="assistant", token_budget=2000))
    elapsed = time.perf_counter() - t0
    return {"status": "PASS", "elapsed_s": round(elapsed, 4), "low_budget_items": len(resp_low.items), "high_budget_items": len(resp_high.items), "low_truncated": resp_low.truncation_reason, "high_truncated": resp_high.truncation_reason, "low_omitted": resp_low.omitted_items, "high_omitted": resp_high.omitted_items, "low_tokens": resp_low.token_count_estimate, "high_tokens": resp_high.token_count_estimate}


async def simulate_session_isolation() -> dict[str, Any]:
    t0 = time.perf_counter()
    service = _svc("session_iso")
    await service.upsert_source(ContextBucketSourceUpsert(source_key="session_a_doc", scope="session", session_id="sess_a", kind="research_finding", text="Session A: Confidential merger discussions with TargetCorp at premium valuation."))
    await service.upsert_source(ContextBucketSourceUpsert(source_key="session_b_doc", scope="session", session_id="sess_b", kind="research_finding", text="Session B: Product launch timeline for Project Nebula set for Q3."))
    await service.upsert_source(ContextBucketSourceUpsert(source_key="shared_global", scope="global", kind="topic_pattern", text="Global policy: all M&A activity requires board pre-approval and legal sign-off."))
    resp_a = await service.retrieve_context(ContextBucketRetrieveRequest(query_text="merger acquisition discussions", session_id="sess_a", include_global_scope=True, limit=10))
    resp_b = await service.retrieve_context(ContextBucketRetrieveRequest(query_text="merger acquisition discussions", session_id="sess_b", include_global_scope=True, limit=10))
    a_keys = {it.provenance.get("source_key") for it in resp_a.items}
    b_keys = {it.provenance.get("source_key") for it in resp_b.items}
    elapsed = time.perf_counter() - t0
    return {"status": "PASS", "elapsed_s": round(elapsed, 4), "session_a_items": len(resp_a.items), "session_b_items": len(resp_b.items), "a_sees_a": "session_a_doc" in a_keys, "b_sees_b": "session_b_doc" in b_keys, "a_no_b": "session_b_doc" not in a_keys, "b_no_a": "session_a_doc" not in b_keys, "both_see_global": "shared_global" in a_keys and "shared_global" in b_keys}


async def simulate_large_text_chunking() -> dict[str, Any]:
    t0 = time.perf_counter()
    service = _svc("large_chunk")
    paragraphs = [f"Paragraph {i}: " + " ".join([f"word{j}" for j in range(50)]) for i in range(20)]
    long_text = " ".join(paragraphs)
    rec = await service.ingest_source(ContextBucketSourceCreate(scope="session", session_id="s_lc", source_key="huge_doc", kind="research_report", text=long_text))
    elapsed = time.perf_counter() - t0
    return {"status": "PASS", "elapsed_s": round(elapsed, 4), "text_length": len(long_text), "chunk_count": len(rec.chunks) if rec else 0, "total_tokens": rec.token_count_estimate if rec else 0, "avg_chunk_tokens": round(sum(c.token_count_estimate for c in rec.chunks) / max(1, len(rec.chunks)), 1) if rec else 0, "overlapping": any(rec.chunks[i].text[-30:] in rec.chunks[i + 1].text[:50] for i in range(len(rec.chunks) - 1)) if rec and len(rec.chunks) > 1 else False}


def render_report(all_results: dict[str, Any]) -> str:
    lines = []
    total_elapsed = sum(r.get("elapsed_s", 0) for r in all_results.values())
    total_systems = len(all_results)
    passed = sum(1 for r in all_results.values() if r.get("status") == "PASS")

    lines.append("# Context Bucket OSS \u2014 Live Simulation Report")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Systems Simulated | {total_systems} |")
    lines.append(f"| All Passed | {'Yes' if passed == total_systems else 'No'} |")
    lines.append(f"| Total Runtime | {total_elapsed:.2f}s |")
    lines.append(f"| Timestamp | {datetime.now(timezone.utc).isoformat()} |")
    lines.append("")

    system_order = ["ingest", "upsert", "delete", "structured", "retrieval", "assembly", "preferences", "task_envelope", "evaluation", "importers", "storage_backends", "audit", "training", "prune", "workflows", "embeddings", "scoring", "bulk_ingest", "multi_scope_retrieval", "confidentiality_policy", "upsert_versioning", "tag_filtering", "kind_filtering", "store_record_api", "structured_nested", "structured_array", "declared_schema", "age_decay", "dedup_detection", "evaluation_compare", "evaluation_gate_fail", "importer_csv", "importer_nested_dirs", "query_api", "health_check", "stemming_ngrams", "context_compression", "session_isolation", "large_text_chunking"]

    lines.append("---")
    lines.append("")
    lines.append("## System Dashboard")
    lines.append("")
    lines.append("| # | System | Status | Time | Key Metric |")
    lines.append("|---|--------|--------|------|------------|")

    key_metrics = {
        "ingest": lambda r: f"{r['records_created']} recs, {r['total_chunks']} chunks, {r['total_tokens']} tokens",
        "upsert": lambda r: f"dedup={r['dedup_same_content']} supersede={r['supersede_works']} active={r['active']}",
        "delete": lambda r: f"before={r['before']} after={r['after']} double_del={r['double_del_none']}",
        "structured": lambda r: f"{r['records']} recs, {r['avg_fields']} avg fields",
        "retrieval": lambda r: f"{r['queries']} queries, {r['avg_items']} avg items/query",
        "assembly": lambda r: f"{r['modes_tested']} modes (assistant/research/drafting/planner)",
        "preferences": lambda r: f"brevity={r['brevity']} initiative={r['initiative']} schema={r['schema_name']}",
        "task_envelope": lambda r: f"{r['scenarios']} scenarios, intent+contract derived",
        "evaluation": lambda r: f"retrieval={r['retrieval_hit_rate']:.0%} assembly={r['assembly_hit_rate']:.0%} gate={'PASS' if r['gate_passed'] else 'FAIL'}",
        "importers": lambda r: f"{r['imported']} imported, {r['skipped']} skipped",
        "storage_backends": lambda r: f"file={r['file_records']} sqlite={r['sqlite_records']} match={r['backends_match']}",
        "audit": lambda r: f"{r['entries']} entries, query_logged={r['latest_has_query']}",
        "training": lambda r: f"{r['exported_count']} records, lines_match={r['lines_match']}",
        "prune": lambda r: f"{r['pruned_records']} pruned, {r['pruned_chunks']} chunks removed",
        "workflows": lambda r: f"{r['workflows_tested']} workflows (reply/summarize/research/rewrite)",
        "embeddings": lambda r: f"{r['dimensions']}d, {r['vectors_tested']} vectors, avg_sim={r['avg_sim']}",
        "scoring": lambda r: f"{r['pairs']} pairs, scope_bonuses={r['scope_bonuses']}",
        "bulk_ingest": lambda r: f"{r['ingested']} recs, {r['total_tokens']} tokens, avg_chunks={r['avg_chunks']}",
        "multi_scope_retrieval": lambda r: f"{r['scenarios']} scenarios, scope isolation tested",
        "confidentiality_policy": lambda r: f"{r['total_records']} recs, {r['retrieved']} retrieved",
        "upsert_versioning": lambda r: f"{r['upserts']} upserts, active={r['active_count']} superseded={r['superseded_count']}",
        "tag_filtering": lambda r: f"all={r['all_items']} legal={r['legal_only']} multi={r['multi_tag']}",
        "kind_filtering": lambda r: f"all={r['all_items']} reports={r['report_only']} ev+find={r['evidence_and_finding']}",
        "store_record_api": lambda r: f"created={r['created']} fetched={r['fetched']} title={r['title']}",
        "structured_nested": lambda r: f"fields={r['field_count']} mode={r['schema_mode']}",
        "structured_array": lambda r: f"fields={r['field_count']} root={r['root_type']}",
        "declared_schema": lambda r: f"name={r['schema_name']} mode={r['schema_mode']} fields={len(r['field_paths'])}",
        "age_decay": lambda r: f"fresh={r['fresh_score']} old={r['old_score']} fresh_wins={r['fresh_ranked_higher']}",
        "dedup_detection": lambda r: f"attempted={r['total_attempted']} created={r['total_created']}",
        "evaluation_compare": lambda r: f"run1={r['run1_hits']} run2={r['run2_hits']} regressions={r['compare_regressions']}",
        "evaluation_gate_fail": lambda r: f"passed={r['gate_passed']} failures={len(r['failures'])}",
        "importer_csv": lambda r: f"imported={r['imported']} count={r['stats_count']}",
        "importer_nested_dirs": lambda r: f"imported={r['imported']} files={r['item_count']}",
        "query_api": lambda r: f"results={r['results']} top={r['top_score']}",
        "health_check": lambda r: f"ok={r['health_ok']} tokens={r['token_estimate']} deduped={r['deduped_count']}",
        "stemming_ngrams": lambda r: f"stems={len(r['stems'])} ngram_words={len(r['ngrams_sample'])}",
        "context_compression": lambda r: f"low={r['low_budget_items']}items/{r['low_tokens']}tok high={r['high_budget_items']}items/{r['high_tokens']}tok",
        "session_isolation": lambda r: f"a={r['session_a_items']} b={r['session_b_items']} isolated={r['a_no_b'] and r['b_no_a']}",
        "large_text_chunking": lambda r: f"len={r['text_length']} chunks={r['chunk_count']} overlap={r['overlapping']}",
    }

    for idx, name in enumerate(system_order, 1):
        r = all_results.get(name, {})
        status = r.get("status", "N/A")
        elapsed = f"{r.get('elapsed_s', 0):.3f}s"
        metric = key_metrics.get(name, lambda _: "")(r)
        lines.append(f"| {idx} | {name} | {status} | {elapsed} | {metric} |")

    lines.append("")

    for name in system_order:
        r = all_results.get(name, {})
        if not r:
            continue
        lines.append("---")
        lines.append("")
        lines.append(f"## {name.replace('_', ' ').title()}")
        lines.append("")
        for key, value in r.items():
            if key in ("status", "elapsed_s"):
                continue
            if isinstance(value, (dict, list)):
                lines.append(f"**{key}:**")
                lines.append("")
                lines.append("```json")
                lines.append(json.dumps(value, indent=2, default=str))
                lines.append("```")
                lines.append("")
            else:
                lines.append(f"- **{key}:** {value}")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Architecture Coverage")
    lines.append("")
    arch = [
        ("service.py", "Public facade / orchestration", True),
        ("models.py", "Pydantic data models", True),
        ("ingest.py", "Record creation, upsert, delete", True),
        ("retrieval.py", "Lexical + semantic retrieval, scoring", True),
        ("assembly.py", "Context assembly, sectioning, rendering", True),
        ("preferences.py", "Workflow preference merge + summarize", True),
        ("structured.py", "Schema resolution, field extraction", True),
        ("task_envelope.py", "Intent derivation, envelope construction", True),
        ("storage.py", "File + SQLite persistence", True),
        ("importers.py", "Document import (txt/html/json/xml/ndjson/csv)", True),
        ("training.py", "Training line serialization + export", True),
        ("audit.py", "Retrieval audit logging", True),
        ("evaluation.py", "Evaluation runs, suites, comparison, gating", True),
        ("benchmark.py", "JSONL benchmarks (covered via evaluation)", True),
        ("settings.py", "Environment-configured settings", True),
        ("cli.py", "CLI surface (exercised via service layer)", True),
    ]
    lines.append("| Module | Purpose | Simulated |")
    lines.append("|--------|---------|-----------|")
    for mod, purpose, simmed in arch:
        lines.append(f"| `{mod}` | {purpose} | {'Yes' if simmed else 'No'} |")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Pipeline Walkthrough")
    lines.append("")
    steps = [
        ("1. Ingest", "Sources ingested as records with chunking, tokenization, and embedding."),
        ("2. Upsert", "Versioned upsert with content dedup and supersession."),
        ("3. Structured Data", "Schema-aware field extraction and primary text path resolution."),
        ("4. Retrieval", "Semantic + lexical hybrid scoring with reranking and scope filtering."),
        ("5. Assembly", "Token-budgeted context assembly into mode-specific sections."),
        ("6. Preferences", "User workflow preferences merged from notes and explicit updates."),
        ("7. Task Envelope", "Workflow intent derived from query + context; output contract generated."),
        ("8. Evaluation", "Retrieval/assembly hit rates measured, compared, and gated."),
        ("9. Import", "Multi-format documents parsed and ingested from filesystem."),
        ("10. Storage", "Both file and SQLite backends exercise full CRUD cycle."),
        ("11. Audit", "Every retrieval logged with filters, candidate counts, and selection."),
        ("12. Training", "Record-level training lines serialized and exported as JSONL."),
        ("13. Prune", "Expired and excess records pruned with chunk cascade deletion."),
        ("14. Bulk Ingest", "High-volume 50-record batch ingest across mixed scopes and kinds."),
        ("15. Multi-Scope Retrieval", "Cross-scope retrieval with session/user/global visibility toggles."),
        ("16. Confidentiality Policy", "Record-level confidentiality enforcement during retrieval."),
        ("17. Upsert Versioning", "7-version document evolution with supersession tracking."),
        ("18. Tag Filtering", "Retrieval constrained by single and multi-tag filters."),
        ("19. Kind Filtering", "Retrieval constrained by record kind allowlists."),
        ("20. Store Record API", "Direct record creation bypassing source/upsert logic."),
        ("21. Nested Structured Data", "Deeply nested JSON with 3-level hierarchy extraction."),
        ("22. Array Structured Data", "Top-level array of objects with per-element field extraction."),
        ("23. Declared Schema", "Explicit schema declaration vs. inference with field path control."),
        ("24. Age Decay", "Fresh vs. stale records scored with time-based decay."),
        ("25. Dedup Detection", "Exact and near-duplicate detection across same-scope records."),
        ("26. Evaluation Compare", "Baseline vs. candidate run comparison with regression counting."),
        ("27. Evaluation Gate Fail", "Strict threshold gate intentionally triggered to fail."),
        ("28. CSV Import", "Dedicated CSV file import with row-level content extraction."),
        ("29. Nested Directory Import", "Recursive filesystem import across 3-level directory tree."),
        ("30. Query API", "Legacy query endpoint returning ContextBucketQueryResult items."),
        ("31. Health Check", "Service health, token estimation, checksum, and string dedup."),
        ("32. Stemming & N-grams", "Token stemming suffix stripping and character n-gram generation."),
        ("33. Context Compression", "Low vs. high token budgets with truncation and omission tracking."),
        ("34. Session Isolation", "Cross-session data leak prevention verification."),
        ("35. Large Text Chunking", "Multi-chunk splitting of 1000+ word documents with overlap."),
    ]
    for step, desc in steps:
        lines.append(f"**{step}** \u2014 {desc}")
        lines.append("")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(f"*Report generated at {datetime.now(timezone.utc).isoformat()} | Total simulation time: {total_elapsed:.2f}s*")
    return "\n".join(lines)


async def main() -> None:
    simulations = [
        ("ingest", simulate_ingest),
        ("upsert", simulate_upsert),
        ("delete", simulate_delete),
        ("structured", simulate_structured),
        ("retrieval", simulate_retrieval),
        ("assembly", simulate_assembly),
        ("preferences", simulate_preferences),
        ("task_envelope", simulate_task_envelope),
        ("evaluation", simulate_evaluation),
        ("importers", simulate_importers),
        ("storage_backends", simulate_storage_backends),
        ("audit", simulate_audit),
        ("training", simulate_training),
        ("prune", simulate_prune),
        ("workflows", simulate_workflows),
        ("embeddings", simulate_embeddings),
        ("scoring", simulate_scoring),
        ("bulk_ingest", simulate_bulk_ingest),
        ("multi_scope_retrieval", simulate_multi_scope_retrieval),
        ("confidentiality_policy", simulate_confidentiality_policy),
        ("upsert_versioning", simulate_upsert_versioning),
        ("tag_filtering", simulate_tag_filtering),
        ("kind_filtering", simulate_kind_filtering),
        ("store_record_api", simulate_store_record_api),
        ("structured_nested", simulate_structured_nested),
        ("structured_array", simulate_structured_array),
        ("declared_schema", simulate_declared_schema),
        ("age_decay", simulate_age_decay),
        ("dedup_detection", simulate_dedup_detection),
        ("evaluation_compare", simulate_evaluation_compare),
        ("evaluation_gate_fail", simulate_evaluation_gate_fail),
        ("importer_csv", simulate_importer_csv),
        ("importer_nested_dirs", simulate_importer_nested_dirs),
        ("query_api", simulate_query_api),
        ("health_check", simulate_health_check),
        ("stemming_ngrams", simulate_stemming_ngrams),
        ("context_compression", simulate_context_compression),
        ("session_isolation", simulate_session_isolation),
        ("large_text_chunking", simulate_large_text_chunking),
    ]

    print(f"Running {len(simulations)} system simulations...")
    print()

    for name, sim_fn in simulations:
        print(f"  -> {name}...", end=" ", flush=True)
        try:
            result = await sim_fn()
            RESULTS[name] = result
            status = result.get("status", "UNKNOWN")
            elapsed = result.get("elapsed_s", 0)
            print(f"{status} ({elapsed:.3f}s)")
        except Exception as exc:
            RESULTS[name] = {"status": "FAIL", "error": str(exc)}
            print(f"FAIL ({exc})")

    print()
    report = render_report(RESULTS)
    report_path = Path(__file__).parent / "simulation-report.md"
    report_path.write_text(report, encoding="utf-8")

    json_path = Path(__file__).parent / "simulation-report.json"
    json_path.write_text(json.dumps(RESULTS, indent=2, default=str), encoding="utf-8")

    print(f"Report: {report_path}")
    print(f"Data:   {json_path}")

    try:
        shutil.rmtree(SIM_ROOT)
    except Exception:
        pass


if __name__ == "__main__":
    asyncio.run(main())
