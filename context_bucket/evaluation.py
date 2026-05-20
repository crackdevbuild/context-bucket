from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from context_bucket.models import (
    ContextBucketAssembleRequest,
    ContextBucketEvaluationCompareItem,
    ContextBucketEvaluationCompareRequest,
    ContextBucketEvaluationCompareResponse,
    ContextBucketEvaluationGateRequest,
    ContextBucketEvaluationGateResponse,
    ContextBucketEvaluationRequest,
    ContextBucketEvaluationResponse,
    ContextBucketEvaluationResult,
    ContextBucketEvaluationRunRecord,
    ContextBucketEvaluationSuite,
    ContextBucketRetrieveRequest,
)


async def run_evaluations(service: Any, payload: ContextBucketEvaluationRequest) -> ContextBucketEvaluationResponse:
    response = await evaluate_cases(service, payload)
    persist_evaluation_run(service, "adhoc", response)
    return response


async def save_evaluation_suite(
    service: Any,
    suite_name: str,
    payload: ContextBucketEvaluationSuite,
) -> ContextBucketEvaluationSuite:
    suite = payload.model_copy(update={"name": suite_name})
    service.eval_suites_root.mkdir(parents=True, exist_ok=True)
    (service.eval_suites_root / f"{suite_name}.json").write_text(
        suite.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return suite


async def list_evaluation_suites(service: Any) -> list[ContextBucketEvaluationSuite]:
    if not service.eval_suites_root.exists():
        return []
    suites: list[ContextBucketEvaluationSuite] = []
    for path in sorted(service.eval_suites_root.glob("*.json")):
        try:
            suites.append(ContextBucketEvaluationSuite.model_validate_json(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    return suites


async def run_evaluation_suite(service: Any, suite_name: str) -> ContextBucketEvaluationResponse:
    path = service.eval_suites_root / f"{suite_name}.json"
    suite = ContextBucketEvaluationSuite.model_validate_json(path.read_text(encoding="utf-8"))
    response = await evaluate_cases(service, ContextBucketEvaluationRequest(cases=suite.cases))
    persist_evaluation_run(service, suite_name, response)
    return response


async def list_evaluation_runs(service: Any) -> list[ContextBucketEvaluationRunRecord]:
    if not service.eval_runs_root.exists():
        return []
    records: list[ContextBucketEvaluationRunRecord] = []
    for path in sorted(service.eval_runs_root.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            records.append(
                ContextBucketEvaluationRunRecord(
                    run_id=payload["run_id"],
                    suite_name=payload["suite_name"],
                    created_at=datetime.fromisoformat(payload["created_at"]),
                    total_cases=payload["total_cases"],
                    retrieval_hits=payload["retrieval_hits"],
                    assembly_hits=payload["assembly_hits"],
                )
            )
        except Exception:
            continue
    records.sort(key=lambda item: item.created_at, reverse=True)
    return records


async def compare_evaluation_runs(
    service: Any,
    payload: ContextBucketEvaluationCompareRequest,
) -> ContextBucketEvaluationCompareResponse:
    baseline = load_evaluation_run(service, payload.baseline_run_id)
    candidate = load_evaluation_run(service, payload.candidate_run_id)
    return build_evaluation_compare_response(
        baseline_run_id=payload.baseline_run_id,
        candidate_run_id=payload.candidate_run_id,
        baseline=baseline,
        candidate=candidate,
    )


async def gate_evaluation_run(
    service: Any,
    payload: ContextBucketEvaluationGateRequest,
) -> ContextBucketEvaluationGateResponse:
    candidate = load_evaluation_run(service, payload.candidate_run_id)
    total_cases = int(candidate.get("total_cases", 0))
    retrieval_hits = int(candidate.get("retrieval_hits", 0))
    assembly_hits = int(candidate.get("assembly_hits", 0))
    retrieval_hit_rate = safe_ratio(retrieval_hits, total_cases)
    assembly_hit_rate = safe_ratio(assembly_hits, total_cases)
    thresholds = payload.thresholds
    retrieval_regressions = 0
    assembly_regressions = 0
    failures: list[str] = []

    if retrieval_hit_rate < thresholds.min_retrieval_hit_rate:
        failures.append(
            "retrieval_hit_rate_below_threshold:"
            f" actual={retrieval_hit_rate:.3f} required={thresholds.min_retrieval_hit_rate:.3f}"
        )
    if assembly_hit_rate < thresholds.min_assembly_hit_rate:
        failures.append(
            "assembly_hit_rate_below_threshold:"
            f" actual={assembly_hit_rate:.3f} required={thresholds.min_assembly_hit_rate:.3f}"
        )

    if payload.baseline_run_id:
        baseline = load_evaluation_run(service, payload.baseline_run_id)
        comparison = build_evaluation_compare_response(
            baseline_run_id=payload.baseline_run_id,
            candidate_run_id=payload.candidate_run_id,
            baseline=baseline,
            candidate=candidate,
        )
        retrieval_regressions = comparison.retrieval_regressions
        assembly_regressions = comparison.assembly_regressions
        if retrieval_regressions > thresholds.max_retrieval_regressions:
            failures.append(
                "retrieval_regressions_above_threshold:"
                f" actual={retrieval_regressions} allowed={thresholds.max_retrieval_regressions}"
            )
        if assembly_regressions > thresholds.max_assembly_regressions:
            failures.append(
                "assembly_regressions_above_threshold:"
                f" actual={assembly_regressions} allowed={thresholds.max_assembly_regressions}"
            )

    return ContextBucketEvaluationGateResponse(
        passed=not failures,
        candidate_run_id=payload.candidate_run_id,
        baseline_run_id=payload.baseline_run_id,
        total_cases=total_cases,
        retrieval_hit_rate=retrieval_hit_rate,
        assembly_hit_rate=assembly_hit_rate,
        retrieval_regressions=retrieval_regressions,
        assembly_regressions=assembly_regressions,
        failures=failures,
    )


def persist_evaluation_run(
    service: Any,
    suite_name: str,
    response: ContextBucketEvaluationResponse,
) -> str:
    run_id = f"eval_{uuid4().hex}"
    service.eval_runs_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "suite_name": suite_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_cases": response.total_cases,
        "retrieval_hits": response.retrieval_hits,
        "assembly_hits": response.assembly_hits,
        "results": [item.model_dump(mode="json") for item in response.results],
    }
    (service.eval_runs_root / f"{run_id}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return run_id


def load_evaluation_run(service: Any, run_id: str) -> dict[str, Any]:
    path = service.eval_runs_root / f"{run_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def build_evaluation_compare_response(
    *,
    baseline_run_id: str,
    candidate_run_id: str,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> ContextBucketEvaluationCompareResponse:
    baseline_map = {item["name"]: item for item in baseline.get("results", [])}
    candidate_map = {item["name"]: item for item in candidate.get("results", [])}
    names = sorted(set(baseline_map) | set(candidate_map))
    items: list[ContextBucketEvaluationCompareItem] = []
    retrieval_regressions = 0
    assembly_regressions = 0
    for name in names:
        left = baseline_map.get(name, {})
        right = candidate_map.get(name, {})
        left_retrieval = bool(left.get("retrieval_hit"))
        right_retrieval = bool(right.get("retrieval_hit"))
        left_assembly = bool(left.get("assembly_hit"))
        right_assembly = bool(right.get("assembly_hit"))
        if left_retrieval and not right_retrieval:
            retrieval_regressions += 1
        if left_assembly and not right_assembly:
            assembly_regressions += 1
        items.append(
            ContextBucketEvaluationCompareItem(
                name=name,
                retrieval_changed=left_retrieval != right_retrieval,
                assembly_changed=left_assembly != right_assembly,
                baseline_retrieval_hit=left_retrieval,
                candidate_retrieval_hit=right_retrieval,
                baseline_assembly_hit=left_assembly,
                candidate_assembly_hit=right_assembly,
            )
        )
    return ContextBucketEvaluationCompareResponse(
        baseline_run_id=baseline_run_id,
        candidate_run_id=candidate_run_id,
        retrieval_regressions=retrieval_regressions,
        assembly_regressions=assembly_regressions,
        items=items,
    )


def safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


async def evaluate_cases(service: Any, payload: ContextBucketEvaluationRequest) -> ContextBucketEvaluationResponse:
    results: list[ContextBucketEvaluationResult] = []
    retrieval_hits = 0
    assembly_hits = 0
    for case in payload.cases:
        retrieved = await service.retrieve_context(
            ContextBucketRetrieveRequest(
                query_text=case.query_text,
                session_id=case.session_id,
                user_id=case.user_id,
                model_target=case.model_target,
                include_user_scope=case.include_user_scope,
                include_global_scope=case.include_global_scope,
                limit=payload.limit,
            )
        )
        assembled = await service.assemble_context(
            ContextBucketAssembleRequest(
                query_text=case.query_text,
                session_id=case.session_id,
                user_id=case.user_id,
                model_target=case.model_target,
                include_user_scope=case.include_user_scope,
                include_global_scope=case.include_global_scope,
                token_budget=case.token_budget,
                limit=payload.limit,
            )
        )
        retrieved_source_keys = [
            str(item.provenance.get("source_key"))
            for item in retrieved.items
            if item.provenance.get("source_key")
        ]
        expected_prefixes = set(k.split(":")[0] for k in case.expected_source_keys if ":" in k)
        if expected_prefixes:
            matched_source_keys = sorted(
                k for k in retrieved_source_keys
                if k.split(":")[0] in expected_prefixes
            )
        else:
            matched_source_keys = sorted(set(retrieved_source_keys) & set(case.expected_source_keys))
        if case.expected_terms_scope == "retrieved_records":
            record_ids = list(dict.fromkeys(item.record_id for item in retrieved.items))
            term_text = "\n".join(
                record.text
                for record_id in record_ids
                for record in [service.get_record_sync(record_id)]
                if record is not None
            )
        else:
            term_text = assembled.context_text
        matched_terms = sorted(
            term for term in case.expected_terms if term.lower() in term_text.lower()
        )
        retrieval_hit = bool(matched_source_keys) if case.expected_source_keys else bool(retrieved.items)
        assembly_hit = bool(matched_terms) if case.expected_terms else retrieval_hit
        if retrieval_hit:
            retrieval_hits += 1
        if assembly_hit:
            assembly_hits += 1
        results.append(
            ContextBucketEvaluationResult(
                name=case.name,
                query_text=case.query_text,
                retrieved_source_keys=retrieved_source_keys,
                matched_expected_source_keys=matched_source_keys,
                matched_expected_terms=matched_terms,
                expected_terms_scope=case.expected_terms_scope,
                retrieval_hit=retrieval_hit,
                assembly_hit=assembly_hit,
                retrieval_count=len(retrieved.items),
                assembly_token_count_estimate=assembled.token_count_estimate,
                audit_id=assembled.audit_id or retrieved.audit_id,
            )
        )
    return ContextBucketEvaluationResponse(
        total_cases=len(payload.cases),
        retrieval_hits=retrieval_hits,
        assembly_hits=assembly_hits,
        results=results,
    )
