from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from context_bucket.models import (
    ContextBucketDataSchema,
    ContextBucketEvaluationCase,
    ContextBucketEvaluationRequest,
    ContextBucketEvaluationSuite,
    ContextBucketSourceUpsert,
)
from context_bucket.service import ContextBucketService
from context_bucket.evaluation import evaluate_cases, persist_evaluation_run


class ContextBucketBenchmarkError(ValueError):
    pass


def load_jsonl_dataset(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise ContextBucketBenchmarkError(f"dataset file not found: {path}") from exc
    for line_number, line in enumerate(lines, start=1):
        cleaned = line.strip()
        if not cleaned:
            continue
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ContextBucketBenchmarkError(f"dataset line {line_number} is not valid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ContextBucketBenchmarkError(f"dataset line {line_number} must be a JSON object")
        source_key = str(payload.get("source_key") or "").strip()
        if not source_key:
            raise ContextBucketBenchmarkError(f"dataset line {line_number} missing required source_key")
        if not str(payload.get("text") or "").strip() and payload.get("structured_data") is None:
            raise ContextBucketBenchmarkError(
                f"dataset line {line_number} must include text or structured_data"
            )
        records.append(payload)
    return records


def load_evaluation_suite(path: Path, *, suite_name: str, token_budget: int) -> ContextBucketEvaluationSuite:
    try:
        raw_payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ContextBucketBenchmarkError(f"cases file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ContextBucketBenchmarkError(f"cases file is not valid JSON: {exc}") from exc
    if not isinstance(raw_payload, dict):
        raise ContextBucketBenchmarkError("cases file must be a JSON object")
    raw_cases = raw_payload.get("cases")
    if not isinstance(raw_cases, list):
        raise ContextBucketBenchmarkError("cases file must include a cases array")
    cases: list[ContextBucketEvaluationCase] = []
    for index, item in enumerate(raw_cases, start=1):
        if not isinstance(item, dict):
            raise ContextBucketBenchmarkError(f"case {index} must be a JSON object")
        payload = dict(item)
        payload.setdefault("token_budget", token_budget)
        try:
            cases.append(ContextBucketEvaluationCase.model_validate(payload))
        except Exception as exc:
            raise ContextBucketBenchmarkError(f"case {index} is invalid: {exc}") from exc
    return ContextBucketEvaluationSuite(name=suite_name or str(raw_payload.get("name") or "benchmark"), cases=cases)


async def run_jsonl_benchmark(
    *,
    service: ContextBucketService,
    dataset_jsonl: Path,
    cases_json: Path,
    output_dir: Path,
    suite_name: str,
    limit: int,
    token_budget: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    dataset_records = load_jsonl_dataset(dataset_jsonl)
    suite = load_evaluation_suite(cases_json, suite_name=suite_name, token_budget=token_budget)
    imported: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []

    for index, raw_record in enumerate(dataset_records, start=1):
        try:
            data_schema = raw_record.get("data_schema")
            record = await service.upsert_source(
                ContextBucketSourceUpsert(
                    source_key=str(raw_record["source_key"]),
                    text=raw_record.get("text"),
                    kind=raw_record.get("kind", "research_finding"),
                    scope=raw_record.get("scope", "session"),
                    title=raw_record.get("title"),
                    summary=raw_record.get("summary"),
                    user_id=raw_record.get("user_id"),
                    session_id=raw_record.get("session_id"),
                    tags=list(raw_record.get("tags") or []),
                    metadata=dict(raw_record.get("metadata") or {}),
                    structured_data=raw_record.get("structured_data"),
                    data_schema=(
                        ContextBucketDataSchema.model_validate(data_schema)
                        if data_schema is not None
                        else None
                    ),
                )
            )
            imported.append({"source_key": str(raw_record["source_key"]), "record_id": record.id})
        except Exception as exc:
            skipped.append(
                {
                    "line": str(index),
                    "source_key": str(raw_record.get("source_key") or ""),
                    "reason": str(exc),
                }
            )

    await service.save_evaluation_suite(suite.name, suite)
    evaluation = await evaluate_cases(service, ContextBucketEvaluationRequest(cases=suite.cases, limit=limit))
    persist_evaluation_run(service, suite.name, evaluation)
    runs = await service.list_evaluation_runs()
    run_id = next((run.run_id for run in runs if run.suite_name == suite.name), None)
    stats = await service.stats()
    elapsed_seconds = time.perf_counter() - started

    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "benchmark-result.json"
    report_path = output_dir / "benchmark-report.md"
    result = {
        "suite_name": suite.name,
        "run_id": run_id,
        "dataset_path": str(dataset_jsonl),
        "cases_path": str(cases_json),
        "data_root": str(service.root),
        "output_dir": str(output_dir),
        "runtime_seconds": round(elapsed_seconds, 3),
        "imported_records": len(imported),
        "skipped_records": skipped,
        "total_cases": evaluation.total_cases,
        "retrieval_hits": evaluation.retrieval_hits,
        "assembly_hits": evaluation.assembly_hits,
        "retrieval_hit_rate": _ratio(evaluation.retrieval_hits, evaluation.total_cases),
        "assembly_hit_rate": _ratio(evaluation.assembly_hits, evaluation.total_cases),
        "stats": stats.model_dump(mode="json"),
        "evaluation": evaluation.model_dump(mode="json"),
        "result_path": str(result_path),
        "report_path": str(report_path),
    }
    result["behavior_evidence"] = behavior_evidence(result)
    result_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    report_path.write_text(render_benchmark_report(result), encoding="utf-8")
    return result


def behavior_evidence(result: dict[str, Any]) -> dict[str, Any]:
    evaluation = result["evaluation"]
    return {
        "evidence_type": "benchmark_run",
        "suite_name": result["suite_name"],
        "run_id": result["run_id"],
        "dataset_path": result["dataset_path"],
        "cases_path": result["cases_path"],
        "data_root": result["data_root"],
        "hit_rates": {
            "retrieval": result["retrieval_hit_rate"],
            "assembly": result["assembly_hit_rate"],
        },
        "case_evidence": [
            {
                "name": item["name"],
                "query_text": item["query_text"],
                "retrieval_hit": item["retrieval_hit"],
                "assembly_hit": item["assembly_hit"],
                "expected_terms_scope": item.get("expected_terms_scope", "assembled_context"),
                "retrieved_source_keys": item["retrieved_source_keys"],
                "matched_expected_source_keys": item["matched_expected_source_keys"],
                "matched_expected_terms": item["matched_expected_terms"],
                "audit_id": item.get("audit_id"),
            }
            for item in evaluation.get("results", [])
        ],
    }


def render_benchmark_report(result: dict[str, Any]) -> str:
    evaluation = result["evaluation"]
    lines = [
        f"# Context Bucket Benchmark: {result['suite_name']}",
        "",
        "## Summary",
        "",
        f"- Dataset: `{result['dataset_path']}`",
        f"- Cases: `{result['cases_path']}`",
        f"- Data root: `{result['data_root']}`",
        f"- Runtime seconds: {result['runtime_seconds']}",
        f"- Imported records: {result['imported_records']}",
        f"- Skipped records: {len(result['skipped_records'])}",
        f"- Retrieval hit rate: {result['retrieval_hit_rate']:.3f}",
        f"- Assembly hit rate: {result['assembly_hit_rate']:.3f}",
        "",
        "## Cases",
        "",
    ]
    for item in evaluation.get("results", []):
        lines.extend(
            [
                f"### {item['name']}",
                "",
                f"- Query: {item['query_text']}",
                f"- Retrieval hit: {str(item['retrieval_hit']).lower()}",
                f"- Assembly hit: {str(item['assembly_hit']).lower()}",
                f"- Retrieved source keys: {', '.join(item['retrieved_source_keys']) or '(none)'}",
                f"- Matched source keys: {', '.join(item['matched_expected_source_keys']) or '(none)'}",
                f"- Matched terms: {', '.join(item['matched_expected_terms']) or '(none)'}",
                f"- Expected terms scope: {item.get('expected_terms_scope', 'assembled_context')}",
                f"- Retrieval count: {item['retrieval_count']}",
                f"- Assembly token estimate: {item['assembly_token_count_estimate']}",
                f"- Audit ID: {item.get('audit_id') or '(none)'}",
                "",
            ]
        )
    if result["skipped_records"]:
        lines.extend(["## Skipped Records", ""])
        for item in result["skipped_records"]:
            lines.append(f"- line {item['line']} `{item['source_key']}`: {item['reason']}")
        lines.append("")
    evidence = result.get("behavior_evidence") or {}
    if evidence:
        lines.extend(
            [
                "## Behavior Evidence",
                "",
                f"- Evidence type: {evidence.get('evidence_type', 'benchmark_run')}",
                f"- Suite: {evidence.get('suite_name') or result['suite_name']}",
                f"- Run ID: {evidence.get('run_id') or result.get('run_id') or '(none)'}",
                f"- Dataset path: `{evidence.get('dataset_path') or result['dataset_path']}`",
                f"- Cases path: `{evidence.get('cases_path') or result['cases_path']}`",
                f"- Data root: `{evidence.get('data_root') or result['data_root']}`",
                f"- Retrieval hit rate: {result['retrieval_hit_rate']:.3f}",
                f"- Assembly hit rate: {result['assembly_hit_rate']:.3f}",
                "",
            ]
        )
        for item in evidence.get("case_evidence", []):
            lines.extend(
                [
                    f"### Evidence: {item['name']}",
                    "",
                    f"- Query: {item['query_text']}",
                    f"- Expected terms scope: {item.get('expected_terms_scope', 'assembled_context')}",
                    f"- Retrieved source keys: {', '.join(item['retrieved_source_keys']) or '(none)'}",
                    f"- Matched source keys: {', '.join(item['matched_expected_source_keys']) or '(none)'}",
                    f"- Matched terms: {', '.join(item['matched_expected_terms']) or '(none)'}",
                    f"- Audit ID: {item.get('audit_id') or '(none)'}",
                    "",
                ]
            )
    lines.extend(
        [
            "## Outputs",
            "",
            f"- JSON: `{result['result_path']}`",
            f"- Markdown: `{result['report_path']}`",
            "",
        ]
    )
    return "\n".join(lines)


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 3)
