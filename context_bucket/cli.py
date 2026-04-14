from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import typer

from context_bucket.benchmark import ContextBucketBenchmarkError, run_jsonl_benchmark
from context_bucket.models import (
    ContextBucketAssembleRequest,
    ContextBucketDataSchema,
    ContextBucketDocumentImportRequest,
    ContextBucketRecordCreate,
    ContextBucketRetrieveRequest,
    ContextBucketSourceCreate,
    ContextBucketSourceDelete,
    ContextBucketSourceUpsert,
    ContextBucketWorkflowPreferenceUpdateRequest,
)
from context_bucket.service import ContextBucketService
from context_bucket.settings import Settings

app = typer.Typer(help="Local-first context bucket CLI.")
service = ContextBucketService()


def _load_json_option(value: str | None, *, option_name: str) -> Any | None:
    if value is None:
        return None
    raw = value.strip()
    if not raw:
        return None
    if raw.startswith("@"):
        raw = Path(raw[1:]).read_text(encoding="utf-8")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"{option_name} must be valid JSON or @path/to/file.json: {exc}") from exc


def _intent_args(intent_data: str | None, intent_schema: str | None) -> tuple[Any | None, ContextBucketDataSchema | None]:
    parsed_intent_data = _load_json_option(intent_data, option_name="--intent-data")
    parsed_intent_schema = _load_json_option(intent_schema, option_name="--intent-schema")
    schema = ContextBucketDataSchema.model_validate(parsed_intent_schema) if parsed_intent_schema is not None else None
    return parsed_intent_data, schema


@app.command("store")
def store(
    kind: str,
    text: str,
    scope: str = "session",
    title: str | None = None,
    summary: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
) -> None:
    record = asyncio.run(
        service.store_record(
            ContextBucketRecordCreate(
                kind=kind,  # type: ignore[arg-type]
                text=text,
                scope=scope,  # type: ignore[arg-type]
                title=title,
                summary=summary,
                user_id=user_id,
                session_id=session_id,
            )
        )
    )
    typer.echo(json.dumps(record.model_dump(mode="json") if record else None, indent=2, default=str))


@app.command("ingest-source")
def ingest_source(
    text: str | None = None,
    kind: str = "research_finding",
    scope: str = "session",
    title: str | None = None,
    summary: str | None = None,
    source_key: str | None = None,
    source_type: str = "direct_text",
    user_id: str | None = None,
    session_id: str | None = None,
) -> None:
    record = asyncio.run(
        service.ingest_source(
            ContextBucketSourceCreate(
                text=text,
                kind=kind,  # type: ignore[arg-type]
                scope=scope,  # type: ignore[arg-type]
                title=title,
                summary=summary,
                source_key=source_key,
                source_type=source_type,  # type: ignore[arg-type]
                user_id=user_id,
                session_id=session_id,
            )
        )
    )
    typer.echo(json.dumps(record.model_dump(mode="json") if record else None, indent=2, default=str))


@app.command("upsert-source")
def upsert_source(
    source_key: str,
    text: str | None = None,
    kind: str = "research_finding",
    scope: str = "session",
    title: str | None = None,
    summary: str | None = None,
    source_type: str = "direct_text",
    user_id: str | None = None,
    session_id: str | None = None,
) -> None:
    record = asyncio.run(
        service.upsert_source(
            ContextBucketSourceUpsert(
                source_key=source_key,
                text=text,
                kind=kind,  # type: ignore[arg-type]
                scope=scope,  # type: ignore[arg-type]
                title=title,
                summary=summary,
                source_type=source_type,  # type: ignore[arg-type]
                user_id=user_id,
                session_id=session_id,
            )
        )
    )
    typer.echo(json.dumps(record.model_dump(mode="json"), indent=2, default=str))


@app.command("delete-source")
def delete_source(
    source_key: str,
    scope: str = "session",
    user_id: str | None = None,
    session_id: str | None = None,
) -> None:
    record = asyncio.run(
        service.delete_source(
            ContextBucketSourceDelete(
                source_key=source_key,
                scope=scope,  # type: ignore[arg-type]
                user_id=user_id,
                session_id=session_id,
            )
        )
    )
    typer.echo(json.dumps(record.model_dump(mode="json") if record else None, indent=2, default=str))


@app.command("import-path")
def import_path(
    path: str,
    kind: str = "research_finding",
    scope: str = "session",
    recursive: bool = False,
    user_id: str | None = None,
    session_id: str | None = None,
    data_schema: str | None = typer.Option(default=None, help="Declared import schema JSON or @path to JSON file."),
) -> None:
    parsed_data_schema = _load_json_option(data_schema, option_name="--data-schema")
    response = asyncio.run(
        service.import_path(
            ContextBucketDocumentImportRequest(
                path=path,
                kind=kind,  # type: ignore[arg-type]
                scope=scope,  # type: ignore[arg-type]
                recursive=recursive,
                user_id=user_id,
                session_id=session_id,
                data_schema=(
                    ContextBucketDataSchema.model_validate(parsed_data_schema)
                    if parsed_data_schema is not None
                    else None
                ),
            )
        )
    )
    typer.echo(json.dumps(response.model_dump(mode="json"), indent=2, default=str))


@app.command("retrieve-context")
def retrieve_context(
    query_text: str,
    session_id: str | None = None,
    user_id: str | None = None,
    limit: int = 6,
    intent_data: str | None = typer.Option(default=None, help="Intent JSON or @path to JSON file."),
    intent_schema: str | None = typer.Option(default=None, help="Declared intent schema JSON or @path to JSON file."),
) -> None:
    parsed_intent_data, parsed_intent_schema = _intent_args(intent_data, intent_schema)
    results = asyncio.run(
        service.retrieve_context(
            ContextBucketRetrieveRequest(
                query_text=query_text,
                intent_data=parsed_intent_data,
                intent_schema=parsed_intent_schema,
                session_id=session_id,
                user_id=user_id,
                limit=limit,
            )
        )
    )
    typer.echo(json.dumps(results.model_dump(mode="json"), indent=2, default=str))


@app.command("assemble-context")
def assemble_context(
    query_text: str,
    session_id: str | None = None,
    user_id: str | None = None,
    assembly_mode: str = "assistant",
    token_budget: int = 1200,
    limit: int = 6,
    intent_data: str | None = typer.Option(default=None, help="Intent JSON or @path to JSON file."),
    intent_schema: str | None = typer.Option(default=None, help="Declared intent schema JSON or @path to JSON file."),
) -> None:
    parsed_intent_data, parsed_intent_schema = _intent_args(intent_data, intent_schema)
    response = asyncio.run(
        service.assemble_context(
            ContextBucketAssembleRequest(
                query_text=query_text,
                intent_data=parsed_intent_data,
                intent_schema=parsed_intent_schema,
                session_id=session_id,
                user_id=user_id,
                assembly_mode=assembly_mode,  # type: ignore[arg-type]
                token_budget=token_budget,
                limit=limit,
            )
        )
    )
    typer.echo(json.dumps(response.model_dump(mode="json"), indent=2, default=str))


@app.command("prepare-context")
def prepare_context(
    query_text: str,
    session_id: str | None = None,
    user_id: str | None = None,
    assembly_mode: str = "assistant",
    token_budget: int = 1200,
    limit: int = 6,
    intent_data: str | None = typer.Option(default=None, help="Intent JSON or @path to JSON file."),
    intent_schema: str | None = typer.Option(default=None, help="Declared intent schema JSON or @path to JSON file."),
) -> None:
    parsed_intent_data, parsed_intent_schema = _intent_args(intent_data, intent_schema)
    response = asyncio.run(
        service.prepare_context(
            ContextBucketAssembleRequest(
                query_text=query_text,
                intent_data=parsed_intent_data,
                intent_schema=parsed_intent_schema,
                session_id=session_id,
                user_id=user_id,
                assembly_mode=assembly_mode,  # type: ignore[arg-type]
                token_budget=token_budget,
                limit=limit,
            )
        )
    )
    typer.echo(json.dumps(response.model_dump(mode="json"), indent=2, default=str))


@app.command("prepare-task-envelope")
def prepare_task_envelope(
    query_text: str,
    session_id: str | None = None,
    user_id: str | None = None,
    assembly_mode: str = "assistant",
    token_budget: int = 1200,
    limit: int = 6,
    intent_data: str | None = typer.Option(default=None, help="Intent JSON or @path to JSON file."),
    intent_schema: str | None = typer.Option(default=None, help="Declared intent schema JSON or @path to JSON file."),
) -> None:
    parsed_intent_data, parsed_intent_schema = _intent_args(intent_data, intent_schema)
    response = asyncio.run(
        service.prepare_task_envelope(
            ContextBucketAssembleRequest(
                query_text=query_text,
                intent_data=parsed_intent_data,
                intent_schema=parsed_intent_schema,
                session_id=session_id,
                user_id=user_id,
                assembly_mode=assembly_mode,  # type: ignore[arg-type]
                token_budget=token_budget,
                limit=limit,
            )
        )
    )
    typer.echo(json.dumps(response.model_dump(mode="json"), indent=2, default=str))


@app.command("update-workflow-preference")
def update_workflow_preference(
    scope: str = "user",
    user_id: str | None = None,
    session_id: str | None = None,
    approved_text: str | None = typer.Option(default=None, help="Approved user-authored text to learn from."),
    preference_data: str | None = typer.Option(default=None, help="Preference JSON or @path to JSON file."),
    source_key: str = "user_workflow_preference",
    title: str | None = None,
    summary: str | None = None,
) -> None:
    parsed_preference_data = _load_json_option(preference_data, option_name="--preference-data")
    record = asyncio.run(
        service.update_workflow_preference(
            ContextBucketWorkflowPreferenceUpdateRequest(
                scope=scope,  # type: ignore[arg-type]
                user_id=user_id,
                session_id=session_id,
                source_key=source_key,
                approved_text=approved_text,
                preference_data=parsed_preference_data or {},
                title=title,
                summary=summary,
            )
        )
    )
    typer.echo(json.dumps(record.model_dump(mode="json"), indent=2, default=str))


@app.command("get")
def get(record_id: str) -> None:
    record = asyncio.run(service.get_record(record_id))
    typer.echo(json.dumps(record.model_dump(mode="json") if record else None, indent=2, default=str))


@app.command("list")
def list_records(kind: str | None = None, scope: str | None = None, limit: int = 100) -> None:
    response = asyncio.run(service.list_records(kind=kind, scope=scope, limit=limit))
    typer.echo(json.dumps(response.model_dump(mode="json"), indent=2, default=str))


@app.command("prune")
def prune() -> None:
    response = asyncio.run(service.prune())
    typer.echo(json.dumps(response.model_dump(mode="json"), indent=2, default=str))


@app.command("stats")
def stats() -> None:
    response = asyncio.run(service.stats())
    typer.echo(json.dumps(response.model_dump(mode="json"), indent=2, default=str))


@app.command("export-training")
def export_training() -> None:
    exported_records, output_path = asyncio.run(service.export_training())
    typer.echo(json.dumps({"exported_records": exported_records, "training_path": str(output_path)}, indent=2))


@app.command("benchmark-jsonl")
def benchmark_jsonl(
    dataset_jsonl: Path,
    cases_json: Path,
    data_root: Path = typer.Option(..., help="Private benchmark data root."),
    output_dir: Path = typer.Option(..., help="Directory for benchmark JSON and Markdown reports."),
    suite_name: str = "benchmark",
    limit: int = 6,
    token_budget: int = 1200,
) -> None:
    benchmark_service = ContextBucketService(Settings(data_root=str(data_root)))
    try:
        result = asyncio.run(
            run_jsonl_benchmark(
                service=benchmark_service,
                dataset_jsonl=dataset_jsonl,
                cases_json=cases_json,
                output_dir=output_dir,
                suite_name=suite_name,
                limit=limit,
                token_budget=token_budget,
            )
        )
    except ContextBucketBenchmarkError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(json.dumps(result, indent=2, default=str))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
