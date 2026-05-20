from __future__ import annotations

import json

from typer.testing import CliRunner

from context_bucket import cli
from context_bucket.settings import Settings


runner = CliRunner()


def test_cli_retrieve_context_accepts_inline_intent_json(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    replacement = cli.service.__class__(Settings(data_root=str(tmp_path / "bucket")))
    monkeypatch.setattr("context_bucket.cli.service", replacement)

    replacement_root = tmp_path / "bucket"
    result = runner.invoke(
        cli.app,
        [
            "ingest-source",
            "--text",
            "Acme Legal prefers concise email updates.",
            "--scope",
            "user",
            "--user-id",
            "u1",
            "--source-key",
            "client_card",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert replacement_root.exists()

    result = runner.invoke(
        cli.app,
        [
            "retrieve-context",
            "help me with this request",
            "--user-id",
            "u1",
            "--intent-data",
            json.dumps(
                {
                    "goal": "draft a client update",
                    "tone": "concise",
                    "channel": "email",
                }
            ),
            "--intent-schema",
            json.dumps(
                {
                    "schema_name": "user_intent",
                    "schema_mode": "declared",
                    "root_type": "object",
                    "primary_text_paths": ["goal", "tone", "channel"],
                }
            ),
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["items"]


def test_cli_prepare_context_accepts_intent_json_files(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    replacement = cli.service.__class__(Settings(data_root=str(tmp_path / "bucket")))
    monkeypatch.setattr("context_bucket.cli.service", replacement)

    result = runner.invoke(
        cli.app,
        [
            "ingest-source",
            "--text",
            "The client prefers concise email updates.",
            "--scope",
            "user",
            "--user-id",
            "u1",
            "--source-key",
            "client_note",
        ],
    )
    assert result.exit_code == 0, result.stdout

    intent_data_path = tmp_path / "intent-data.json"
    intent_schema_path = tmp_path / "intent-schema.json"
    intent_data_path.write_text(
        json.dumps({"goal": "draft a client update", "tone": "concise", "channel": "email"}),
        encoding="utf-8",
    )
    intent_schema_path.write_text(
        json.dumps(
            {
                "schema_name": "user_intent",
                "schema_mode": "declared",
                "root_type": "object",
                "primary_text_paths": ["goal", "tone", "channel"],
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        cli.app,
        [
            "prepare-context",
            "help me with this request",
            "--user-id",
            "u1",
            "--assembly-mode",
            "drafting",
            "--intent-data",
            f"@{intent_data_path}",
            "--intent-schema",
            f"@{intent_schema_path}",
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["context_blocks"]


def test_cli_import_path_accepts_declared_data_schema(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    replacement = cli.service.__class__(Settings(data_root=str(tmp_path / "bucket")))
    monkeypatch.setattr("context_bucket.cli.service", replacement)

    json_path = tmp_path / "intent.json"
    schema_path = tmp_path / "schema.json"
    json_path.write_text(
        json.dumps({"goal": "draft a client update", "tone": "concise", "channel": "email"}),
        encoding="utf-8",
    )
    schema_path.write_text(
        json.dumps(
            {
                "schema_name": "user_intent",
                "schema_mode": "declared",
                "root_type": "object",
                "primary_text_paths": ["goal", "tone", "channel"],
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        cli.app,
        [
            "import-path",
            str(json_path),
            "--scope",
            "user",
            "--user-id",
            "u1",
            "--data-schema",
            f"@{schema_path}",
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    record_id = payload["items"][0]["record_id"]

    result = runner.invoke(cli.app, ["get", record_id])
    assert result.exit_code == 0, result.stdout
    record = json.loads(result.stdout)
    assert record["data_schema"]["schema_name"] == "user_intent"
    assert record["data_schema"]["schema_mode"] == "declared"


def test_cli_prepare_task_envelope(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    replacement = cli.service.__class__(Settings(data_root=str(tmp_path / "bucket")))
    monkeypatch.setattr("context_bucket.cli.service", replacement)

    result = runner.invoke(
        cli.app,
        [
            "ingest-source",
            "--scope",
            "user",
            "--user-id",
            "u1",
            "--kind",
            "user_profile_note",
            "--source-key",
            "workflow_prefs",
            "--text",
            "Prefer concise outputs. Prefer best effort over clarification. Include clear next steps.",
        ],
    )
    assert result.exit_code == 0, result.stdout

    result = runner.invoke(
        cli.app,
        [
            "ingest-source",
            "--scope",
            "user",
            "--user-id",
            "u1",
            "--source-key",
            "email_1",
            "--text",
            "Interview invitation email asking for a reply with availability.",
        ],
    )
    assert result.exit_code == 0, result.stdout

    result = runner.invoke(
        cli.app,
        [
            "prepare-task-envelope",
            "answer the email",
            "--user-id",
            "u1",
            "--assembly-mode",
            "drafting",
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["workflow_intent"]["workflow_type"] == "reply"
    assert payload["user_workflow_preference"]["brevity_preference"] == "high"


def test_cli_update_workflow_preference(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    replacement = cli.service.__class__(Settings(data_root=str(tmp_path / "bucket")))
    monkeypatch.setattr("context_bucket.cli.service", replacement)

    result = runner.invoke(
        cli.app,
        [
            "update-workflow-preference",
            "--user-id",
            "u1",
            "--approved-text",
            "Prefer concise outputs. Include clear next steps.",
            "--preference-data",
            json.dumps({"clarification_preference": "low"}),
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["data_schema"]["schema_name"] == "user_workflow_preference"
    assert payload["structured_data"]["brevity_preference"] == "high"
    assert payload["structured_data"]["clarification_preference"] == "low"


def test_cli_benchmark_jsonl_writes_reports(tmp_path) -> None:  # type: ignore[no-untyped-def]
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "source_key": "profile",
                        "scope": "user",
                        "user_id": "u1",
                        "kind": "user_profile_note",
                        "text": "Client ACME prefers concise Friday status updates.",
                    }
                ),
                json.dumps(
                    {
                        "source_key": "matter_note",
                        "scope": "session",
                        "session_id": "s1",
                        "text": "ACME-17 discovery is ongoing and legal review is due Friday.",
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(
        json.dumps(
            {
                "name": "private_export",
                "cases": [
                    {
                        "name": "client_update",
                        "query_text": "draft a concise ACME-17 client update",
                        "user_id": "u1",
                        "session_id": "s1",
                        "include_user_scope": True,
                        "expected_source_keys": ["profile", "matter_note"],
                        "expected_terms": ["ACME-17"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "reports"

    result = runner.invoke(
        cli.app,
        [
            "benchmark-jsonl",
            str(dataset_path),
            str(cases_path),
            "--data-root",
            str(tmp_path / "bucket"),
            "--output-dir",
            str(output_dir),
            "--suite-name",
            "private_export",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["imported_records"] == 2
    assert payload["retrieval_hits"] == 1
    assert payload["assembly_hits"] == 1
    assert payload["behavior_evidence"]["evidence_type"] == "benchmark_run"
    assert payload["behavior_evidence"]["suite_name"] == "private_export"
    assert payload["behavior_evidence"]["case_evidence"][0]["name"] == "client_update"
    assert payload["behavior_evidence"]["case_evidence"][0]["expected_terms_scope"] == "assembled_context"
    assert payload["behavior_evidence"]["case_evidence"][0]["matched_expected_terms"] == ["ACME-17"]
    assert (output_dir / "benchmark-result.json").exists()
    report = (output_dir / "benchmark-report.md").read_text(encoding="utf-8")
    assert "Context Bucket Benchmark" in report
    assert "Behavior Evidence" in report
    assert "Evidence: client_update" in report
    assert "Expected terms scope: assembled_context" in report


def test_cli_benchmark_jsonl_rejects_missing_source_key(tmp_path) -> None:  # type: ignore[no-untyped-def]
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(json.dumps({"text": "missing source key"}) + "\n", encoding="utf-8")
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(json.dumps({"name": "bad", "cases": []}), encoding="utf-8")

    result = runner.invoke(
        cli.app,
        [
            "benchmark-jsonl",
            str(dataset_path),
            str(cases_path),
            "--data-root",
            str(tmp_path / "bucket"),
            "--output-dir",
            str(tmp_path / "reports"),
        ],
    )

    assert result.exit_code != 0
    assert "missing required source_key" in result.output
