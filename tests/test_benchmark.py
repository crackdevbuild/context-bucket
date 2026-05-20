from __future__ import annotations

import json

import pytest

from context_bucket.benchmark import ContextBucketBenchmarkError, load_jsonl_dataset


def test_load_jsonl_dataset_accepts_text_and_structured_records(tmp_path) -> None:  # type: ignore[no-untyped-def]
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                json.dumps({"source_key": "note_1", "text": "plain text note"}),
                json.dumps({"source_key": "profile", "structured_data": {"client": "ACME"}}),
            ]
        ),
        encoding="utf-8",
    )

    records = load_jsonl_dataset(dataset_path)

    assert [record["source_key"] for record in records] == ["note_1", "profile"]


def test_load_jsonl_dataset_rejects_missing_source_key(tmp_path) -> None:  # type: ignore[no-untyped-def]
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(json.dumps({"text": "missing source key"}), encoding="utf-8")

    with pytest.raises(ContextBucketBenchmarkError, match="missing required source_key"):
        load_jsonl_dataset(dataset_path)


def test_load_jsonl_dataset_rejects_missing_text_and_structured_data(tmp_path) -> None:  # type: ignore[no-untyped-def]
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(json.dumps({"source_key": "empty"}), encoding="utf-8")

    with pytest.raises(ContextBucketBenchmarkError, match="text or structured_data"):
        load_jsonl_dataset(dataset_path)
