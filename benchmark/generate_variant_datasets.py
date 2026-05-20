from __future__ import annotations

import json
from pathlib import Path

BENCHMARK_ROOT = Path(__file__).resolve().parent
DATASETS_DIR = BENCHMARK_ROOT / "datasets"

SOURCE_FILES = [
    "base_dataset.jsonl",
    "perturbed_dataset.jsonl",
    "medical_dataset.jsonl",
    "adversarial_dataset.jsonl",
]


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def make_structured_only(records: list[dict]) -> list[dict]:
    out: list[dict] = []
    for rec in records:
        if rec.get("structured_data") is not None:
            out.append(rec)
    return out


def make_unstructured_only(records: list[dict]) -> list[dict]:
    out: list[dict] = []
    for rec in records:
        rec2 = {k: v for k, v in rec.items() if k not in ("structured_data", "data_schema")}
        out.append(rec2)
    return out


def main() -> None:
    for source_name in SOURCE_FILES:
        source_path = DATASETS_DIR / source_name
        if not source_path.exists():
            print(f"SKIP {source_name} — not found")
            continue

        records = load_jsonl(source_path)
        base_name = source_name.replace("_dataset.jsonl", "")

        structured = make_structured_only(records)
        unstructured = make_unstructured_only(records)

        struct_path = DATASETS_DIR / f"{base_name}_structured.jsonl"
        unstruct_path = DATASETS_DIR / f"{base_name}_unstructured.jsonl"

        write_jsonl(structured, struct_path)
        write_jsonl(unstructured, unstruct_path)

        print(f"{source_name}: {len(records)} total -> {len(structured)} structured, {len(unstructured)} unstructured")

    print("Variant datasets generated.")


if __name__ == "__main__":
    main()
