from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

BENCHMARK_ROOT = Path(__file__).resolve().parent
DATASETS_DIR = BENCHMARK_ROOT / "datasets"
CASES_DIR = BENCHMARK_ROOT / "cases"
RESULTS_DIR = BENCHMARK_ROOT / "results"

CLI_CMD = [sys.executable, "-m", "context_bucket.cli", "benchmark-jsonl"]


def _dataset_map_for_variant(variant: str) -> dict[str, dict[int, Path]]:
    if variant == "structured":
        return {
            "series_a": {
                **{n: DATASETS_DIR / "base_structured.jsonl" for n in range(1, 26)},
                **{n: DATASETS_DIR / "perturbed_structured.jsonl" for n in range(26, 51)},
            },
            "series_b": {
                **{n: DATASETS_DIR / "medical_structured.jsonl" for n in range(1, 26)},
                **{n: DATASETS_DIR / "medical_structured.jsonl" for n in range(26, 51)},
            },
            "series_c": {
                **{n: DATASETS_DIR / "base_structured.jsonl" for n in range(1, 26)},
                **{n: DATASETS_DIR / "adversarial_structured.jsonl" for n in range(26, 51)},
            },
        }
    if variant == "unstructured":
        return {
            "series_a": {
                **{n: DATASETS_DIR / "base_unstructured.jsonl" for n in range(1, 26)},
                **{n: DATASETS_DIR / "perturbed_unstructured.jsonl" for n in range(26, 51)},
            },
            "series_b": {
                **{n: DATASETS_DIR / "medical_unstructured.jsonl" for n in range(1, 26)},
                **{n: DATASETS_DIR / "medical_unstructured.jsonl" for n in range(26, 51)},
            },
            "series_c": {
                **{n: DATASETS_DIR / "base_unstructured.jsonl" for n in range(1, 26)},
                **{n: DATASETS_DIR / "adversarial_unstructured.jsonl" for n in range(26, 51)},
            },
        }
    return {
        "series_a": {
            **{n: DATASETS_DIR / "base_dataset.jsonl" for n in range(1, 26)},
            **{n: DATASETS_DIR / "perturbed_dataset.jsonl" for n in range(26, 51)},
        },
        "series_b": {
            **{n: DATASETS_DIR / "medical_dataset.jsonl" for n in range(1, 26)},
            **{n: DATASETS_DIR / "medical_dataset.jsonl" for n in range(26, 51)},
        },
        "series_c": {
            **{n: DATASETS_DIR / "base_dataset.jsonl" for n in range(1, 26)},
            **{n: DATASETS_DIR / "adversarial_dataset.jsonl" for n in range(26, 51)},
        },
    }


RUN_TIMEOUT = 3600


def _results_dir_for_variant(variant: str) -> Path:
    if variant == "default":
        return RESULTS_DIR
    return RESULTS_DIR / variant


def _summary_file_for_variant(variant: str) -> Path:
    if variant == "default":
        return BENCHMARK_ROOT / "run_summary.json"
    return BENCHMARK_ROOT / f"run_summary_{variant}.json"


def _data_root_for_variant(variant: str, series: str, run_num: int) -> Path:
    if variant == "default":
        return Path(f"/tmp/cb-bm-{series}-{run_num:02d}")
    return Path(f"/tmp/cb-bm-{variant}-{series}-{run_num:02d}")


def _load_existing_summaries(summary_file: Path) -> dict[str, dict]:
    existing: dict[str, dict] = {}
    if summary_file.exists():
        try:
            for item in json.loads(summary_file.read_text(encoding="utf-8")):
                key = f"{item.get('series')}_{item.get('run'):02d}" if isinstance(item.get("run"), int) else ""
                if key:
                    existing[key] = item
        except (json.JSONDecodeError, KeyError):
            pass
    return existing


def _check_output_dir(results_dir: Path, series: str, run_num: int) -> dict | None:
    output_dir = results_dir / series / f"run_{run_num:02d}"
    result_file = output_dir / "benchmark-result.json"
    if result_file.exists():
        try:
            data = json.loads(result_file.read_text(encoding="utf-8"))
            return {
                "series": series,
                "run": run_num,
                "suite_name": data.get("suite_name", f"{series}_run_{run_num:02d}"),
                "elapsed_wall": data.get("runtime_seconds", 0),
                "exit_code": 0,
                "retrieval_hit_rate": data.get("retrieval_hit_rate"),
                "assembly_hit_rate": data.get("assembly_hit_rate"),
                "runtime_seconds": data.get("runtime_seconds"),
                "imported_records": data.get("imported_records"),
                "skipped_count": len(data.get("skipped_records", [])),
                "total_cases": data.get("total_cases"),
                "retrieval_hits": data.get("retrieval_hits"),
                "assembly_hits": data.get("assembly_hits"),
            }
        except (json.JSONDecodeError, KeyError):
            pass
    return None


def run_one(
    series: str,
    run_num: int,
    dataset_path: Path,
    cases_path: Path,
    *,
    variant: str,
    embedding_backend: str,
    embedding_dimensions: int,
    results_dir: Path,
) -> dict:
    data_root = _data_root_for_variant(variant, series, run_num)
    output_dir = results_dir / series / f"run_{run_num:02d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)

    suite_name = f"{series}_run_{run_num:02d}"
    cmd = [
        *CLI_CMD,
        str(dataset_path),
        str(cases_path),
        "--data-root", str(data_root),
        "--output-dir", str(output_dir),
        "--suite-name", suite_name,
        "--embedding-backend", embedding_backend,
        "--embedding-dimensions", str(embedding_dimensions),
    ]

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=RUN_TIMEOUT)
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - t0
        result_data: dict = {"series": series, "run": run_num, "suite_name": suite_name, "elapsed_wall": round(elapsed, 3), "exit_code": -1, "error": f"timeout after {RUN_TIMEOUT}s"}
        print(f" TIMEOUT {series} run_{run_num:02d} ({RUN_TIMEOUT}s)", file=sys.stderr)
        return result_data

    elapsed = time.perf_counter() - t0
    result_data = {"series": series, "run": run_num, "suite_name": suite_name, "elapsed_wall": round(elapsed, 3), "exit_code": proc.returncode, "variant": variant}

    if proc.returncode != 0:
        result_data["error"] = proc.stderr[-2000:] if proc.stderr else "unknown error"
        print(f" FAIL {series} run_{run_num:02d} (exit {proc.returncode})", file=sys.stderr)
    else:
        try:
            parsed = json.loads(proc.stdout)
            result_data["retrieval_hit_rate"] = parsed.get("retrieval_hit_rate")
            result_data["assembly_hit_rate"] = parsed.get("assembly_hit_rate")
            result_data["runtime_seconds"] = parsed.get("runtime_seconds")
            result_data["imported_records"] = parsed.get("imported_records")
            result_data["skipped_count"] = len(parsed.get("skipped_records", []))
            result_data["total_cases"] = parsed.get("total_cases")
            result_data["retrieval_hits"] = parsed.get("retrieval_hits")
            result_data["assembly_hits"] = parsed.get("assembly_hits")
        except json.JSONDecodeError:
            result_data["error"] = "could not parse CLI stdout"
    print(f"  OK {variant} {series} run_{run_num:02d} ret={result_data.get('retrieval_hit_rate', '?')} asm={result_data.get('assembly_hit_rate', '?')} {elapsed:.1f}s")

    if data_root.exists():
        shutil.rmtree(data_root, ignore_errors=True)

    return result_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 150 benchmark workflows")
    parser.add_argument("--variant", choices=["default", "structured", "unstructured"], default="default", help="Dataset variant to use")
    parser.add_argument("--embedding-backend", default="onnx_minilm", help="Embedding backend for the service")
    parser.add_argument("--embedding-dimensions", type=int, default=384, help="Embedding dimensions")
    parser.add_argument("--force", action="store_true", help="Force re-run even if cached results exist")
    args = parser.parse_args()

    variant = args.variant
    embedding_backend = args.embedding_backend
    embedding_dimensions = args.embedding_dimensions

    dataset_map = _dataset_map_for_variant(variant)
    results_dir = _results_dir_for_variant(variant)
    summary_file = _summary_file_for_variant(variant)

    results_dir.mkdir(parents=True, exist_ok=True)

    all_series = ["series_a", "series_b", "series_c"]
    total = 150

    existing = {} if args.force else _load_existing_summaries(summary_file)
    print(f"Found {len(existing)} previously completed runs (variant={variant}).")

    ordered_runs: list[tuple[str, int]] = []
    for series in all_series:
        for run_num in range(1, 51):
            ordered_runs.append((series, run_num))

    print(f"Starting {total} benchmark runs (variant={variant}, backend={embedding_backend}, dim={embedding_dimensions}, timeout={RUN_TIMEOUT}s per run) ...")
    overall_t0 = time.perf_counter()
    summaries: list[dict] = []
    completed = 0
    skipped = 0

    for series, run_num in ordered_runs:
        key = f"{series}_{run_num:02d}"
        if not args.force and key in existing:
            summaries.append(existing[key])
            skipped += 1
            continue

        output_result = _check_output_dir(results_dir, series, run_num) if not args.force else None
        if output_result is not None:
            summaries.append(output_result)
            existing[key] = output_result
            skipped += 1
            print(f" CACHED {series} run_{run_num:02d}")
            continue

        dataset_path = dataset_map[series][run_num]
        cases_path = CASES_DIR / series / f"run_{run_num:02d}_cases.json"
        if not cases_path.exists():
            print(f" SKIP {series} run_{run_num:02d} — cases file missing", file=sys.stderr)
            continue
        if not dataset_path.exists():
            print(f" SKIP {series} run_{run_num:02d} — dataset missing ({dataset_path})", file=sys.stderr)
            continue

        summary = run_one(
            series, run_num, dataset_path, cases_path,
            variant=variant,
            embedding_backend=embedding_backend,
            embedding_dimensions=embedding_dimensions,
            results_dir=results_dir,
        )
        summaries.append(summary)
        existing[key] = summary
        completed += 1

        if completed % 5 == 0:
            summary_file.write_text(json.dumps(summaries, indent=2, default=str), encoding="utf-8")

        if (skipped + completed) % 10 == 0:
            elapsed_total = time.perf_counter() - overall_t0
            done = skipped + completed
            rate = completed / elapsed_total if elapsed_total > 0 else 0
            remaining = total - done
            eta = remaining / rate * (completed / max(done, 1)) if rate > 0 and done > 0 else 0
            print(f" Progress: {done}/{total} ({completed} new, {skipped} cached, ETA {eta/60:.1f} min)")

    summary_file.write_text(json.dumps(summaries, indent=2, default=str), encoding="utf-8")
    total_elapsed = time.perf_counter() - overall_t0
    ok_count = sum(1 for s in summaries if s.get("exit_code") == 0)
    fail_count = sum(1 for s in summaries if s.get("exit_code") != 0)
    print(f"\nDone. {ok_count} ok, {fail_count} failed, {total_elapsed/60:.1f} min total.")
    print(f"Summary written to {summary_file}")


if __name__ == "__main__":
    main()
