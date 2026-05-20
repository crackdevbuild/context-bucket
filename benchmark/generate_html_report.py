from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

BENCHMARK_ROOT = Path(__file__).resolve().parent
REPO_ROOT = BENCHMARK_ROOT.parent
RESULTS_DIR = BENCHMARK_ROOT / "results"
CASES_DIR = BENCHMARK_ROOT / "cases"
HTML_PARTS = BENCHMARK_ROOT / "html_parts"
DATASETS_DIR = BENCHMARK_ROOT / "datasets"

DIFFICULTY_TIERS = [
    "trivial", "easy", "moderate_easy", "moderate",
    "moderate_hard", "hard", "harder", "very_hard", "extreme", "maximum",
]

TIER_CONFIG = {
    "trivial": {"runs": (1, 5), "token_budget": 2000, "num_cases": (3, 5), "include_user": True, "include_global": True, "terms_scope": "assembled_context"},
    "easy": {"runs": (6, 10), "token_budget": 1500, "num_cases": (5, 8), "include_user": True, "include_global": True, "terms_scope": "assembled_context"},
    "moderate_easy": {"runs": (11, 15), "token_budget": 1200, "num_cases": (8, 12), "include_user": True, "include_global": True, "terms_scope": "assembled_context"},
    "moderate": {"runs": (16, 20), "token_budget": 900, "num_cases": (12, 16), "include_user": True, "include_global": True, "terms_scope": "assembled_context"},
    "moderate_hard": {"runs": (21, 25), "token_budget": 700, "num_cases": (16, 20), "include_user": True, "include_global": True, "terms_scope": "assembled_context"},
    "hard": {"runs": (26, 30), "token_budget": 600, "num_cases": (20, 24), "include_user": True, "include_global": True, "terms_scope": "assembled_context"},
    "harder": {"runs": (31, 35), "token_budget": 500, "num_cases": (24, 28), "include_user": False, "include_global": False, "terms_scope": "retrieved_records"},
    "very_hard": {"runs": (36, 40), "token_budget": 400, "num_cases": (28, 32), "include_user": False, "include_global": False, "terms_scope": "retrieved_records"},
    "extreme": {"runs": (41, 45), "token_budget": 300, "num_cases": (32, 36), "include_user": False, "include_global": False, "terms_scope": "retrieved_records"},
    "maximum": {"runs": (46, 50), "token_budget": 200, "num_cases": (36, 40), "include_user": False, "include_global": False, "terms_scope": "retrieved_records"},
}

TIER_COLORS = {
    "trivial": "#3b82f6", "easy": "#6366f1", "moderate_easy": "#8b5cf6",
    "moderate": "#a855f7", "moderate_hard": "#c026d3", "hard": "#d946ef",
    "harder": "#ec4899", "very_hard": "#f43f5e", "extreme": "#ef4444", "maximum": "#b91c1c",
}

SERIES_LABELS = {"series_a": "Series A - Shuffled/Perturbed", "series_b": "Series B - Medical Domain", "series_c": "Series C - Adversarial"}


def _dataset_map_for_variant(variant: str) -> dict[str, dict[int, str]]:
    if variant == "structured":
        return {
            "series_a": {**{n: "base_structured.jsonl" for n in range(1, 26)}, **{n: "perturbed_structured.jsonl" for n in range(26, 51)}},
            "series_b": {**{n: "medical_structured.jsonl" for n in range(1, 51)}},
            "series_c": {**{n: "base_structured.jsonl" for n in range(1, 26)}, **{n: "adversarial_structured.jsonl" for n in range(26, 51)}},
        }
    if variant == "unstructured":
        return {
            "series_a": {**{n: "base_unstructured.jsonl" for n in range(1, 26)}, **{n: "perturbed_unstructured.jsonl" for n in range(26, 51)}},
            "series_b": {**{n: "medical_unstructured.jsonl" for n in range(1, 51)}},
            "series_c": {**{n: "base_unstructured.jsonl" for n in range(1, 26)}, **{n: "adversarial_unstructured.jsonl" for n in range(26, 51)}},
        }
    return {
        "series_a": {**{n: "base_dataset.jsonl" for n in range(1, 26)}, **{n: "perturbed_dataset.jsonl" for n in range(26, 51)}},
        "series_b": {**{n: "medical_dataset.jsonl" for n in range(1, 51)}},
        "series_c": {**{n: "base_dataset.jsonl" for n in range(1, 26)}, **{n: "adversarial_dataset.jsonl" for n in range(26, 51)}},
    }


def _pipeline_config_for_variant(variant: str) -> dict:
    base: dict[str, Any] = {
        "embedding": {
            "backend": "onnx_minilm",
            "model": "sentence-transformers/all-MiniLM-L6-v2 (ONNX)",
            "dimensions": 384,
            "method": "ONNX Runtime inference with AutoTokenizer — mean-pool over token embeddings (attention-mask weighted), L2-normalize to unit vector; truncation + renormalization if dimensions != 384",
            "normalization": "L2 (unit vector)",
        },
        "text_processing": {
            "lexical_tokenization": "regex [a-z0-9]{2,}",
            "stemming": "suffix stripping (ingly, edly, ment, tion, ions, ness, able, ible, ing, ers, ies, ied, ed, es, s) with ies/ied→y",
            "synonym_expansion": {
                "brief": ["concise", "short"],
                "update": ["status", "summary"],
                "email": ["mail", "draft"],
                "client": ["customer", "matter"],
                "case": ["matter"],
                "redline": ["revision", "edit"],
            },
            "char_ngrams": "4-grams with ^$ padding (^token$)",
        },
        "chunking": {
            "chunk_chars": 900,
            "chunk_overlap_chars": 120,
            "per_chunk_fields": ["text", "lexical_tokens", "semantic_terms", "embedding", "token_count_estimate"],
        },
        "retrieval": {
            "strategy": "embedding_lexical_rerank",
            "query_top_k": 6,
            "semantic_candidate_multiplier": 4,
            "lexical_candidate_multiplier": 4,
            "pool_selection": "semantic top-(K×4) and lexical top-(K×4) merged, then reranked",
            "search_type": "brute-force (scan all chunks in index, no ANN)",
        },
        "scoring_weights": {
            "semantic_score": 0.60,
            "lexical_score": 0.25,
            "keyword_bonus": 0.08,
            "metadata_bonus": 0.07,
            "record_rank_bonus_research": 0.15,
            "record_rank_bonus_decision": 0.08,
            "scope_priority_session": 0.12,
            "scope_priority_user": 0.08,
            "scope_priority_global": 0.03,
            "age_decay_range": "+0.05 to -0.25",
        },
        "dedup": {
            "method": "token set similarity (Jaccard)",
            "threshold": 0.82,
        },
        "index_storage": {
            "record_backend": "file",
            "index_backend": "json",
            "record_index": "records.json",
            "chunk_index": "chunks.jsonl (includes per-chunk embedding vectors)",
            "alternate_backend": "sqlite (record_index + chunk_index tables with payload_json)",
        },
        "assembly": {
            "flow": "retrieve → compress (dedup ≥0.82) → fill token budget → section ordering → render context",
            "modes": ["planner", "research", "drafting", "default"],
        },
        "variant": variant,
    }
    if variant == "structured":
        base["variant_note"] = "Dataset contains only records with structured_data fields; structured_data and data_schema included in ingestion"
    elif variant == "unstructured":
        base["variant_note"] = "Dataset contains only unstructured text records; structured_data and data_schema fields stripped before ingestion"
    return base


def tier_for_run(run_num: int) -> str:
    for name, cfg in TIER_CONFIG.items():
        if cfg["runs"][0] <= run_num <= cfg["runs"][1]:
            return name
    return "maximum"


def load_all_data(summary_file: Path, results_dir: Path, variant: str) -> list[dict[str, Any]]:
    dataset_map = _dataset_map_for_variant(variant)
    summaries = json.loads(summary_file.read_text(encoding="utf-8"))
    runs: list[dict[str, Any]] = []
    for s in summaries:
        series = s["series"]
        run_num = s["run"]
        tier = tier_for_run(run_num)
        result_path = results_dir / series / f"run_{run_num:02d}" / "benchmark-result.json"
        cases_path = CASES_DIR / series / f"run_{run_num:02d}_cases.json"
        run_data: dict[str, Any] = {
            **s,
            "tier": tier,
            "dataset": dataset_map.get(series, {}).get(run_num, "unknown"),
        }
        if result_path.exists():
            try:
                br = json.loads(result_path.read_text(encoding="utf-8"))
                run_data["run_id"] = br.get("run_id")
                run_data["stats"] = br.get("stats", {})
                run_data["evaluation_results"] = br.get("evaluation", {}).get("results", [])
                ev_results = run_data["evaluation_results"]
                ret_counts = [e.get("retrieval_count", 0) for e in ev_results if e.get("retrieval_count") is not None]
                asm_tokens = [e.get("assembly_token_count_estimate", 0) for e in ev_results if e.get("assembly_token_count_estimate") is not None]
                ret_hits = sum(1 for e in ev_results if e.get("retrieval_hit"))
                asm_hits = sum(1 for e in ev_results if e.get("assembly_hit"))
                run_data["retrieval_stats"] = {
                    "total_cases": len(ev_results),
                    "retrieval_hits": ret_hits,
                    "assembly_hits": asm_hits,
                    "mean_retrieval_count": round(statistics.mean(ret_counts), 1) if ret_counts else 0,
                    "min_retrieval_count": min(ret_counts) if ret_counts else 0,
                    "max_retrieval_count": max(ret_counts) if ret_counts else 0,
                    "mean_assembly_tokens": round(statistics.mean(asm_tokens), 0) if asm_tokens else 0,
                    "total_assembly_tokens": sum(asm_tokens),
                }
            except Exception:
                pass
        if cases_path.exists():
            try:
                cases_data = json.loads(cases_path.read_text(encoding="utf-8"))
                run_data["case_definitions"] = cases_data.get("cases", [])
            except Exception:
                pass
        runs.append(run_data)
    return runs


def compute_summary_stats(runs: list[dict]) -> dict[str, Any]:
    ok = [r for r in runs if r.get("exit_code") == 0]
    ret_rates = [r["retrieval_hit_rate"] for r in ok if r.get("retrieval_hit_rate") is not None]
    asm_rates = [r["assembly_hit_rate"] for r in ok if r.get("assembly_hit_rate") is not None]
    runtimes = [r["runtime_seconds"] for r in ok if r.get("runtime_seconds") is not None]
    total_wall = sum(r.get("elapsed_wall", 0) for r in ok)
    return {
        "total_runs": len(runs),
        "successful": len(ok),
        "failed": len(runs) - len(ok),
        "retrieval_mean": statistics.mean(ret_rates) if ret_rates else 0,
        "retrieval_median": statistics.median(ret_rates) if ret_rates else 0,
        "assembly_mean": statistics.mean(asm_rates) if asm_rates else 0,
        "assembly_median": statistics.median(asm_rates) if asm_rates else 0,
        "runtime_mean": statistics.mean(runtimes) if runtimes else 0,
        "runtime_median": statistics.median(runtimes) if runtimes else 0,
        "total_wall_seconds": total_wall,
        "total_wall_hours": round(total_wall / 3600, 1),
    }


def compute_tier_stats(runs: list[dict], series: str | None = None) -> dict[str, dict]:
    filtered = runs if series is None else [r for r in runs if r.get("series") == series]
    ok = [r for r in filtered if r.get("exit_code") == 0]
    tiers: dict[str, dict] = {}
    for tier_name in DIFFICULTY_TIERS:
        tier_runs = [r for r in ok if r.get("tier") == tier_name]
        ret = [r["retrieval_hit_rate"] for r in tier_runs if r.get("retrieval_hit_rate") is not None]
        asm = [r["assembly_hit_rate"] for r in tier_runs if r.get("assembly_hit_rate") is not None]
        rt = [r["runtime_seconds"] for r in tier_runs if r.get("runtime_seconds") is not None]
        tiers[tier_name] = {
            "count": len(tier_runs),
            "retrieval_mean": round(statistics.mean(ret), 4) if ret else 0,
            "retrieval_min": round(min(ret), 4) if ret else 0,
            "retrieval_max": round(max(ret), 4) if ret else 0,
            "assembly_mean": round(statistics.mean(asm), 4) if asm else 0,
            "assembly_min": round(min(asm), 4) if asm else 0,
            "assembly_max": round(max(asm), 4) if asm else 0,
            "runtime_mean": round(statistics.mean(rt), 1) if rt else 0,
        }
    return tiers


def analyze_datasets(variant: str) -> list[dict[str, Any]]:
    if variant == "structured":
        dataset_files = {
            "base_structured.jsonl": {"label": "Base Structured (Legal/Business)", "used_in": "Series A (runs 1-25), Series C (runs 1-25)"},
            "perturbed_structured.jsonl": {"label": "Perturbed Structured (Shuffled/Shortened)", "used_in": "Series A (runs 26-50)"},
            "medical_structured.jsonl": {"label": "Medical Structured (Healthcare Domain)", "used_in": "Series B (runs 1-50)"},
            "adversarial_structured.jsonl": {"label": "Adversarial Structured (Distractors/Near-Miss)", "used_in": "Series C (runs 26-50)"},
        }
    elif variant == "unstructured":
        dataset_files = {
            "base_unstructured.jsonl": {"label": "Base Unstructured (Legal/Business)", "used_in": "Series A (runs 1-25), Series C (runs 1-25)"},
            "perturbed_unstructured.jsonl": {"label": "Perturbed Unstructured (Shuffled/Shortened)", "used_in": "Series A (runs 26-50)"},
            "medical_unstructured.jsonl": {"label": "Medical Unstructured (Healthcare Domain)", "used_in": "Series B (runs 1-50)"},
            "adversarial_unstructured.jsonl": {"label": "Adversarial Unstructured (Distractors/Near-Miss)", "used_in": "Series C (runs 26-50)"},
        }
    else:
        dataset_files = {
            "base_dataset.jsonl": {"label": "Base (Legal/Business)", "used_in": "Series A (runs 1-25), Series C (runs 1-25)"},
            "perturbed_dataset.jsonl": {"label": "Perturbed (Shuffled/Shortened)", "used_in": "Series A (runs 26-50)"},
            "medical_dataset.jsonl": {"label": "Medical (Healthcare Domain)", "used_in": "Series B (runs 1-50)"},
            "adversarial_dataset.jsonl": {"label": "Adversarial (Distractors/Near-Miss)", "used_in": "Series C (runs 26-50)"},
        }
    datasets_info: list[dict[str, Any]] = []
    for fname, meta in dataset_files.items():
        fpath = DATASETS_DIR / fname
        if not fpath.exists():
            continue
        records = []
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        source_prefixes: set[str] = set()
        kinds: dict[str, int] = {}
        scopes: dict[str, int] = {}
        text_lengths: list[int] = []
        has_structured_data = 0
        for rec in records:
            sk = rec.get("source_key", "")
            prefix = sk.split("_")[0] if "_" in sk else sk[:8]
            source_prefixes.add(prefix)
            kind = rec.get("kind", "unknown")
            kinds[kind] = kinds.get(kind, 0) + 1
            scope = rec.get("scope", "unknown")
            scopes[scope] = scopes.get(scope, 0) + 1
            text = rec.get("text", "")
            text_lengths.append(len(text))
            if rec.get("structured_data"):
                has_structured_data += 1
        datasets_info.append({
            "filename": fname,
            "label": meta["label"],
            "used_in": meta["used_in"],
            "record_count": len(records),
            "source_prefix_count": len(source_prefixes),
            "source_prefixes": sorted(source_prefixes),
            "kinds": kinds,
            "scopes": scopes,
            "avg_text_length": round(sum(text_lengths) / len(text_lengths), 1) if text_lengths else 0,
            "min_text_length": min(text_lengths) if text_lengths else 0,
            "max_text_length": max(text_lengths) if text_lengths else 0,
            "median_text_length": round(statistics.median(text_lengths), 1) if text_lengths else 0,
            "has_structured_data_count": has_structured_data,
            "fields": sorted(set(k for rec in records for k in rec.keys())),
        })
    return datasets_info


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate HTML benchmark report")
    parser.add_argument("--variant", choices=["default", "structured", "unstructured"], default="default", help="Dataset variant")
    args = parser.parse_args()

    variant = args.variant

    if variant == "default":
        summary_file = BENCHMARK_ROOT / "run_summary.json"
        results_dir = RESULTS_DIR
        output_file = REPO_ROOT / "benchmark-150run-report.html"
        title = "Context Bucket - 150-Run Benchmark Report"
    else:
        summary_file = BENCHMARK_ROOT / f"run_summary_{variant}.json"
        results_dir = RESULTS_DIR / variant
        output_file = REPO_ROOT / f"benchmark-150run-report-{variant}.html"
        title = f"Context Bucket - 150-Run Benchmark Report ({variant})"

    if not summary_file.exists():
        print(f"Summary file not found: {summary_file}")
        return

    print(f"Loading all benchmark data (variant={variant})...")
    runs = load_all_data(summary_file, results_dir, variant)
    summary = compute_summary_stats(runs)
    tier_stats = compute_tier_stats(runs)
    series_tier_stats = {s: compute_tier_stats(runs, s) for s in ["series_a", "series_b", "series_c"]}
    datasets_info = analyze_datasets(variant)
    pipeline_config = _pipeline_config_for_variant(variant)

    css = (HTML_PARTS / "css.html").read_text(encoding="utf-8")
    template = (HTML_PARTS / "template.html").read_text(encoding="utf-8")

    html = template.replace("{{CSS}}", css)
    html = html.replace("{{TITLE}}", title)
    html = html.replace("{{VARIANT}}", variant)
    html = html.replace("{{RUNS_JSON}}", json.dumps(runs, default=str))
    html = html.replace("{{TIER_CONFIG_JSON}}", json.dumps(TIER_CONFIG))
    html = html.replace("{{TIER_COLORS_JSON}}", json.dumps(TIER_COLORS))
    html = html.replace("{{SUMMARY_JSON}}", json.dumps(summary))
    html = html.replace("{{TIER_STATS_JSON}}", json.dumps(tier_stats))
    html = html.replace("{{SERIES_TIER_STATS_JSON}}", json.dumps(series_tier_stats))
    html = html.replace("{{TIERS_JSON}}", json.dumps(DIFFICULTY_TIERS))
    html = html.replace("{{SERIES_LABELS_JSON}}", json.dumps(SERIES_LABELS))
    html = html.replace("{{DATASETS_JSON}}", json.dumps(datasets_info, default=str))
    html = html.replace("{{PIPELINE_JSON}}", json.dumps(pipeline_config))
    html = html.replace("{{TIMESTAMP}}", time.strftime("%Y-%m-%d %H:%M:%S"))
    html = html.replace("{{TIMESTAMP_SHORT}}", time.strftime("%Y-%m-%d %H:%M"))

    output_file.write_text(html, encoding="utf-8")
    size_mb = len(html.encode("utf-8")) / (1024 * 1024)
    print(f"HTML report written to {output_file} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
