#!/usr/bin/env python3
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any
from datetime import datetime

BENCHMARK_ROOT = Path(__file__).resolve().parent
REPO_ROOT = BENCHMARK_ROOT.parent
RESULTS_DIR = BENCHMARK_ROOT / "results"
CASES_DIR = BENCHMARK_ROOT / "cases"
HTML_PARTS = BENCHMARK_ROOT / "html_parts"

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

def tier_for_run(run_num: int) -> str:
    for tier, cfg in TIER_CONFIG.items():
        low, high = cfg["runs"]
        if low <= run_num <= high:
            return tier
    return "unknown"

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

def load_all_data(summary_file: Path, results_dir: Path, variant: str) -> list[dict[str, Any]]:
    dataset_map = _dataset_map_for_variant(variant)
    if not summary_file.exists():
        print(f"  Summary file not found: {summary_file}")
        return []
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
            except Exception as e:
                print(f"    Error reading result path {result_path}: {e}")
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

def compute_tier_stats(runs: list[dict]) -> dict[str, dict]:
    ok = [r for r in runs if r.get("exit_code") == 0]
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

def get_pruning_analysis(structured_runs: list[dict], unstructured_runs: list[dict], default_runs: list[dict]) -> list[dict]:
    # Compute active, pruned, and total imported record counts for each variant by grouping on dataset name
    # We will average the values across all runs of the same dataset
    variants = [
        ("structured", structured_runs),
        ("unstructured", unstructured_runs),
        ("default", default_runs)
    ]
    
    datasets_stats: list[dict] = []
    
    for v_name, runs in variants:
        # Group by dataset filename
        ds_groups: dict[str, list[dict]] = {}
        for r in runs:
            ds = r.get("dataset")
            if ds and r.get("exit_code") == 0:
                ds_groups.setdefault(ds, []).append(r)
                
        for filename, grp in sorted(ds_groups.items()):
            active_list = []
            pruned_list = []
            for r in grp:
                stats = r.get("stats", {})
                active = stats.get("record_count", 0)
                pruned = stats.get("state", {}).get("pruned_records_total", 0)
                if active > 0 or pruned > 0:
                    active_list.append(active)
                    pruned_list.append(pruned)
            
            if active_list:
                avg_active = round(statistics.mean(active_list))
                avg_pruned = round(statistics.mean(pruned_list))
                total_imported = avg_active + avg_pruned
                pruned_pct = (avg_pruned / total_imported * 100) if total_imported > 0 else 0
                
                datasets_stats.append({
                    "variant": v_name,
                    "filename": filename,
                    "imported": total_imported,
                    "active": avg_active,
                    "pruned": avg_pruned,
                    "pruned_pct": pruned_pct
                })
                
    return datasets_stats

def main() -> None:
    print("Loading structured runs...")
    struct_summary_file = BENCHMARK_ROOT / "run_summary_structured.json"
    struct_results_dir = RESULTS_DIR / "structured"
    structured_runs = load_all_data(struct_summary_file, struct_results_dir, "structured")
    structured_summary = compute_summary_stats(structured_runs)
    structured_tier_stats = compute_tier_stats(structured_runs)
    
    print("Loading unstructured runs...")
    unstruct_summary_file = BENCHMARK_ROOT / "run_summary_unstructured.json"
    unstruct_results_dir = RESULTS_DIR / "unstructured"
    unstructured_runs = load_all_data(unstruct_summary_file, unstruct_results_dir, "unstructured")
    unstructured_summary = compute_summary_stats(unstructured_runs)
    unstructured_tier_stats = compute_tier_stats(unstructured_runs)
    
    print("Loading default runs...")
    default_summary_file = BENCHMARK_ROOT / "run_summary.json"
    default_results_dir = RESULTS_DIR
    default_runs = load_all_data(default_summary_file, default_results_dir, "default")
    default_summary = compute_summary_stats(default_runs)
    default_tier_stats = compute_tier_stats(default_runs)
    
    print("Extracting pruning stats...")
    datasets_stats = get_pruning_analysis(structured_runs, unstructured_runs, default_runs)
    
    print("Reading templates...")
    css = (HTML_PARTS / "comparison_css.html").read_text(encoding="utf-8")
    template = (HTML_PARTS / "comparison_template.html").read_text(encoding="utf-8")
    
    # Render variables into template
    title = "Context Bucket - Benchmark Comparison Presentation"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    html = template
    html = html.replace("{{TITLE}}", title)
    html = html.replace("{{TIMESTAMP}}", timestamp)
    html = html.replace("{{CSS}}", css)
    
    # Inject runs
    html = html.replace("{{STRUCTURED_RUNS_JSON}}", json.dumps(structured_runs, default=str))
    html = html.replace("{{UNSTRUCTURED_RUNS_JSON}}", json.dumps(unstructured_runs, default=str))
    html = html.replace("{{DEFAULT_RUNS_JSON}}", json.dumps(default_runs, default=str))
    
    # Inject summaries
    html = html.replace("{{STRUCTURED_SUMMARY_JSON}}", json.dumps(structured_summary))
    html = html.replace("{{UNSTRUCTURED_SUMMARY_JSON}}", json.dumps(unstructured_summary))
    html = html.replace("{{DEFAULT_SUMMARY_JSON}}", json.dumps(default_summary))
    
    # Inject tier stats
    html = html.replace("{{STRUCTURED_TIER_STATS_JSON}}", json.dumps(structured_tier_stats))
    html = html.replace("{{UNSTRUCTURED_TIER_STATS_JSON}}", json.dumps(unstructured_tier_stats))
    html = html.replace("{{DEFAULT_TIER_STATS_JSON}}", json.dumps(default_tier_stats))
    
    # Other configs
    html = html.replace("{{TIERS_JSON}}", json.dumps(DIFFICULTY_TIERS))
    html = html.replace("{{TIER_COLORS_JSON}}", json.dumps(TIER_COLORS))
    html = html.replace("{{DATASETS_STATS_JSON}}", json.dumps(datasets_stats))
    
    output_file = REPO_ROOT / "benchmark-comparison-report.html"
    output_file.write_text(html, encoding="utf-8")
    print(f"SUCCESS: Written comparison report to {output_file}")

if __name__ == "__main__":
    main()
