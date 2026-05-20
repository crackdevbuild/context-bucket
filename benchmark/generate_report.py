from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

BENCHMARK_ROOT = Path(__file__).resolve().parent
REPO_ROOT = BENCHMARK_ROOT.parent
SUMMARY_FILE = BENCHMARK_ROOT / "run_summary.json"

DIFFICULTY_TIERS = [
    "trivial", "easy", "moderate_easy", "moderate",
    "moderate_hard", "hard", "harder", "very_hard", "extreme", "maximum",
]

TIER_RUNS = {
    "trivial": (1, 5), "easy": (6, 10), "moderate_easy": (11, 15), "moderate": (16, 20),
    "moderate_hard": (21, 25), "hard": (26, 30), "harder": (31, 35), "very_hard": (36, 40),
    "extreme": (41, 45), "maximum": (46, 50),
}


def _tier_for_run(run_num: int) -> str:
    for name, (lo, hi) in TIER_RUNS.items():
        if lo <= run_num <= hi:
            return name
    return "maximum"


def load_results() -> list[dict[str, Any]]:
    raw = SUMMARY_FILE.read_text(encoding="utf-8")
    return json.loads(raw)


def compute_report(results: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in results if r.get("exit_code") == 0]
    report: dict[str, Any] = {"total_runs": len(results), "successful_runs": len(ok), "failed_runs": len(results) - len(ok)}

    series_data: dict[str, dict[str, Any]] = {}
    for series in ["series_a", "series_b", "series_c"]:
        s_results = [r for r in ok if r.get("series") == series]
        if not s_results:
            continue
        ret_rates = [r["retrieval_hit_rate"] for r in s_results if r.get("retrieval_hit_rate") is not None]
        asm_rates = [r["assembly_hit_rate"] for r in s_results if r.get("assembly_hit_rate") is not None]
        runtimes = [r["runtime_seconds"] for r in s_results if r.get("runtime_seconds") is not None]

        tier_breakdown: dict[str, dict[str, Any]] = {}
        for tier in DIFFICULTY_TIERS:
            lo, hi = TIER_RUNS[tier]
            tier_results = [r for r in s_results if lo <= r.get("run", 0) <= hi]
            if not tier_results:
                tier_breakdown[tier] = {"count": 0}
                continue
            t_ret = [r["retrieval_hit_rate"] for r in tier_results if r.get("retrieval_hit_rate") is not None]
            t_asm = [r["assembly_hit_rate"] for r in tier_results if r.get("assembly_hit_rate") is not None]
            tier_breakdown[tier] = {
                "count": len(tier_results),
                "retrieval_hit_rate_mean": statistics.mean(t_ret) if t_ret else None,
                "retrieval_hit_rate_min": min(t_ret) if t_ret else None,
                "retrieval_hit_rate_max": max(t_ret) if t_ret else None,
                "assembly_hit_rate_mean": statistics.mean(t_asm) if t_asm else None,
                "assembly_hit_rate_min": min(t_asm) if t_asm else None,
                "assembly_hit_rate_max": max(t_asm) if t_asm else None,
            }

        series_data[series] = {
            "total_runs": len(s_results),
            "retrieval_hit_rate_mean": statistics.mean(ret_rates) if ret_rates else None,
            "retrieval_hit_rate_median": statistics.median(ret_rates) if ret_rates else None,
            "assembly_hit_rate_mean": statistics.mean(asm_rates) if asm_rates else None,
            "assembly_hit_rate_median": statistics.median(asm_rates) if asm_rates else None,
            "runtime_mean": statistics.mean(runtimes) if runtimes else None,
            "runtime_median": statistics.median(runtimes) if runtimes else None,
            "by_tier": tier_breakdown,
        }

    report["series"] = series_data

    overall_ret = [r["retrieval_hit_rate"] for r in ok if r.get("retrieval_hit_rate") is not None]
    overall_asm = [r["assembly_hit_rate"] for r in ok if r.get("assembly_hit_rate") is not None]
    report["overall"] = {
        "retrieval_hit_rate_mean": statistics.mean(overall_ret) if overall_ret else None,
        "retrieval_hit_rate_median": statistics.median(overall_ret) if overall_ret else None,
        "assembly_hit_rate_mean": statistics.mean(overall_asm) if overall_asm else None,
        "assembly_hit_rate_median": statistics.median(overall_asm) if overall_asm else None,
    }

    overall_tier: dict[str, dict[str, Any]] = {}
    for tier in DIFFICULTY_TIERS:
        lo, hi = TIER_RUNS[tier]
        tier_results = [r for r in ok if lo <= r.get("run", 0) <= hi]
        t_ret = [r["retrieval_hit_rate"] for r in tier_results if r.get("retrieval_hit_rate") is not None]
        t_asm = [r["assembly_hit_rate"] for r in tier_results if r.get("assembly_hit_rate") is not None]
        overall_tier[tier] = {
            "count": len(tier_results),
            "retrieval_hit_rate_mean": statistics.mean(t_ret) if t_ret else None,
            "assembly_hit_rate_mean": statistics.mean(t_asm) if t_asm else None,
        }
    report["overall_by_tier"] = overall_tier

    return report


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Context Bucket 150-Run Benchmark Report",
        "",
        f"- Total runs: {report['total_runs']}",
        f"- Successful: {report['successful_runs']}",
        f"- Failed: {report['failed_runs']}",
        "",
        "## Overall Summary",
        "",
        f"- Retrieval hit rate (mean): {report['overall']['retrieval_hit_rate_mean']:.3f}" if report['overall'].get('retrieval_hit_rate_mean') is not None else "- Retrieval hit rate (mean): N/A",
        f"- Retrieval hit rate (median): {report['overall']['retrieval_hit_rate_median']:.3f}" if report['overall'].get('retrieval_hit_rate_median') is not None else "- Retrieval hit rate (median): N/A",
        f"- Assembly hit rate (mean): {report['overall']['assembly_hit_rate_mean']:.3f}" if report['overall'].get('assembly_hit_rate_mean') is not None else "- Assembly hit rate (mean): N/A",
        f"- Assembly hit rate (median): {report['overall']['assembly_hit_rate_median']:.3f}" if report['overall'].get('assembly_hit_rate_median') is not None else "- Assembly hit rate (median): N/A",
        "",
        "## Per-Series Summary",
        "",
        "| Series | Runs | Ret. Rate (mean) | Asm. Rate (mean) | Runtime (mean) |",
        "|--------|------|------------------|-------------------|----------------|",
    ]
    for series in ["series_a", "series_b", "series_c"]:
        sd = report["series"].get(series, {})
        if not sd:
            continue
        ret_m = f"{sd['retrieval_hit_rate_mean']:.3f}" if sd.get("retrieval_hit_rate_mean") is not None else "N/A"
        asm_m = f"{sd['assembly_hit_rate_mean']:.3f}" if sd.get("assembly_hit_rate_mean") is not None else "N/A"
        rt_m = f"{sd['runtime_mean']:.1f}s" if sd.get("runtime_mean") is not None else "N/A"
        lines.append(f"| {series} | {sd['total_runs']} | {ret_m} | {asm_m} | {rt_m} |")

    lines.extend(["", "## Difficulty Tier Breakdown (Overall)", "",
                   "| Tier | Runs | Ret. Rate (mean) | Asm. Rate (mean) |",
                   "|------|------|------------------|-------------------|"])
    for tier in DIFFICULTY_TIERS:
        td = report["overall_by_tier"].get(tier, {})
        cnt = td.get("count", 0)
        ret_m = f"{td['retrieval_hit_rate_mean']:.3f}" if td.get("retrieval_hit_rate_mean") is not None else "N/A"
        asm_m = f"{td['assembly_hit_rate_mean']:.3f}" if td.get("assembly_hit_rate_mean") is not None else "N/A"
        lines.append(f"| {tier} | {cnt} | {ret_m} | {asm_m} |")

    lines.extend(["", "## Per-Series Per-Tier Breakdown", ""])
    for series in ["series_a", "series_b", "series_c"]:
        sd = report["series"].get(series, {})
        if not sd:
            continue
        lines.extend([f"### {series}", "",
                       "| Tier | Ret. Rate (mean) | Asm. Rate (mean) |",
                       "|------|------------------|-------------------|"])
        for tier in DIFFICULTY_TIERS:
            td = sd.get("by_tier", {}).get(tier, {})
            if td.get("count", 0) == 0:
                lines.append(f"| {tier} | — | — |")
                continue
            ret_m = f"{td['retrieval_hit_rate_mean']:.3f}" if td.get("retrieval_hit_rate_mean") is not None else "N/A"
            asm_m = f"{td['assembly_hit_rate_mean']:.3f}" if td.get("assembly_hit_rate_mean") is not None else "N/A"
            lines.append(f"| {tier} | {ret_m} | {asm_m} |")
        lines.append("")

    lines.extend(["---", "", "_Generated for cross-AI-model comparison._", ""])
    return "\n".join(lines)


def main() -> None:
    if not SUMMARY_FILE.exists():
        print(f"Summary file not found: {SUMMARY_FILE}", file=__import__("sys").stderr)
        __import__("sys").exit(1)

    results = load_results()
    report = compute_report(results)

    json_path = REPO_ROOT / "benchmark-150run-report.json"
    md_path = REPO_ROOT / "benchmark-150run-report.md"

    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")

    print(f"JSON report: {json_path}")
    print(f"Markdown report: {md_path}")


if __name__ == "__main__":
    main()
