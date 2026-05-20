#!/usr/bin/env python3
"""Export README chart PNGs from benchmark run_summary JSON files."""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BENCHMARK_ROOT = Path(__file__).resolve().parent
REPO_ROOT = BENCHMARK_ROOT.parent
DOCS_IMAGES = REPO_ROOT / "docs" / "images"
COMPARISON_HTML = REPO_ROOT / "benchmark-comparison-report.html"

DIFFICULTY_TIERS = [
    "trivial", "easy", "moderate_easy", "moderate",
    "moderate_hard", "hard", "harder", "very_hard", "extreme", "maximum",
]

TIER_CONFIG = {
    "trivial": {"runs": (1, 5)},
    "easy": {"runs": (6, 10)},
    "moderate_easy": {"runs": (11, 15)},
    "moderate": {"runs": (16, 20)},
    "moderate_hard": {"runs": (21, 25)},
    "hard": {"runs": (26, 30)},
    "harder": {"runs": (31, 35)},
    "very_hard": {"runs": (36, 40)},
    "extreme": {"runs": (41, 45)},
    "maximum": {"runs": (46, 50)},
}


def tier_for_run(run_num: int) -> str:
    for tier, cfg in TIER_CONFIG.items():
        low, high = cfg["runs"]
        if low <= run_num <= high:
            return tier
    return "unknown"


def compute_tier_stats(runs: list[dict]) -> dict[str, dict]:
    ok = [r for r in runs if r.get("exit_code") == 0]
    tiers: dict[str, dict] = {}
    for tier_name in DIFFICULTY_TIERS:
        tier_runs = [r for r in ok if tier_for_run(r["run"]) == tier_name]
        ret = [r["retrieval_hit_rate"] for r in tier_runs if r.get("retrieval_hit_rate") is not None]
        asm = [r["assembly_hit_rate"] for r in tier_runs if r.get("assembly_hit_rate") is not None]
        rt = [r["runtime_seconds"] for r in tier_runs if r.get("runtime_seconds") is not None]
        tiers[tier_name] = {
            "retrieval_mean": statistics.mean(ret) if ret else 0,
            "assembly_mean": statistics.mean(asm) if asm else 0,
            "runtime_mean": statistics.mean(rt) if rt else 0,
        }
    return tiers


def _parse_tier_stats_from_html(name: str) -> dict[str, dict] | None:
    if not COMPARISON_HTML.exists():
        return None
    text = COMPARISON_HTML.read_text(encoding="utf-8")
    match = re.search(rf"const {name} = (\{{.*?\}});", text)
    if not match:
        return None
    return json.loads(match.group(1))


def _tier_series(stats: dict[str, dict], key: str, *, percent: bool = False) -> list[float]:
    values = [stats[t][key] for t in DIFFICULTY_TIERS if t in stats]
    return [v * 100 for v in values] if percent else values


def _setup_axes(title: str) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(12, 5), facecolor="#0f172a")
    ax.set_facecolor("#1e293b")
    ax.set_title(title, color="#e2e8f0", fontsize=14, pad=12)
    ax.tick_params(colors="#94a3b8")
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.grid(True, color=(1, 1, 1, 0.08))
    return fig, ax


def export_line_chart(
    out_path: Path,
    title: str,
    ylabel: str,
    struct_stats: dict,
    unstruct_stats: dict,
    default_stats: dict,
    key: str,
) -> None:
    labels = [t.replace("_", " ") for t in DIFFICULTY_TIERS]
    x = np.arange(len(labels))
    fig, ax = _setup_axes(title)
    ax.plot(x, _tier_series(struct_stats, key, percent=True), label="Structured", color="#10b981", linewidth=2.5)
    ax.plot(x, _tier_series(unstruct_stats, key, percent=True), label="Unstructured", color="#ef4444", linewidth=2.5)
    ax.plot(
        x, _tier_series(default_stats, key, percent=True),
        label="Default", color="#64748b", linewidth=2, linestyle="--",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel(ylabel, color="#94a3b8")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False, labelcolor="#94a3b8")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def export_runtime_chart(
    out_path: Path,
    struct_stats: dict,
    unstruct_stats: dict,
    default_stats: dict,
) -> None:
    labels = [t.replace("_", " ") for t in DIFFICULTY_TIERS]
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = _setup_axes("Average Runtime by Difficulty Tier (Logarithmic Scale)")
    ax.bar(x - width, _tier_series(struct_stats, "runtime_mean"), width, label="Structured", color="#10b981")
    ax.bar(x, _tier_series(unstruct_stats, "runtime_mean"), width, label="Unstructured", color="#ef4444")
    ax.bar(x + width, _tier_series(default_stats, "runtime_mean"), width, label="Default", color="#64748b")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Runtime (seconds)", color="#94a3b8")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False, labelcolor="#94a3b8")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    struct_runs = json.loads((BENCHMARK_ROOT / "run_summary_structured.json").read_text(encoding="utf-8"))
    unstruct_runs = json.loads((BENCHMARK_ROOT / "run_summary_unstructured.json").read_text(encoding="utf-8"))
    default_runs = json.loads((BENCHMARK_ROOT / "run_summary.json").read_text(encoding="utf-8"))

    struct_stats = _parse_tier_stats_from_html("STRUCTURED_TIER_STATS") or compute_tier_stats(struct_runs)
    unstruct_stats = _parse_tier_stats_from_html("UNSTRUCTURED_TIER_STATS") or compute_tier_stats(unstruct_runs)
    default_stats = _parse_tier_stats_from_html("DEFAULT_TIER_STATS") or compute_tier_stats(default_runs)

    export_line_chart(
        DOCS_IMAGES / "retrieval_by_tier.png",
        "Retrieval Hit Rate by Difficulty Tier",
        "Retrieval Hit Rate (%)",
        struct_stats, unstruct_stats, default_stats, "retrieval_mean",
    )
    export_line_chart(
        DOCS_IMAGES / "assembly_by_tier.png",
        "Assembly Hit Rate by Difficulty Tier",
        "Assembly Hit Rate (%)",
        struct_stats, unstruct_stats, default_stats, "assembly_mean",
    )
    export_runtime_chart(DOCS_IMAGES / "runtime_by_tier.png", struct_stats, unstruct_stats, default_stats)
    print(f"Wrote charts to {DOCS_IMAGES}")


if __name__ == "__main__":
    main()
