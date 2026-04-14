from __future__ import annotations

from typing import Any

from context_bucket.models import ContextBucketRecord, ContextBucketWorkflowPreference


def latest_schema_record(records: list[ContextBucketRecord], schema_name: str) -> ContextBucketRecord | None:
    matches = [
        record for record in records
        if record.data_schema is not None and (record.data_schema.schema_name or "").strip().lower() == schema_name
    ]
    matches.sort(key=lambda record: record.created_at, reverse=True)
    return matches[0] if matches else None


def summarize_preference_from_notes(records: list[ContextBucketRecord]) -> ContextBucketWorkflowPreference:
    texts = [record.text.lower() for record in records if record.text.strip()]
    style_notes: list[str] = []
    style_preferences: dict[str, str] = {}
    brevity_hits = sum("concise" in text or "brief" in text or "short" in text for text in texts)
    structure_hits = sum("bullet" in text or "structured" in text or "outline" in text for text in texts)
    clarify_low_hits = sum("best effort" in text or "don't ask" in text or "do not ask" in text for text in texts)
    initiative_hits = sum("proactive" in text or "next step" in text or "take initiative" in text for text in texts)
    evidence_hits = sum("cite" in text or "source" in text or "evidence" in text for text in texts)
    direct_hits = sum("direct" in text or "clear" in text for text in texts)
    warm_hits = sum("warm" in text or "friendly" in text for text in texts)
    formal_hits = sum("professional" in text or "formal" in text for text in texts)

    if brevity_hits:
        style_notes.append("prefer concise outputs")
    if structure_hits:
        style_notes.append("prefer structured responses when useful")
    if clarify_low_hits:
        style_notes.append("prefer best-effort execution over clarification")
    if initiative_hits:
        style_notes.append("prefer clear next steps")
    if direct_hits:
        style_preferences["directness"] = "high"
    if warm_hits:
        style_preferences["warmth"] = "medium"
    if formal_hits:
        style_preferences["formality"] = "medium_high"

    return ContextBucketWorkflowPreference(
        autonomy_level="medium_high" if initiative_hits or clarify_low_hits else "medium",
        clarification_preference="low" if clarify_low_hits else "medium",
        brevity_preference="high" if brevity_hits else "medium",
        structure_preference="medium_high" if structure_hits else "medium",
        initiative_preference="high" if initiative_hits else "medium",
        risk_tolerance="moderate",
        evidence_preference="medium_high" if evidence_hits else "medium",
        style_preferences=style_preferences,
        workflow_defaults=workflow_defaults_from_preferences(
            prefer_best_effort=clarify_low_hits > 0,
            prefer_next_step=initiative_hits > 0,
            prefer_evidence=evidence_hits > 0,
        ),
        style_notes=style_notes,
        evidence_count=len(records),
    )


def workflow_defaults_from_preferences(
    *,
    prefer_best_effort: bool,
    prefer_next_step: bool,
    prefer_evidence: bool,
) -> dict[str, dict[str, Any]]:
    return {
        "reply": {
            "draft_full_response": True,
            "include_next_step": prefer_next_step,
            "prefer_best_effort_over_questions": prefer_best_effort,
        },
        "research": {
            "summary_first": True,
            "cite_sources": prefer_evidence,
        },
    }


def merge_workflow_preference(
    *,
    base: ContextBucketWorkflowPreference,
    preference_data: dict[str, Any],
    approved_text: str,
) -> ContextBucketWorkflowPreference:
    merged = base.model_dump(mode="json")
    explicit = dict(preference_data or {})
    text = approved_text.lower().strip()

    for field in (
        "autonomy_level",
        "clarification_preference",
        "brevity_preference",
        "structure_preference",
        "initiative_preference",
        "risk_tolerance",
        "evidence_preference",
    ):
        value = explicit.get(field)
        if value not in (None, ""):
            merged[field] = str(value)

    merged_style = dict(merged.get("style_preferences") or {})
    explicit_style = explicit.get("style_preferences") or {}
    if isinstance(explicit_style, dict):
        for key, value in explicit_style.items():
            if value not in (None, ""):
                merged_style[str(key)] = str(value)
    if "concise" in text or "brief" in text or "short" in text:
        merged["brevity_preference"] = "high"
    if "best effort" in text or "do not ask" in text or "don't ask" in text:
        merged["clarification_preference"] = "low"
    if "proactive" in text or "next step" in text or "take initiative" in text:
        merged["initiative_preference"] = "high"
        merged["autonomy_level"] = "medium_high"
    if "structured" in text or "bullet" in text or "outline" in text:
        merged["structure_preference"] = "medium_high"
    if "source" in text or "cite" in text or "evidence" in text:
        merged["evidence_preference"] = "medium_high"
    if "direct" in text or "clear" in text:
        merged_style["directness"] = "high"
    if "warm" in text or "friendly" in text:
        merged_style["warmth"] = "medium"
    if "professional" in text or "formal" in text:
        merged_style["formality"] = "medium_high"
    merged["style_preferences"] = merged_style

    existing_notes = [str(item) for item in (merged.get("style_notes") or []) if str(item).strip()]
    explicit_notes = explicit.get("style_notes") or []
    notes = existing_notes + [str(item) for item in explicit_notes if str(item).strip()]
    if "concise" in text or "brief" in text or "short" in text:
        notes.append("prefer concise outputs")
    if "best effort" in text or "do not ask" in text or "don't ask" in text:
        notes.append("prefer best-effort execution over clarification")
    if "proactive" in text or "next step" in text or "take initiative" in text:
        notes.append("prefer clear next steps")
    if "structured" in text or "bullet" in text or "outline" in text:
        notes.append("prefer structured responses when useful")
    merged["style_notes"] = list(dict.fromkeys(notes))

    defaults = dict(merged.get("workflow_defaults") or {})
    explicit_defaults = explicit.get("workflow_defaults") or {}
    if isinstance(explicit_defaults, dict):
        for workflow_name, workflow_data in explicit_defaults.items():
            current = dict(defaults.get(str(workflow_name)) or {})
            if isinstance(workflow_data, dict):
                current.update(workflow_data)
            defaults[str(workflow_name)] = current
    defaults.update(
        workflow_defaults_from_preferences(
            prefer_best_effort=merged.get("clarification_preference") == "low",
            prefer_next_step=merged.get("initiative_preference") == "high",
            prefer_evidence=merged.get("evidence_preference") in {"medium_high", "high"},
        )
    )
    merged["workflow_defaults"] = defaults
    merged["evidence_count"] = int(merged.get("evidence_count") or 0) + (1 if (approved_text.strip() or explicit) else 0)
    return ContextBucketWorkflowPreference.model_validate(merged)


def workflow_preference_text(preference: ContextBucketWorkflowPreference) -> str:
    parts = [
        f"autonomy: {preference.autonomy_level}",
        f"clarification: {preference.clarification_preference}",
        f"brevity: {preference.brevity_preference}",
        f"structure: {preference.structure_preference}",
        f"initiative: {preference.initiative_preference}",
        f"evidence: {preference.evidence_preference}",
    ]
    for key, value in sorted(preference.style_preferences.items()):
        parts.append(f"{key}: {value}")
    parts.extend(preference.style_notes)
    return ". ".join(part for part in parts if part).strip()
