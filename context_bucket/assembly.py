from __future__ import annotations

from typing import Any

from context_bucket.models import (
    ContextBucketAssembleRequest,
    ContextBucketAssembleResponse,
    ContextBucketContextBlock,
    ContextBucketContextItem,
    ContextBucketContextSection,
    ContextBucketPrepareContextResponse,
)


async def assemble_context(service: Any, payload: ContextBucketAssembleRequest) -> ContextBucketAssembleResponse:
    retrieved = await service.retrieve_context(payload)
    compressed = service._compress_context_items(retrieved.items)
    selected: list[ContextBucketContextItem] = []
    omitted_items = 0
    budget = int(payload.token_budget)
    consumed = 0
    for item in compressed:
        item_tokens = max(1, item.token_count_estimate)
        if selected and (consumed + item_tokens) > budget:
            omitted_items += 1
            continue
        selected.append(item)
        consumed += item_tokens
    omitted_items += max(0, len(retrieved.items) - len(compressed))
    sections = build_context_sections(service, payload.assembly_mode, selected)
    context_text = render_context(payload.query_text, payload.assembly_mode, sections)
    token_count_estimate = service.token_estimate(context_text)
    truncation_reason = "token_budget_reached" if omitted_items else None
    return ContextBucketAssembleResponse(
        query_text=payload.query_text,
        context_text=context_text,
        items=selected,
        sections=sections,
        assembly_mode=payload.assembly_mode,
        token_budget=budget,
        token_count_estimate=token_count_estimate,
        omitted_items=omitted_items,
        truncation_reason=truncation_reason,
        retrieval_strategy=retrieved.retrieval_strategy,
        audit_id=retrieved.audit_id,
    )


async def prepare_context(service: Any, payload: ContextBucketAssembleRequest) -> ContextBucketPrepareContextResponse:
    assembled = await assemble_context(service, payload)
    context_blocks: list[ContextBucketContextBlock] = []
    provenance: list[dict[str, Any]] = []
    seen_provenance: set[str] = set()
    for section in assembled.sections:
        block_text = "\n\n".join(item.text for item in section.items if item.text.strip())
        context_blocks.append(
            ContextBucketContextBlock(
                name=section.name,
                text=block_text,
                items=section.items,
                token_count_estimate=section.token_count_estimate,
            )
        )
        for item in section.items:
            key = f"{item.record_id}:{item.chunk_id}"
            if key in seen_provenance:
                continue
            seen_provenance.add(key)
            provenance.append(dict(item.provenance))
    return ContextBucketPrepareContextResponse(
        query_text=payload.query_text,
        request_summary=f"Prepare {payload.assembly_mode} context for: {payload.query_text}",
        context_blocks=context_blocks,
        provenance=provenance,
        token_estimate=assembled.token_count_estimate,
        retrieval_strategy=assembled.retrieval_strategy,
        audit_id=assembled.audit_id,
    )


def build_context_sections(
    service: Any,
    assembly_mode: str,
    items: list[ContextBucketContextItem],
) -> list[ContextBucketContextSection]:
    if not items:
        return []
    section_names = section_order(assembly_mode)
    grouped: dict[str, list[ContextBucketContextItem]] = {name: [] for name in section_names}
    for item in items:
        grouped[section_for_item(assembly_mode, item)].append(item)
    sections: list[ContextBucketContextSection] = []
    for name in section_names:
        section_items = grouped.get(name, [])
        if not section_items:
            continue
        sections.append(
            ContextBucketContextSection(
                name=name,
                items=section_items,
                token_count_estimate=sum(max(1, item.token_count_estimate) for item in section_items),
            )
        )
    return sections


def section_order(assembly_mode: str) -> list[str]:
    if assembly_mode == "planner":
        return ["objective_context", "active_constraints", "reference_memory"]
    if assembly_mode == "research":
        return ["key_evidence", "supporting_context", "background_memory"]
    if assembly_mode == "drafting":
        return ["drafting_instructions", "matter_context", "reference_material"]
    return ["priority_context", "supporting_context", "background_memory"]


def section_for_item(assembly_mode: str, item: ContextBucketContextItem) -> str:
    if assembly_mode == "drafting":
        if item.kind in {"user_profile_note", "decision_outcome"}:
            return "drafting_instructions"
        if item.scope == "session":
            return "matter_context"
        return "reference_material"
    if assembly_mode == "planner":
        if item.scope in {"session", "user"}:
            return "objective_context"
        if item.kind in {"decision_trace", "routing_decision", "decision_outcome"}:
            return "active_constraints"
        return "reference_memory"
    if assembly_mode == "research":
        if item.kind in {"research_report", "research_finding", "evidence_summary"}:
            return "key_evidence"
        if item.scope == "session":
            return "supporting_context"
        return "background_memory"
    if item.scope in {"session", "user"}:
        return "priority_context"
    if item.kind in {"research_report", "research_finding"}:
        return "supporting_context"
    return "background_memory"


def render_context(query_text: str, assembly_mode: str, sections: list[ContextBucketContextSection]) -> str:
    if not sections:
        return f"Request: {query_text}\nMode: {assembly_mode}\n\nNo stored context matched the request."
    lines = [
        f"Request: {query_text}",
        f"Mode: {assembly_mode}",
        "",
        "Use the following context sections when answering:",
    ]
    item_index = 1
    for section in sections:
        lines.append("")
        lines.append(f"## {section.name}")
        for item in section.items:
            label = item.title or item.kind
            lines.append(
                f"[{item_index}] {label} ({item.scope}, {item.source_type}, score={item.score:.3f}, reason={item.selection_reason})"
            )
            lines.append(item.text)
            if item.summary:
                lines.append(f"summary={item.summary}")
            if item.provenance.get("source_key"):
                lines.append(f"source_key={item.provenance['source_key']}")
            item_index += 1
            lines.append("")
    return "\n".join(lines).strip()
