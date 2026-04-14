from __future__ import annotations

from typing import Any

from context_bucket.models import (
    ContextBucketAssembleRequest,
    ContextBucketContextBlock,
    ContextBucketTaskEnvelopeResponse,
    ContextBucketWorkflowIntent,
    ContextBucketWorkflowPreference,
)


async def prepare_task_envelope(service: Any, payload: ContextBucketAssembleRequest) -> ContextBucketTaskEnvelopeResponse:
    prepared = await service.prepare_context(payload)
    workflow_intent = derive_workflow_intent(payload, prepared.context_blocks)
    workflow_preference = service._derive_user_workflow_preference(payload)
    context_summary = context_summary_from_blocks(prepared.context_blocks)
    objective = task_objective(workflow_intent, prepared.context_blocks)
    return ContextBucketTaskEnvelopeResponse(
        query_text=payload.query_text,
        objective=objective,
        workflow_intent=workflow_intent,
        user_workflow_preference=workflow_preference,
        context_blocks=prepared.context_blocks,
        context_summary=context_summary,
        output_contract=output_contract(workflow_intent, workflow_preference),
        retrieval_strategy=prepared.retrieval_strategy,
        token_estimate=prepared.token_estimate,
        audit_id=prepared.audit_id,
    )


def derive_workflow_intent(
    payload: ContextBucketAssembleRequest,
    context_blocks: list[ContextBucketContextBlock],
) -> ContextBucketWorkflowIntent:
    query_text = payload.query_text.strip().lower()
    workflow_type = "analyze"
    action = "respond"
    output_type = "text"
    confidence = 0.55
    workflow_rules = [
        ("reply", {"reply", "answer", "respond"}, "draft_response", "message"),
        ("rewrite", {"rewrite", "edit", "improve"}, "rewrite_content", "rewrite"),
        ("code", {"code", "fix", "implement", "refactor"}, "produce_code", "code"),
        ("draft", {"draft", "write", "compose"}, "draft_content", "draft"),
        ("summarize", {"summarize", "summary", "recap"}, "summarize_context", "summary"),
        ("research", {"research", "investigate", "look up"}, "research_topic", "report"),
        ("analyze", {"analyze", "review", "inspect", "evaluate"}, "analyze_context", "analysis"),
        ("extract", {"extract", "pull", "get fields"}, "extract_fields", "structured_data"),
        ("plan", {"plan", "next steps", "what should i do next"}, "propose_plan", "plan"),
        ("decide", {"decide", "choose", "recommend"}, "recommend_decision", "recommendation"),
        ("update_record", {"update", "save", "sync"}, "update_record", "record_update"),
    ]
    for candidate_workflow, phrases, candidate_action, candidate_output in workflow_rules:
        if any(phrase in query_text for phrase in phrases):
            workflow_type = candidate_workflow
            action = candidate_action
            output_type = candidate_output
            confidence = 0.82
            break

    source_refs, data_schema_names, source_types, kinds = context_characteristics(context_blocks)

    target_type = infer_target_type(
        query_text=query_text,
        workflow_type=workflow_type,
        data_schema_names=data_schema_names,
        source_types=source_types,
        kinds=kinds,
    )
    if target_type == "email":
        confidence = max(confidence, 0.88)
        if workflow_type == "reply":
            output_type = "email_reply"
    elif target_type in {"note", "document", "research_material", "codebase"}:
        confidence = max(confidence, 0.84)

    goal_map = {
        "reply": "produce_sendable_reply",
        "draft": "produce_working_draft",
        "summarize": "produce_compact_summary",
        "research": "produce_evidence_backed_report",
        "analyze": "produce_actionable_analysis",
        "extract": "produce_structured_extraction",
        "rewrite": "produce_improved_revision",
        "plan": "produce_next_step_plan",
        "decide": "produce_recommendation",
        "code": "produce_code_change",
        "update_record": "update_system_record",
    }
    constraints = intent_constraints(query_text, workflow_type)
    if payload.intent_data:
        constraints.append("use provided intent data")
        confidence = max(confidence, 0.9)
    return ContextBucketWorkflowIntent(
        workflow_type=workflow_type,
        action=action,
        target_type=target_type,
        goal=goal_map.get(workflow_type, "produce_helpful_result"),
        constraints=list(dict.fromkeys(constraints)),
        output_type=output_type,
        confidence=min(1.0, confidence),
        source_refs=source_refs[:6],
    )


def context_characteristics(
    context_blocks: list[ContextBucketContextBlock],
) -> tuple[list[str], list[str], set[str], set[str]]:
    source_refs: list[str] = []
    data_schema_names: list[str] = []
    source_types: set[str] = set()
    kinds: set[str] = set()
    for block in context_blocks:
        for item in block.items:
            source_types.add(item.source_type)
            kinds.add(item.kind)
            if item.provenance.get("source_key"):
                source_refs.append(str(item.provenance["source_key"]))
            data_schema = item.provenance.get("data_schema") or {}
            schema_name = str(data_schema.get("schema_name") or "").strip().lower()
            if schema_name:
                data_schema_names.append(schema_name)
    return list(dict.fromkeys(source_refs)), data_schema_names, source_types, kinds


def infer_target_type(
    *,
    query_text: str,
    workflow_type: str,
    data_schema_names: list[str],
    source_types: set[str],
    kinds: set[str],
) -> str:
    doc_terms = ("document", "doc", "file", "pdf")
    note_terms = ("note", "notes", "meeting note", "meeting notes")
    if workflow_type == "code":
        return "codebase"
    if "email" in query_text or "email_source" in data_schema_names:
        return "email"
    if any(term in query_text for term in note_terms):
        return "note"
    if "imported_document" in source_types or any(term in query_text for term in doc_terms):
        return "document"
    if workflow_type == "research" and kinds & {"research_report", "research_finding", "evidence_summary"}:
        return "research_material"
    return "context"


def intent_constraints(query_text: str, workflow_type: str) -> list[str]:
    constraints = ["use user workflow preferences"]
    if any(token in query_text for token in ("concise", "brief", "short")):
        constraints.append("keep output concise")
    if any(token in query_text for token in ("source", "citation", "cite", "evidence")):
        constraints.append("ground response in evidence")
    if workflow_type == "reply":
        constraints.append("produce a sendable response")
    return constraints


def context_summary_from_blocks(context_blocks: list[ContextBucketContextBlock]) -> str:
    if not context_blocks:
        return "No relevant stored context was found."
    snippets: list[str] = []
    for block in context_blocks[:3]:
        text = block.text.strip()
        if not text:
            continue
        snippets.append(f"{block.name}: {text[:180].strip()}")
    return " ".join(snippets).strip()


def task_objective(
    workflow_intent: ContextBucketWorkflowIntent,
    context_blocks: list[ContextBucketContextBlock],
) -> str:
    if workflow_intent.workflow_type == "reply" and workflow_intent.target_type == "email":
        return "Draft a reply to the active email using the user's workflow preferences."
    if workflow_intent.target_type == "note":
        return f"{workflow_intent.action.replace('_', ' ').capitalize()} for the active note using the retrieved context."
    if workflow_intent.target_type == "document":
        return f"{workflow_intent.action.replace('_', ' ').capitalize()} for the active document using the retrieved context."
    if workflow_intent.target_type == "research_material":
        return f"{workflow_intent.action.replace('_', ' ').capitalize()} using the active research material."
    if context_blocks:
        return (
            f"{workflow_intent.action.replace('_', ' ').capitalize()} for the active "
            f"{workflow_intent.target_type} using the retrieved context."
        )
    return "Produce a useful result for the current user request."


def output_contract(
    workflow_intent: ContextBucketWorkflowIntent,
    workflow_preference: ContextBucketWorkflowPreference,
) -> dict[str, Any]:
    must_include = ["match user workflow preferences"]
    must_avoid = ["irrelevant context"]
    if workflow_preference.brevity_preference == "high":
        must_avoid.append("unnecessary verbosity")
    if workflow_preference.clarification_preference == "low":
        must_include.append("make a best-effort attempt before asking questions")
    if workflow_preference.initiative_preference == "high":
        must_include.append("include a clear next step when appropriate")
    return {
        "type": workflow_intent.output_type,
        "must_include": must_include,
        "must_avoid": must_avoid,
    }
