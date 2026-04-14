from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from context_bucket import (
    ContextBucketAssembleRequest,
    ContextBucketDataSchema,
    ContextBucketService,
    ContextBucketSourceCreate,
    Settings,
)


async def _service(name: str) -> ContextBucketService:
    return ContextBucketService(Settings(data_root=str(ROOT / ".local" / "example_workflows" / name)))


async def _reply_example() -> dict[str, object]:
    service = await _service("reply")
    await service.ingest_source(
        ContextBucketSourceCreate(
            scope="user",
            user_id="u1",
            source_key="workflow_prefs",
            kind="user_profile_note",
            structured_data={
                "clarification_preference": "low",
                "brevity_preference": "high",
                "initiative_preference": "high",
                "style_notes": ["prefer concise professional outputs"],
            },
            data_schema=ContextBucketDataSchema(
                schema_name="user_workflow_preference",
                schema_mode="declared",
                root_type="object",
                primary_text_paths=[
                    "clarification_preference",
                    "brevity_preference",
                    "initiative_preference",
                ],
            ),
        )
    )
    await service.ingest_source(
        ContextBucketSourceCreate(
            scope="user",
            user_id="u1",
            source_key="email_1",
            structured_data={
                "subject": "Interview invitation",
                "from_name": "Recruiter",
                "body_text": "We would like to invite you to interview next week.",
            },
            data_schema=ContextBucketDataSchema(
                schema_name="email_source",
                schema_mode="declared",
                root_type="object",
                primary_text_paths=["subject", "from_name", "body_text"],
            ),
        )
    )
    envelope = await service.prepare_task_envelope(
        ContextBucketAssembleRequest(
            query_text="answer the email",
            user_id="u1",
            include_user_scope=True,
            assembly_mode="drafting",
            token_budget=400,
        )
    )
    return {
        "objective": envelope.objective,
        "workflow_intent": envelope.workflow_intent.model_dump(mode="json"),
        "output_contract": envelope.output_contract,
        "context_summary": envelope.context_summary,
    }


async def _summarize_example() -> dict[str, object]:
    service = await _service("summarize")
    await service.ingest_source(
        ContextBucketSourceCreate(
            scope="session",
            session_id="s1",
            source_key="meeting_note",
            kind="research_finding",
            text="Meeting notes: launch slips two weeks, budget unchanged, legal review due Friday.",
        )
    )
    envelope = await service.prepare_task_envelope(
        ContextBucketAssembleRequest(
            query_text="summarize the meeting notes",
            session_id="s1",
            assembly_mode="assistant",
            token_budget=400,
        )
    )
    return {
        "objective": envelope.objective,
        "workflow_intent": envelope.workflow_intent.model_dump(mode="json"),
        "output_contract": envelope.output_contract,
        "context_summary": envelope.context_summary,
    }


async def _research_example() -> dict[str, object]:
    service = await _service("research")
    await service.ingest_source(
        ContextBucketSourceCreate(
            scope="session",
            session_id="s1",
            source_key="market_report",
            kind="research_report",
            text="ACME expansion research: regulatory risk is rising and customer acquisition costs are volatile.",
        )
    )
    await service.ingest_source(
        ContextBucketSourceCreate(
            scope="session",
            session_id="s1",
            source_key="risk_note",
            kind="research_finding",
            text="Expansion risk concentrates in pricing pressure, compliance overhead, and channel conflict.",
        )
    )
    envelope = await service.prepare_task_envelope(
        ContextBucketAssembleRequest(
            query_text="research ACME expansion risks",
            session_id="s1",
            assembly_mode="research",
            token_budget=400,
        )
    )
    return {
        "objective": envelope.objective,
        "workflow_intent": envelope.workflow_intent.model_dump(mode="json"),
        "output_contract": envelope.output_contract,
        "context_summary": envelope.context_summary,
    }


async def _rewrite_example() -> dict[str, object]:
    service = await _service("rewrite")
    await service.ingest_source(
        ContextBucketSourceCreate(
            scope="user",
            user_id="u1",
            source_key="rewrite_prefs",
            kind="user_profile_note",
            text="Prefer concise professional rewrites with short paragraphs.",
        )
    )
    await service.ingest_source(
        ContextBucketSourceCreate(
            scope="session",
            session_id="s1",
            source_key="draft_note",
            kind="research_finding",
            text="This draft note is too long and repetitive and needs to be clearer for the client.",
        )
    )
    envelope = await service.prepare_task_envelope(
        ContextBucketAssembleRequest(
            query_text="rewrite this note to be concise and professional",
            user_id="u1",
            session_id="s1",
            include_user_scope=True,
            assembly_mode="drafting",
            token_budget=400,
        )
    )
    return {
        "objective": envelope.objective,
        "workflow_intent": envelope.workflow_intent.model_dump(mode="json"),
        "output_contract": envelope.output_contract,
        "context_summary": envelope.context_summary,
    }


async def main() -> None:
    output = {
        "reply": await _reply_example(),
        "summarize": await _summarize_example(),
        "research": await _research_example(),
        "rewrite": await _rewrite_example(),
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
