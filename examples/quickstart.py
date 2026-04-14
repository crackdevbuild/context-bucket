from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from context_bucket import ContextBucketAssembleRequest, ContextBucketService, ContextBucketSourceCreate


async def main() -> None:
    service = ContextBucketService()

    await service.ingest_source(
        ContextBucketSourceCreate(
            scope="user",
            user_id="u1",
            source_key="user_profile",
            kind="user_profile_note",
            text="The client prefers concise email updates and Friday summaries.",
        )
    )

    await service.ingest_source(
        ContextBucketSourceCreate(
            scope="user",
            user_id="u1",
            source_key="matter_note",
            kind="research_finding",
            text="Matter ACME-17 is in discovery. Drafts should avoid speculative statements.",
        )
    )

    prepared = await service.prepare_context(
        ContextBucketAssembleRequest(
            query_text="draft an update for the client on ACME-17",
            user_id="u1",
            assembly_mode="drafting",
            limit=6,
            token_budget=1200,
        )
    )

    print(json.dumps(prepared.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
