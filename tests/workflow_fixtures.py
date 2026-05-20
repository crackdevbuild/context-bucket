from __future__ import annotations


WORKFLOW_FIXTURES = [
    {
        "name": "reply",
        "records": [
            {
                "kind": "user_profile_note",
                "scope": "user",
                "user_id": "u1",
                "source_key": "workflow_prefs",
                "structured_data": {
                    "clarification_preference": "low",
                    "brevity_preference": "high",
                    "initiative_preference": "high",
                    "style_notes": ["prefer concise outputs"],
                    "workflow_defaults": {"reply": {"draft_full_response": True}},
                },
                "data_schema": {
                    "schema_name": "user_workflow_preference",
                    "schema_mode": "declared",
                    "root_type": "object",
                    "primary_text_paths": [
                        "clarification_preference",
                        "brevity_preference",
                        "initiative_preference",
                    ],
                },
            },
            {
                "scope": "user",
                "user_id": "u1",
                "source_key": "email_1",
                "structured_data": {
                    "subject": "Interview invitation",
                    "from_name": "Recruiter",
                    "body_text": "We would like to invite you to interview next week.",
                },
                "data_schema": {
                    "schema_name": "email_source",
                    "schema_mode": "declared",
                    "root_type": "object",
                    "primary_text_paths": ["subject", "from_name", "body_text"],
                },
            },
        ],
        "query_text": "answer the email",
        "assembly_mode": "drafting",
        "expected_workflow_type": "reply",
        "expected_output_type": "email_reply",
        "expected_target_type": "email",
    },
    {
        "name": "summarize",
        "records": [
            {
                "kind": "research_finding",
                "scope": "session",
                "session_id": "s1",
                "source_key": "meeting_note",
                "text": "Client meeting notes: launch slips two weeks, budget unchanged, legal review due Friday.",
            }
        ],
        "query_text": "summarize the meeting notes",
        "assembly_mode": "assistant",
        "expected_workflow_type": "summarize",
        "expected_output_type": "summary",
        "expected_target_type": "note",
    },
    {
        "name": "research",
        "records": [
            {
                "kind": "research_report",
                "scope": "session",
                "session_id": "s1",
                "source_key": "market_report",
                "text": "ACME market expansion research: regulatory risk is rising and customer acquisition costs are volatile.",
            },
            {
                "kind": "research_finding",
                "scope": "session",
                "session_id": "s1",
                "source_key": "risk_note",
                "text": "Expansion risk concentrates in pricing pressure, compliance overhead, and channel conflict.",
            },
        ],
        "query_text": "research ACME expansion risks",
        "assembly_mode": "research",
        "expected_workflow_type": "research",
        "expected_output_type": "report",
        "expected_target_type": "research_material",
    },
    {
        "name": "rewrite",
        "records": [
            {
                "kind": "user_profile_note",
                "scope": "user",
                "user_id": "u1",
                "source_key": "rewrite_prefs",
                "text": "Prefer concise professional rewrites with short paragraphs.",
            },
            {
                "kind": "research_finding",
                "scope": "session",
                "session_id": "s1",
                "source_key": "draft_note",
                "text": "This draft note is too long and repetitive and needs to be clearer for the client.",
            },
        ],
        "query_text": "rewrite this note to be concise and professional",
        "assembly_mode": "drafting",
        "expected_workflow_type": "rewrite",
        "expected_output_type": "rewrite",
        "expected_target_type": "note",
    },
]
