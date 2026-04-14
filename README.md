# context-bucket

Local-first context storage, retrieval, and workflow mediation for AI applications.

`context-bucket` is a lightweight memory and workflow-mediation layer for assistants, research tools, and workflow apps. It keeps data on local or user-controlled storage, ingests text and structured data, retrieves relevant chunks, learns compact workflow preferences, and prepares model-ready context blocks and task envelopes.

Core workflows:

- reply to a source such as an email
- summarize active notes or documents
- research across stored findings and reports
- rewrite text using user workflow preferences

## What it is

- a Python library for local context ingestion, retrieval, and workflow mediation
- a CLI for lightweight local workflows
- a file-first context store with optional SQLite backends

## What it includes

- local source ingest, upsert, and delete
- structured data support with schema-aware field extraction
- local file import for text, HTML, XML, JSON, and NDJSON
- chunked retrieval with a local hashing embedder
- first-class `user_workflow_preference` records and update flow
- `retrieve_context`, `assemble_context`, `prepare_context`, and `prepare_task_envelope`
- local evaluation helpers for retrieval and assembly quality checks
- file storage by default
- optional SQLite record and index backends

## What it does not include

- live SaaS connector platform
- worker queue or distributed execution
- external vector database dependencies

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

For contributor tooling and tests:

```bash
pip install -e '.[dev]'
```

## Python quickstart

```python
from context_bucket import ContextBucketAssembleRequest, ContextBucketService, ContextBucketSourceCreate

service = ContextBucketService()

await service.ingest_source(
    ContextBucketSourceCreate(
        scope="user",
        user_id="u1",
        source_key="client_profile",
        text="The client prefers concise email updates.",
    )
)

prepared = await service.prepare_context(
    ContextBucketAssembleRequest(
        query_text="draft a client update",
        user_id="u1",
        assembly_mode="drafting",
    )
)
```

A runnable example is in [examples/quickstart.py]
Multi-workflow examples are in [examples/workflows.py]

## CLI quickstart

```bash
context-bucket ingest-source --text "The client prefers concise email updates." --scope user --user-id u1 --source-key client_profile
context-bucket prepare-context "draft a client update" --user-id u1 --assembly-mode drafting
```

## Core workflow patterns

- `reply`: ingest the active source and call `prepare-task-envelope "answer the email"` or `prepare-context "answer the email" --assembly-mode drafting`
- `summarize`: ingest notes or docs and call `prepare-context "summarize the meeting notes"`
- `research`: ingest findings/reports and call `prepare-context "research ACME expansion risks" --assembly-mode research`
- `rewrite`: ingest the draft text and user preferences, then call `prepare-task-envelope "rewrite this note to be concise and professional"`

## Verify install

```bash
context-bucket --help
python -c "from context_bucket import ContextBucketService; print(ContextBucketService().__class__.__name__)"
```

## Workflow mediation

The system is intended to sit between the caller and the downstream model:

- ingest task data and source objects
- load or update the user's workflow preference profile
- derive lightweight workflow intent from the current call
- retrieve relevant memory and source context
- produce a compact task envelope for the model

This stays lightweight by using local storage, declared schemas, deterministic update rules, and compact preference summaries instead of heavy training infrastructure.

## Storage

By default the service writes local files under `.local/context_bucket`.

Optional advanced local modes:

- `CONTEXT_BUCKET_RECORD_BACKEND=sqlite`
- `CONTEXT_BUCKET_INDEX_BACKEND=sqlite`

## Current embedder

The retrieval pipeline uses a pluggable embedding interface with a local hashing embedder as the built-in backend. It is fully local and fast, but it is not a trained semantic model.

## Evaluation

The Python library exposes local evaluation helpers for retrieval and assembly quality checks, including saved suites and run comparisons. Local benchmark and behavior reports can be generated at the repo root during testing, but should stay out of commits unless explicitly approved.

