# Architecture

`context-bucket` is a local-first workflow mediation layer for AI applications.

## Core pieces

- `context_bucket/service.py`: thin public facade that coordinates the internal modules
- `context_bucket/models.py`: request/response and record models
- `context_bucket/structured.py`: schema resolution, structured field extraction, and content preparation
- `context_bucket/retrieval.py`: lexical/semantic retrieval helpers, scoring, and compression
- `context_bucket/preferences.py`: user workflow preference merge and summarization logic
- `context_bucket/storage.py`: local file/SQLite record and index persistence
- `context_bucket/ingest.py`: record creation, source lifecycle, and upsert/delete flow
- `context_bucket/assembly.py`: context assembly and model-ready block preparation
- `context_bucket/importers.py`: file import expansion and document parsing
- `context_bucket/task_envelope.py`: workflow intent derivation and task envelope construction
- `context_bucket/evaluation.py`: local evaluation runs, suites, comparison, and gating
- `context_bucket/audit.py`: retrieval audit logging
- `context_bucket/training.py`: training-line serialization and local training export helpers
- `context_bucket/cli.py`: CLI for local workflows
- `context_bucket/settings.py`: storage and backend configuration

## Public interfaces

- Python library: `ContextBucketService` and the exported request/response models
- Python library evaluation helpers: retrieval/assembly evaluation, saved suites, run comparison, and gating
- CLI: `context-bucket`
- No HTTP server surface in this OSS cut

## Primary workflow shapes

- `reply`: active source plus user workflow preference become a sendable-response task envelope
- `summarize`: retrieved notes and documents are assembled into compact summary context
- `research`: findings and reports are ranked into evidence-heavy research context
- `rewrite`: active draft material is re-framed through workflow preference and drafting context

## Storage model

- Records are the source of truth.
- Records are chunked on ingest.
- Chunk and record indexes are maintained locally.
- File storage is the default backend.
- SQLite can be enabled for record and index backends.
- First-class workflow preference records can be stored and updated locally.

## Mediation pipeline

1. Ingest source data and optional structured schemas.
2. Update or load the user's workflow preference record.
3. Filter visible records by scope, source metadata, and freshness.
4. Generate lexical and embedding-style candidates.
5. Rerank candidates with lexical, semantic, recency, and scope signals.
6. Deduplicate overlapping results.
7. Derive lightweight workflow intent from the caller request plus active context.
8. Assemble context blocks and build a task envelope for the downstream model.

## First-class mediation objects

- `user_workflow_preference`: stable user-level AI workflow preferences such as autonomy, clarification, brevity, structure, initiative, evidence, and style
- `workflow_intent`: compact task interpretation for the current call
- `task_envelope`: normalized model-facing package that combines intent, preference, and retrieved context
- evaluation runs and suites: local quality checks for retrieval and assembly behavior

## Current embedder

The built-in embedder is a local hashing embedder. It creates deterministic vectors without network calls or external model dependencies. It is fast and local, but it is not a trained semantic model.
