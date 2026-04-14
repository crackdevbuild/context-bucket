from __future__ import annotations

import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from context_bucket.models import (
    ContextBucketDocumentImportItem,
    ContextBucketDocumentImportRequest,
    ContextBucketDocumentImportResponse,
    ContextBucketSourceUpsert,
)


def expand_import_paths(service: Any, target: Path, *, recursive: bool) -> list[Path]:
    if not target.exists():
        return []
    if target.is_file():
        return [target] if is_importable_file(path=target, text_extensions=service._text_import_extensions) else []
    pattern = "**/*" if recursive else "*"
    return [
        path for path in sorted(target.glob(pattern))
        if path.is_file() and is_importable_file(path=path, text_extensions=service._text_import_extensions)
    ]


def document_source_key(path: Path) -> str:
    return f"file:{str(path.resolve())}"
def is_importable_file(*, path: Path, text_extensions: set[str]) -> bool:
    return path.suffix.lower() in text_extensions


def read_importable_content(service: Any, path: Path) -> dict[str, Any] | None:
    if not is_importable_file(path=path, text_extensions=service._text_import_extensions):
        return None
    suffix = path.suffix.lower()
    try:
        raw_text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            raw_text = path.read_text(encoding="utf-8-sig")
        except Exception:
            return None
    except Exception:
        return None
    parsed_content = parse_imported_content(service, suffix, raw_text)
    cleaned = str(parsed_content.get("text") or "").strip()
    if not cleaned and parsed_content.get("structured_data") is None:
        return None
    parsed_content["text"] = cleaned or None
    return parsed_content


def parse_imported_content(service: Any, suffix: str, raw_text: str) -> dict[str, Any]:
    if suffix in {".html", ".htm"}:
        without_scripts = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", raw_text)
        stripped = re.sub(r"(?s)<[^>]+>", " ", without_scripts)
        return {"text": service._collapse_whitespace(html.unescape(stripped)), "structured_data": None, "data_schema": None}
    if suffix == ".xml":
        try:
            root = ElementTree.fromstring(raw_text)
            return {"text": service._collapse_whitespace(" ".join(text for text in root.itertext())), "structured_data": None, "data_schema": None}
        except Exception:
            return {"text": service._collapse_whitespace(raw_text), "structured_data": None, "data_schema": None}
    if suffix == ".ndjson":
        structured_items: list[Any] = []
        lines: list[str] = []
        for line in raw_text.splitlines():
            cleaned = line.strip()
            if not cleaned:
                continue
            try:
                payload = json.loads(cleaned)
                structured_items.append(payload)
                lines.append(service._flatten_json_text(payload))
            except Exception:
                lines.append(cleaned)
        structured_data: Any | None = structured_items or None
        data_schema = service._infer_data_schema(structured_data)
        return {
            "text": service._collapse_whitespace("\n".join(lines)),
            "structured_data": structured_data,
            "data_schema": data_schema,
        }
    if suffix == ".json":
        try:
            payload = json.loads(raw_text)
            return {
                "text": service._collapse_whitespace(service._flatten_json_text(payload)),
                "structured_data": payload,
                "data_schema": service._infer_data_schema(payload),
            }
        except Exception:
            return {"text": service._collapse_whitespace(raw_text), "structured_data": None, "data_schema": None}
    return {"text": service._collapse_whitespace(raw_text), "structured_data": None, "data_schema": None}


async def import_path(service: Any, payload: ContextBucketDocumentImportRequest) -> ContextBucketDocumentImportResponse:
    target = Path(payload.path).expanduser().resolve()
    items: list[ContextBucketDocumentImportItem] = []
    paths = service._expand_import_paths(target, recursive=payload.recursive)
    if not paths:
        return ContextBucketDocumentImportResponse(
            imported=0,
            skipped=1,
            items=[
                ContextBucketDocumentImportItem(
                    path=str(target),
                    source_key=service._document_source_key(target),
                    status="skipped",
                    reason="path_not_found_or_unsupported",
                )
            ],
        )
    imported = 0
    skipped = 0
    for file_path in paths:
        source_key = service._document_source_key(file_path)
        imported_content = service._read_importable_content(file_path)
        if imported_content is None:
            skipped += 1
            items.append(
                ContextBucketDocumentImportItem(
                    path=str(file_path),
                    source_key=source_key,
                    status="skipped",
                    reason="unsupported_or_unreadable",
                )
            )
            continue
        metadata = dict(payload.metadata)
        metadata.update(
            {
                "import_path": str(file_path),
                "import_name": file_path.name,
                "import_suffix": file_path.suffix.lower(),
            }
        )
        create_payload = ContextBucketSourceUpsert(
            source_key=source_key,
            text=imported_content["text"],
            kind=payload.kind,
            scope=payload.scope,
            user_id=payload.user_id,
            session_id=payload.session_id,
            title=file_path.name,
            source_type="imported_document",
            tags=list(payload.tags),
            metadata=metadata,
            structured_data=imported_content.get("structured_data"),
            data_schema=payload.data_schema or imported_content.get("data_schema"),
            external_id=str(file_path),
            policy=payload.policy,
            source_updated_at=datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc),
            source_last_synced_at=datetime.now(timezone.utc),
        )
        if payload.import_mode == "create":
            record = await service.ingest_source(create_payload)
        else:
            record = await service.upsert_source(create_payload)
        if record is None:
            skipped += 1
            items.append(
                ContextBucketDocumentImportItem(
                    path=str(file_path),
                    source_key=source_key,
                    status="skipped",
                    reason="duplicate_or_rejected",
                )
            )
            continue
        imported += 1
        items.append(
            ContextBucketDocumentImportItem(
                path=str(file_path),
                source_key=source_key,
                status="imported",
                record_id=record.id,
            )
        )
    return ContextBucketDocumentImportResponse(imported=imported, skipped=skipped, items=items)
