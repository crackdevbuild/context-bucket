from __future__ import annotations

import re
from typing import Any, Callable

from context_bucket.models import ContextBucketDataSchema, ContextBucketStructuredField


def flatten_json_text(payload: Any) -> str:
    if isinstance(payload, dict):
        parts: list[str] = []
        for key, value in payload.items():
            parts.append(f"{key}: {flatten_json_text(value)}")
        return " ".join(parts)
    if isinstance(payload, list):
        return " ".join(flatten_json_text(item) for item in payload)
    return str(payload)


def prepare_content(
    text: str | None,
    structured_data: Any | None,
    data_schema: ContextBucketDataSchema | None,
    *,
    lexical_tokens_fn: Callable[[str], list[str]],
    semantic_terms_fn: Callable[[str], list[str]],
) -> dict[str, Any]:
    normalized_data = normalize_structured_data(structured_data)
    resolved_schema = resolve_data_schema(normalized_data, data_schema)
    structured_fields = extract_structured_fields(normalized_data)
    generated_text = structured_text(
        structured_fields,
        resolved_schema.primary_text_paths if resolved_schema else [],
        include_all_fields=not declared_user_intent_schema(resolved_schema),
    )
    cleaned_text = collapse_whitespace((text or "").strip())
    if cleaned_text and generated_text and generated_text.lower() not in cleaned_text.lower():
        combined_text = collapse_whitespace(f"{cleaned_text}\n{generated_text}")
    else:
        combined_text = cleaned_text or generated_text
    analysis_text = combined_text
    if structured_fields and not declared_user_intent_schema(resolved_schema):
        analysis_text = collapse_whitespace(
            " ".join(
                [combined_text]
                + [field.path.replace(".", " ").replace("[", " ").replace("]", " ") for field in structured_fields]
            )
        )
    return {
        "text": combined_text,
        "structured_data": normalized_data,
        "data_schema": resolved_schema,
        "structured_fields": structured_fields,
        "lexical_tokens": lexical_tokens_fn(analysis_text),
        "semantic_terms": semantic_terms_fn(analysis_text),
    }


def resolve_data_schema(
    structured_data: Any | None,
    data_schema: ContextBucketDataSchema | None,
) -> ContextBucketDataSchema | None:
    if structured_data is None and data_schema is None:
        return None
    inferred = infer_data_schema(structured_data)
    if data_schema is None:
        return inferred
    field_paths = list(dict.fromkeys([*list(data_schema.field_paths), *(inferred.field_paths if inferred else [])]))
    primary_text_paths = list(
        dict.fromkeys([*list(data_schema.primary_text_paths), *((inferred.primary_text_paths if inferred else []))])
    )
    return ContextBucketDataSchema(
        schema_name=data_schema.schema_name,
        schema_mode=data_schema.schema_mode,
        root_type=data_schema.root_type or (inferred.root_type if inferred else "object"),
        field_paths=field_paths,
        primary_text_paths=primary_text_paths,
    )


def infer_data_schema(structured_data: Any | None) -> ContextBucketDataSchema | None:
    if structured_data is None:
        return None
    fields = extract_structured_fields(structured_data)
    field_paths = [field.path for field in fields]
    return ContextBucketDataSchema(
        schema_mode="inferred",
        root_type=value_type(structured_data),
        field_paths=field_paths,
        primary_text_paths=infer_primary_text_paths(fields),
    )


def extract_structured_fields(payload: Any, prefix: str = "") -> list[ContextBucketStructuredField]:
    if payload is None:
        return []
    if isinstance(payload, dict):
        items: list[ContextBucketStructuredField] = []
        for key, value in payload.items():
            child_path = f"{prefix}.{key}" if prefix else str(key)
            items.extend(extract_structured_fields(value, child_path))
        return items
    if isinstance(payload, list):
        items = []
        for index, value in enumerate(payload):
            child_path = f"{prefix}[{index}]" if prefix else f"[{index}]"
            items.extend(extract_structured_fields(value, child_path))
        return items
    value_text = collapse_whitespace(str(payload))
    if not value_text:
        return []
    return [ContextBucketStructuredField(path=prefix or "value", value_text=value_text, value_type=value_type(payload))]


def infer_primary_text_paths(fields: list[ContextBucketStructuredField]) -> list[str]:
    preferred = ("text", "body", "content", "summary", "description", "name", "title", "message", "note", "notes")
    selected: list[str] = []
    for field in fields:
        leaf = field.path.split(".")[-1].split("[")[0].lower()
        if leaf in preferred:
            selected.append(field.path)
    return list(dict.fromkeys(selected))


def structured_text(
    fields: list[ContextBucketStructuredField],
    primary_paths: list[str],
    *,
    include_all_fields: bool = True,
) -> str:
    if not fields:
        return ""
    ordered_fields = fields
    if primary_paths:
        primary_set = set(primary_paths)
        primary_fields = [field for field in fields if field.path in primary_set]
        secondary_fields = [field for field in fields if field.path not in primary_set]
        ordered_fields = primary_fields + secondary_fields
        if not include_all_fields:
            ordered_fields = primary_fields
    elif not include_all_fields:
        ordered_fields = fields
    parts = [f"{field.path}: {field.value_text}" for field in ordered_fields]
    return collapse_whitespace(" ".join(parts))


def declared_user_intent_schema(data_schema: ContextBucketDataSchema | None) -> bool:
    if data_schema is None:
        return False
    return data_schema.schema_mode == "declared" and (data_schema.schema_name or "").strip().lower() == "user_intent"


def normalize_structured_data(payload: Any | None) -> Any | None:
    if isinstance(payload, dict):
        return {str(key): normalize_structured_data(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [normalize_structured_data(item) for item in payload]
    return payload


def value_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int) and not isinstance(value, bool):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    if value is None:
        return "null"
    return "string"


def collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
