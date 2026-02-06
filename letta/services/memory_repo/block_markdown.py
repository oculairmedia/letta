"""Serialize and parse block data as Markdown with YAML frontmatter.

File format:
    ---
    description: "Who I am and how I approach work"
    limit: 20000
    ---
    My name is Memo. I'm a stateful coding assistant...

- Frontmatter fields are only rendered when they differ from defaults.
- Files without frontmatter are treated as value-only (backward compat).
"""

from typing import Any, Dict, Optional

import yaml

from letta.schemas.block import BaseBlock


def _get_field_default(field_name: str) -> Any:
    """Get the default value for a BaseBlock field."""
    field = BaseBlock.model_fields[field_name]
    return field.default


def serialize_block(
    value: str,
    *,
    description: Optional[str] = None,
    limit: Optional[int] = None,
    read_only: bool = False,
    metadata: Optional[dict] = None,
) -> str:
    """Serialize a block to Markdown with optional YAML frontmatter.

    Only non-default fields are included in the frontmatter.
    If all fields are at their defaults, no frontmatter is emitted.
    """
    # description and limit are always included in frontmatter.
    # read_only and metadata are only included when non-default.
    front: Dict[str, Any] = {}

    front["description"] = description
    front["limit"] = limit if limit is not None else _get_field_default("limit")

    if read_only != _get_field_default("read_only"):
        front["read_only"] = read_only
    if metadata and metadata != _get_field_default("metadata"):
        front["metadata"] = metadata

    # Use block style for cleaner YAML, default_flow_style=False
    yaml_str = yaml.dump(front, default_flow_style=False, sort_keys=False, allow_unicode=True).rstrip("\n")
    return f"---\n{yaml_str}\n---\n{value}"


def parse_block_markdown(content: str) -> Dict[str, Any]:
    """Parse a Markdown file into block fields.

    Returns a dict with:
        - "value": the body content after frontmatter
        - "description", "limit", "read_only", "metadata": from frontmatter (if present)

    If no frontmatter is detected, the entire content is treated as the value
    (backward compat with old repos that stored raw values).
    """
    if not content.startswith("---\n"):
        return {"value": content}

    # Find the closing --- delimiter
    end_idx = content.find("\n---\n", 4)
    if end_idx == -1:
        # No closing delimiter — treat entire content as value
        return {"value": content}

    yaml_str = content[4:end_idx]
    body = content[end_idx + 5 :]  # skip past \n---\n

    try:
        front = yaml.safe_load(yaml_str)
    except yaml.YAMLError:
        # Malformed YAML — treat entire content as value
        return {"value": content}

    if not isinstance(front, dict):
        return {"value": content}

    result: Dict[str, Any] = {"value": body}

    if "description" in front:
        result["description"] = front["description"]
    if "limit" in front:
        result["limit"] = front["limit"]
    if "read_only" in front:
        result["read_only"] = front["read_only"]
    if "metadata" in front:
        result["metadata"] = front["metadata"]

    return result
