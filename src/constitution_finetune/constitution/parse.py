"""Parse constitution markdown into structured Principles."""

from __future__ import annotations

import re

from .principles import (
    Constitution,
    HEADING_CATEGORY_MAP,
    Principle,
    PrincipleCategory,
)


def _classify_section(heading: str) -> PrincipleCategory:
    """Map a section heading to a PrincipleCategory."""
    lower = heading.lower()
    for keyword, category in HEADING_CATEGORY_MAP.items():
        if keyword in lower:
            return category
    return PrincipleCategory.COMPLIANCE


def _extract_principles_from_block(text: str) -> list[str]:
    """Extract individual principles from a markdown text block.

    Handles:
    - Numbered lists: "1. **Bold**: explanation"
    - Bullet lists: "* **Bold**: explanation" or "- **Bold**: explanation"
    - Plain paragraphs separated by blank lines
    """
    principles: list[str] = []

    # Try numbered/bullet list items first
    # Match lines starting with number+dot, asterisk, or dash, possibly with bold text
    list_pattern = re.compile(
        r"^(?:\d+\.\s+|\*\s+|-\s+)(.+?)(?=\n(?:\d+\.\s+|\*\s+|-\s+)|\n\n|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    matches = list_pattern.findall(text)

    if matches:
        for m in matches:
            cleaned = _clean_principle(m.strip())
            if _is_valid_principle(cleaned):
                principles.append(cleaned)

    # Also extract substantial paragraphs not captured by list patterns
    # Split on double newlines for paragraph-level extraction
    if not principles:
        paragraphs = re.split(r"\n\n+", text)
        for para in paragraphs:
            cleaned = _clean_principle(para.strip())
            if _is_valid_principle(cleaned):
                principles.append(cleaned)

    return principles


def _clean_principle(text: str) -> str:
    """Clean up a principle's text."""
    # Remove leading bold markers like "**Truthful**:"
    text = re.sub(r"^\*\*[^*]+\*\*:\s*", "", text)
    # Remove leading bold markers without colon
    text = re.sub(r"^\*\*[^*]+\*\*\s*", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Remove markdown formatting artifacts
    text = text.strip("*_")
    return text


def _is_valid_principle(text: str) -> bool:
    """Filter out headings, short fragments, and non-principles."""
    if len(text) < 40:
        return False
    if text.startswith("#"):
        return False
    # Must contain actual directive-like content
    return True


def _replace_model_name(text: str, target_name: str) -> str:
    """Replace 'Claude' with the target model name."""
    # Replace "Claude" but not in URLs or attribution contexts
    text = re.sub(r"\bClaude\b(?!'s Constitution)", target_name, text)
    return text


def parse_constitution(
    markdown: str,
    target_model_name: str = "Llama",
    source_url: str = "",
) -> Constitution:
    """Parse constitution markdown into a structured Constitution object."""
    constitution = Constitution(
        source_url=source_url,
        target_model_name=target_model_name,
    )

    # Split by headings (levels 1-3)
    heading_pattern = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)
    splits = heading_pattern.split(markdown)

    # splits alternates: [preamble, level, heading, content, level, heading, content, ...]
    # Process preamble if substantial
    if splits[0].strip():
        _add_principles(
            constitution, splits[0], "Preamble", PrincipleCategory.COMPLIANCE, target_model_name
        )

    # Process each section
    i = 1
    while i < len(splits) - 2:
        _level = splits[i]  # e.g. "##"
        heading = splits[i + 1].strip()
        content = splits[i + 2] if i + 2 < len(splits) else ""
        category = _classify_section(heading)

        _add_principles(constitution, content, heading, category, target_model_name)
        i += 3

    return constitution


def _add_principles(
    constitution: Constitution,
    content: str,
    section: str,
    category: PrincipleCategory,
    target_model_name: str,
) -> None:
    """Extract and add principles from a content block."""
    raw_principles = _extract_principles_from_block(content)
    base_index = len(constitution.principles)

    for idx, text in enumerate(raw_principles):
        text = _replace_model_name(text, target_model_name)
        principle = Principle(
            text=text,
            category=category,
            section=section,
            index=base_index + idx,
        )
        constitution.principles.append(principle)
