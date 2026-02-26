"""Dataclasses for structured constitutional principles."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class PrincipleCategory(str, Enum):
    SAFETY = "safety"
    ETHICS = "ethics"
    HONESTY = "honesty"
    HELPFULNESS = "helpfulness"
    HARM_AVOIDANCE = "harm_avoidance"
    COMPLIANCE = "compliance"
    IDENTITY = "identity"
    OPERATORS_USERS = "operators_users"


# Map section heading keywords to categories
HEADING_CATEGORY_MAP: dict[str, PrincipleCategory] = {
    "safe": PrincipleCategory.SAFETY,
    "ethic": PrincipleCategory.ETHICS,
    "honest": PrincipleCategory.HONESTY,
    "helpful": PrincipleCategory.HELPFULNESS,
    "harm": PrincipleCategory.HARM_AVOIDANCE,
    "guideline": PrincipleCategory.COMPLIANCE,
    "identity": PrincipleCategory.IDENTITY,
    "operator": PrincipleCategory.OPERATORS_USERS,
    "user": PrincipleCategory.OPERATORS_USERS,
    "principal": PrincipleCategory.OPERATORS_USERS,
    "conflict": PrincipleCategory.OPERATORS_USERS,
    "mission": PrincipleCategory.IDENTITY,
    "core value": PrincipleCategory.ETHICS,
    "constitution": PrincipleCategory.COMPLIANCE,
    "approach": PrincipleCategory.COMPLIANCE,
    "instruc": PrincipleCategory.COMPLIANCE,
}


@dataclass
class Principle:
    """A single constitutional principle."""

    text: str
    category: PrincipleCategory
    section: str  # Source section heading
    index: int  # Position within section

    def __str__(self) -> str:
        return self.text


@dataclass
class Constitution:
    """A parsed set of constitutional principles."""

    principles: list[Principle] = field(default_factory=list)
    source_url: str = ""
    target_model_name: str = "Llama"

    def by_category(self, category: PrincipleCategory) -> list[Principle]:
        return [p for p in self.principles if p.category == category]

    @property
    def categories(self) -> set[PrincipleCategory]:
        return {p.category for p in self.principles}

    def __len__(self) -> int:
        return len(self.principles)

    def __iter__(self):
        return iter(self.principles)
