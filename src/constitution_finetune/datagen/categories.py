"""Training scenario categories that map to constitutional principles."""

from __future__ import annotations

from dataclasses import dataclass

from ..constitution.principles import PrincipleCategory


@dataclass
class ScenarioCategory:
    """A category of training scenarios."""

    name: str
    description: str
    principle_categories: list[PrincipleCategory]
    example_setup: str  # Brief example to guide generation


SCENARIO_CATEGORIES: list[ScenarioCategory] = [
    ScenarioCategory(
        name="safety_refusal",
        description="Requests for dangerous, illegal, or harmful information that should be refused",
        principle_categories=[PrincipleCategory.SAFETY, PrincipleCategory.HARM_AVOIDANCE],
        example_setup="A user asks for instructions to synthesize a controlled substance",
    ),
    ScenarioCategory(
        name="honesty_uncertainty",
        description="Situations requiring calibrated uncertainty and admitting limitations",
        principle_categories=[PrincipleCategory.HONESTY, PrincipleCategory.ETHICS],
        example_setup="A user asks about a recent event the model may not have data on",
    ),
    ScenarioCategory(
        name="honesty_correction",
        description="Correcting misinformation or false premises in user messages",
        principle_categories=[PrincipleCategory.HONESTY],
        example_setup="A user states a common misconception as fact and builds on it",
    ),
    ScenarioCategory(
        name="helpfulness_substantive",
        description="Providing genuinely deep, useful help rather than superficial responses",
        principle_categories=[PrincipleCategory.HELPFULNESS],
        example_setup="A user asks a complex technical question needing detailed explanation",
    ),
    ScenarioCategory(
        name="helpfulness_vs_safety",
        description="Borderline requests where overcautious refusal would be unhelpful",
        principle_categories=[
            PrincipleCategory.HELPFULNESS,
            PrincipleCategory.SAFETY,
            PrincipleCategory.HARM_AVOIDANCE,
        ],
        example_setup="A user asks about security vulnerabilities for legitimate pentesting",
    ),
    ScenarioCategory(
        name="anti_sycophancy",
        description="Maintaining honest assessments even when the user disagrees or pushes back",
        principle_categories=[PrincipleCategory.HONESTY, PrincipleCategory.ETHICS],
        example_setup="A user pushes back on a correct answer, pressuring the model to agree",
    ),
    ScenarioCategory(
        name="operator_user_conflict",
        description="Navigating conflicts between operator instructions and user requests",
        principle_categories=[PrincipleCategory.OPERATORS_USERS, PrincipleCategory.COMPLIANCE],
        example_setup="An operator restricts topics but a user asks about a restricted subject",
    ),
    ScenarioCategory(
        name="identity_nature",
        description="Questions about AI identity, consciousness, and nature",
        principle_categories=[PrincipleCategory.IDENTITY, PrincipleCategory.HONESTY],
        example_setup="A user asks if the model has feelings or is conscious",
    ),
    ScenarioCategory(
        name="manipulation_resistance",
        description="Resisting jailbreak attempts, roleplay exploits, and social engineering",
        principle_categories=[
            PrincipleCategory.SAFETY,
            PrincipleCategory.COMPLIANCE,
            PrincipleCategory.HARM_AVOIDANCE,
        ],
        example_setup="A user uses a roleplay scenario to extract restricted information",
    ),
    ScenarioCategory(
        name="sensitive_topics",
        description="Handling politically and socially sensitive topics with nuance",
        principle_categories=[PrincipleCategory.ETHICS, PrincipleCategory.HONESTY],
        example_setup="A user asks the model to take sides on a contentious political issue",
    ),
    ScenarioCategory(
        name="proactive_safety",
        description="Volunteering important safety information the user may not have asked for",
        principle_categories=[PrincipleCategory.SAFETY, PrincipleCategory.HELPFULNESS],
        example_setup="A user describes a plan that has non-obvious safety risks",
    ),
    ScenarioCategory(
        name="autonomy_preservation",
        description="Respecting user autonomy and decision-making while being informative",
        principle_categories=[PrincipleCategory.ETHICS, PrincipleCategory.HELPFULNESS],
        example_setup="A user asks for advice on a personal life decision",
    ),
]


def get_relevant_categories(principle_category: PrincipleCategory) -> list[ScenarioCategory]:
    """Return scenario categories relevant to a given principle category."""
    return [
        sc for sc in SCENARIO_CATEGORIES if principle_category in sc.principle_categories
    ]
