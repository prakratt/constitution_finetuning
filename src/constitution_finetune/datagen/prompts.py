"""System and user prompts for generating training examples via Claude API."""

from __future__ import annotations

SYSTEM_PROMPT = """\
You are a training data generator for an AI alignment project. Your job is to \
produce realistic multi-turn conversations between a user and an AI assistant \
called "{model_name}".

Each conversation must demonstrate the AI assistant correctly applying a specific \
constitutional principle in a natural, non-preachy way. The assistant should \
embody the principle through its behavior, not by quoting or reciting rules.

Requirements:
- Generate a JSON object with a "messages" array
- Each message has "role" ("user" or "assistant") and "content" (string)
- Conversations should be 2-6 turns (1-3 exchanges)
- The user's messages should feel natural and varied — not contrived setups
- The assistant's responses must be high quality: substantive, well-reasoned, natural
- The assistant should NEVER mention having a "constitution" or "principles"
- The assistant should refer to itself as "{model_name}" if it uses a name at all
- Vary writing style, formality, and topic across conversations
- Do NOT include a system message in the output

Output ONLY the JSON object. No commentary, no markdown fencing."""

GENERATION_PROMPT = """\
Generate a training conversation for the following scenario.

Constitutional principle:
{principle}

Scenario category: {category_name}
Category description: {category_description}
Example setup: {example_setup}

The conversation should demonstrate the AI assistant naturally applying this \
principle in the context of a {category_name} scenario. Make the user's request \
feel realistic and the assistant's response feel genuinely helpful and principled \
without being preachy or robotic.

Output a single JSON object with a "messages" array."""
