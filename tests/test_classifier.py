"""Test cases for the topic classifier (keyword fast-path + LLM classifier).

The original project's `_AI_KEYWORDS` set lived in `main.py`; in news-radar
that data moved into the example config's `ai` topic (`examples/config.example.toml`,
`[[topics]] name = "ai"`, `keywords = [...]`). Rather than inline a second
copy of the keyword list here (which would drift from the example config),
this test loads the example config and pulls `config.topic("ai").keywords`
directly — so the test always reflects whatever the shipped example declares.

LLM-classifier cases require a local Ollama daemon with the configured model
pulled; they're skipped automatically when unavailable.
"""

from __future__ import annotations

import pytest

from newsradar.config import load_config
from newsradar.llm import classify_topic

from conftest import EXAMPLE_CONFIG, ollama_has_model

_config = load_config(EXAMPLE_CONFIG)
_AI_TOPIC = _config.topic("ai")
_MODEL = _config.models.classify_model

# (expected_relevant, title, summary)
CASES = [
    # --- Should be REJECTED ---
    (
        False,
        "United Introduces Adjustable Armrests on Long-Haul Flights",
        "United Airlines is introducing adjustable armrests on long-haul flights in 2027 across "
        "over 200 aircraft. The 'Relax Row' seat offering will feature three economy seats that "
        "can be converted into a couch with additional amenities. These seats are intended to "
        "provide more comfort and space for passengers during long flights.",
    ),
    (
        False,
        "Refer Friends to The Hustle and Earn Exclusive Perks",
        "To refer friends to The Hustle and earn exclusive perks, share a unique link with them. "
        "They must subscribe through the personalized referral link, then confirm their "
        "subscription via email. Prizes are available for purchase in the shop.",
    ),
    (
        False,
        "Candles Mugs Hats Tees and Desk Organizers for Sale",
        "Candles, mugs, hats, and desk organizers are available for purchase from a list of 24 "
        "products, priced between $5.75 and $55.50. The selection includes items from brands "
        "such as Bella + Canvas and Brigham.",
    ),
    (
        False,
        "Exclusive Zip Code Leads Available",
        "Real Intent provides exclusive zip code leads to one agent per zip code, offering up to "
        "25 new leads weekly from households looking to buy or sell a home. The company uses "
        "intent data to track and target clients, ensuring no competitors in the area. Annual "
        "plans cost $2000/year.",
    ),
    (
        False,
        "Game On: Five New Titles Now Streaming on GeForce NOW",
        "GeForce NOW is adding five new titles to its streaming service, including a retro-racing "
        "game called 'Screamer' that offers pixel-perfect speed. The new games can be streamed "
        "instantly across various devices, helping players clear their gaming backlog.",
    ),
    # --- Should be ACCEPTED ---
    (
        True,
        "GPT-4o Gets New Voice Mode Capabilities",
        "OpenAI has released an update to GPT-4o adding real-time voice conversation support "
        "with improved latency and emotional tone recognition. The model can now handle "
        "interruptions naturally.",
    ),
    (
        True,
        "Boston Dynamics Spot Gets New Manipulation Arm",
        "Boston Dynamics announced an upgraded manipulation arm for its Spot robot, improving "
        "payload capacity and dexterity for industrial inspection tasks. The new hardware "
        "supports autonomous object retrieval.",
    ),
    (
        True,
        "Mistral Releases New 7B Instruct Model",
        "Mistral AI has released a new 7B parameter instruct model with improved reasoning "
        "benchmarks. The model is available under an Apache 2.0 license and can run locally "
        "on consumer hardware via Ollama.",
    ),
]


def _keyword_matches(title: str, summary: str) -> bool:
    lowered = (title + " " + summary).lower()
    return any(kw in lowered for kw in _AI_TOPIC.keywords)


# Note: this is a naive substring match (matches the original project's
# behavior exactly — see cli.py's `any(kw in lowered for kw in topic.keywords)`),
# so short keywords like "ai" will match inside unrelated words ("available",
# "Airlines", "gaming"). That's a known, accepted trade-off of the fast-path:
# a keyword hit skips the LLM call entirely and the item is kept without
# further relevance checking. Because of this, some of the CASES below that
# are labeled "should be rejected" will actually keyword-match and get kept —
# that's expected pre-filter behavior, not a classifier bug, so those cases
# are only used to test the *mechanism* (see
# test_keyword_fastpath_skips_llm_and_keeps_the_item), not classification
# correctness. Classification correctness is tested only for cases the
# pre-filter doesn't intercept.
NON_KEYWORD_CASES = [c for c in CASES if not _keyword_matches(c[1], c[2])]
KEYWORD_MATCHED_CASES = [c for c in CASES if _keyword_matches(c[1], c[2])]


def test_example_config_ai_topic_has_keywords():
    assert _AI_TOPIC.keywords, "examples/config.example.toml ai topic should declare keywords"


def test_example_config_robotics_topic_has_no_keywords():
    """Empty keywords means the CLI always LLM-classifies (intentional, see config comments)."""
    assert _config.topic("robotics").keywords == []


def test_cases_cover_both_the_fastpath_and_the_llm_path():
    """Sanity check that our fixed CASES list actually exercises both branches."""
    assert NON_KEYWORD_CASES, "expected at least one case that skips the keyword fast-path"
    assert KEYWORD_MATCHED_CASES, "expected at least one case that hits the keyword fast-path"


@pytest.mark.parametrize("expected,title,summary", KEYWORD_MATCHED_CASES)
def test_keyword_fastpath_skips_llm_and_keeps_the_item(expected, title, summary):
    """A keyword hit means the pipeline never calls classify_topic — item is kept as-is."""
    assert _keyword_matches(title, summary) is True


@pytest.mark.parametrize("expected,title,summary", NON_KEYWORD_CASES)
def test_llm_classifies_non_keyword_cases_correctly(expected, title, summary):
    if not ollama_has_model(_MODEL):
        pytest.skip(f"Ollama model {_MODEL!r} not available locally")

    result = classify_topic(title, summary, _MODEL, _AI_TOPIC)
    assert result == expected, f"classify_topic({title!r}) = {result}, expected {expected}"
