"""Intent inference helpers for goal-aware source routing."""

from __future__ import annotations

from dataclasses import dataclass
import re


def goal_tokens(goal: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", goal.lower()))


def infer_entity_intent(goal: str) -> str:
    tokens = goal_tokens(goal)
    if {"program", "programs"} & tokens and ({"ncaa", "college"} & tokens or "basketball" in tokens):
        return "school"
    if {"player", "players"} & tokens:
        return "player"
    if {"team", "teams"} & tokens:
        return "team"
    if {"club", "clubs"} & tokens:
        return "club"
    if {"company", "companies", "startup", "startups"} & tokens:
        return "company"
    if {"bank", "banks"} & tokens:
        return "bank"
    if {"school", "schools"} & tokens:
        return "school"
    if {"state", "states"} & tokens:
        return "state"
    return "entity"


def infer_domain_intent(goal: str) -> str:
    tokens = goal_tokens(goal)
    if {"nba"} & tokens:
        return "nba"
    if {"ncaa", "college"} & tokens and "basketball" in tokens:
        return "ncaa-basketball"
    if {"soccer", "football"} & tokens:
        return "soccer"
    if {"startup", "startups", "unicorn", "unicorns"} & tokens:
        return "startup"
    if {"bank", "banks"} & tokens:
        return "bank"
    return "generic"


@dataclass(frozen=True, slots=True)
class GoalCardinality:
    """Expected row cardinality implied by the user's goal."""

    count: int
    exact: bool = True
    reason: str = ""


def infer_goal_cardinality(goal: str) -> GoalCardinality | None:
    lowered = goal.lower()
    tokens = goal_tokens(goal)

    numeric_cardinality = _infer_numeric_cardinality(lowered)
    if numeric_cardinality is not None:
        return numeric_cardinality

    if infer_domain_intent(goal) == "nba" and infer_entity_intent(goal) == "team":
        return GoalCardinality(count=30, exact=True, reason="all_nba_teams")

    if {"u", "s", "us"} & tokens and infer_entity_intent(goal) == "state":
        return GoalCardinality(count=50, exact=True, reason="us_states")
    if "u.s." in lowered and infer_entity_intent(goal) == "state":
        return GoalCardinality(count=50, exact=True, reason="us_states")

    if "fortune 500" in lowered:
        return GoalCardinality(count=500, exact=True, reason="fortune_500")

    if "ncaa" in lowered and "basketball" in lowered and any(
        phrase in lowered for phrase in ("division i", "programs", "teams", "team statistics")
    ):
        return GoalCardinality(count=365, exact=False, reason="ncaa_programs_approx")

    return None


def _infer_numeric_cardinality(lowered_goal: str) -> GoalCardinality | None:
    patterns = (
        r"\btop\s+(\d{1,4})\b",
        r"\blargest\s+(\d{1,4})\b",
        r"\bfirst\s+(\d{1,4})\b",
        r"\bfor\s+(\d{1,4})\s+(?:teams|players|clubs|companies|banks|schools|states)\b",
        r"\b(\d{1,4})\s+(?:teams|players|clubs|companies|banks|schools|states)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, lowered_goal)
        if not match:
            continue
        count = int(match.group(1))
        if 1 <= count <= 5000:
            return GoalCardinality(count=count, exact=True, reason=f"numeric_goal:{pattern}")
    return None
