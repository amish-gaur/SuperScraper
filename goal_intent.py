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
    if {"laptop", "laptops", "notebook", "notebooks", "pc"} & tokens:
        return "laptop"
    if {"population", "gdp", "state", "states", "territory", "territories"} & tokens:
        return "public-data"
    return "generic"


@dataclass(frozen=True, slots=True)
class GoalDecomposition:
    """Structured interpretation of a user goal for source discovery."""

    raw_goal: str
    normalized_goal: str
    domain_intent: str
    entity_intent: str
    target_hint: str | None
    feature_hints: tuple[str, ...]
    modifiers: tuple[str, ...]
    temporal_scope: str
    row_granularity: str
    row_count_hint: int | None
    tokens: tuple[str, ...]


TARGET_HINT_PATTERNS: tuple[tuple[str, str], ...] = (
    ("salary", r"\bsalar(?:y|ies)\b|\bpaid\b|\bpay\b"),
    ("valuation", r"\bvalu(?:ation|ations)\b|\bunicorn\b"),
    ("population_growth", r"\bpopulation growth\b|\bgrowth rate\b"),
    ("revenue", r"\brevenue\b|\bsales\b"),
    ("price", r"\bprice\b|\bcost\b"),
    ("gdp", r"\bgdp\b|\bgross domestic product\b"),
    ("market_cap", r"\bmarket cap\b|\bmarket value\b"),
    ("wins", r"\bwins?\b|\bwinning percentage\b"),
)

FEATURE_HINT_PATTERNS: tuple[tuple[str, str], ...] = (
    ("performance_stats", r"\bperformance\b|\bstats?\b|\badvanced\b|\bon-court\b"),
    ("funding", r"\bfunding\b|\bventure\b|\braised\b"),
    ("gdp", r"\bgdp\b|\beconomic\b"),
    ("population", r"\bpopulation\b|\bdemographic\b"),
    ("hardware_specs", r"\bspecs?\b|\bcpu\b|\bgpu\b|\bram\b|\bdisplay\b"),
    ("efficiency", r"\befficiency\b|\bmargin\b|\brating\b"),
    ("geography", r"\bstate\b|\bregion\b|\blocation\b"),
)

MODIFIER_PATTERNS: tuple[tuple[str, str], ...] = (
    ("official", r"\bofficial\b"),
    ("historical", r"\bhistorical\b|\bhistory\b|\barchive\b"),
    ("current", r"\bcurrent\b|\bmodern\b|\blatest\b|\bthis season\b"),
    ("top_n", r"\btop\s+\d+\b|\blargest\s+\d+\b"),
)


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
        if {
            "historical",
            "history",
            "season",
            "seasons",
            "yearly",
            "multi",
            "multiseason",
        } & tokens:
            return None
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


def decompose_goal(goal: str) -> GoalDecomposition:
    """Convert a raw goal string into reusable discovery signals."""
    normalized_goal = re.sub(r"\s+", " ", goal.strip()).lower()
    tokens = tuple(sorted(goal_tokens(normalized_goal)))
    cardinality = infer_goal_cardinality(goal)

    target_hint = None
    for label, pattern in TARGET_HINT_PATTERNS:
        if re.search(pattern, normalized_goal):
            target_hint = label
            break

    feature_hints: list[str] = []
    for label, pattern in FEATURE_HINT_PATTERNS:
        if re.search(pattern, normalized_goal):
            feature_hints.append(label)

    modifiers: list[str] = []
    for label, pattern in MODIFIER_PATTERNS:
        if re.search(pattern, normalized_goal):
            modifiers.append(label)

    if re.search(r"\b(19\d{2}|20\d{2}|21\d{2})\b", normalized_goal) or "historical" in modifiers:
        temporal_scope = "historical"
    elif "current" in modifiers:
        temporal_scope = "current"
    else:
        temporal_scope = "recent"

    entity_intent = infer_entity_intent(goal)
    row_granularity = {
        "player": "player",
        "team": "team",
        "club": "club",
        "company": "company",
        "bank": "bank",
        "school": "school",
        "state": "state",
    }.get(entity_intent, "entity")

    return GoalDecomposition(
        raw_goal=goal,
        normalized_goal=normalized_goal,
        domain_intent=infer_domain_intent(goal),
        entity_intent=entity_intent,
        target_hint=target_hint,
        feature_hints=tuple(dict.fromkeys(feature_hints)),
        modifiers=tuple(dict.fromkeys(modifiers)),
        temporal_scope=temporal_scope,
        row_granularity=row_granularity,
        row_count_hint=cardinality.count if cardinality is not None else None,
        tokens=tokens,
    )


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
