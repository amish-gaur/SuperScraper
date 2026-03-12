"""Structured browser page state derived from agent-browser snapshots."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any


LINE_REF_RE = re.compile(r"@?(e\d+)\b", flags=re.IGNORECASE)
LINK_LINE_RE = re.compile(r'link\s+"([^"]+)"\s+\[ref=(e\d+)\]', flags=re.IGNORECASE)
HEADING_LINE_RE = re.compile(r'heading\s+"([^"]+)"', flags=re.IGNORECASE)
URL_RE = re.compile(r"https?://\S+", flags=re.IGNORECASE)
TABLE_SIGNAL_RE = re.compile(r"\b(row|cell|columnheader|table|listitem)\b", flags=re.IGNORECASE)
SUMMARY_BLOCKED_PHRASES = (
    "navigation item",
    "main menu",
    "jump to content",
    "create account",
    "log in",
    "search",
)


@dataclass(slots=True)
class PageElement:
    ref: str
    role: str
    label: str
    raw_line: str
    url: str | None = None


@dataclass(slots=True)
class PageState:
    current_url: str | None
    title: str | None
    visible_text_summary: str
    headings: list[str]
    clickable_elements: list[PageElement]
    table_like_lines: list[str]
    row_signal_count: int
    blocked_signals: list[str]
    raw_line_count: int

    def model_dump(self) -> dict[str, Any]:
        return {
            "current_url": self.current_url,
            "title": self.title,
            "visible_text_summary": self.visible_text_summary,
            "headings": self.headings,
            "clickable_elements": [asdict(element) for element in self.clickable_elements],
            "table_like_lines": self.table_like_lines,
            "row_signal_count": self.row_signal_count,
            "blocked_signals": self.blocked_signals,
            "raw_line_count": self.raw_line_count,
        }

    def compact_summary(self, *, max_links: int = 15, max_table_lines: int = 12) -> str:
        link_lines = [
            f"- {element.ref}: {element.label}"
            for element in self.clickable_elements[:max_links]
        ]
        table_lines = [f"- {line}" for line in self.table_like_lines[:max_table_lines]]
        blocked = ", ".join(self.blocked_signals) if self.blocked_signals else "none"
        title = self.title or "unknown"
        headings = ", ".join(self.headings[:8]) or "none"
        return (
            f"Title: {title}\n"
            f"Current URL: {self.current_url or 'unknown'}\n"
            f"Headings: {headings}\n"
            f"Row signal count: {self.row_signal_count}\n"
            f"Blocked signals: {blocked}\n"
            f"Visible text summary:\n{self.visible_text_summary}\n\n"
            f"Clickable elements:\n{chr(10).join(link_lines) if link_lines else '- none'}\n\n"
            f"Table-like lines:\n{chr(10).join(table_lines) if table_lines else '- none'}"
        )


class PageStateParser:
    """Build structured page state from a text snapshot."""

    BLOCKED_PHRASES = (
        "search results",
        "verify you are human",
        "cloudflare",
        "just a moment",
        "create account",
        "log in",
        "request that a redirect be created",
    )

    def parse(self, snapshot: str, *, current_url: str | None) -> PageState:
        lines = [line.strip() for line in snapshot.splitlines() if line.strip()]
        clickable_elements: list[PageElement] = []
        headings: list[str] = []
        table_like_lines: list[str] = []

        for line in lines:
            heading_match = HEADING_LINE_RE.search(line)
            if heading_match:
                headings.append(heading_match.group(1).strip())

            link_match = LINK_LINE_RE.search(line)
            if link_match:
                clickable_elements.append(
                    PageElement(
                        ref=f"@{link_match.group(2)}",
                        role="link",
                        label=link_match.group(1).strip(),
                        raw_line=line,
                        url=_extract_url(line),
                    )
                )
            elif LINE_REF_RE.search(line):
                ref = LINE_REF_RE.search(line)
                if ref is not None:
                    clickable_elements.append(
                        PageElement(
                            ref=f"@{ref.group(1).lower()}",
                            role=_infer_role(line),
                            label=_clean_line_text(line),
                            raw_line=line,
                            url=_extract_url(line),
                        )
                    )

            if TABLE_SIGNAL_RE.search(line):
                cleaned = _clean_line_text(line)
                if cleaned:
                    table_like_lines.append(cleaned)

        visible_summary = "\n".join(self._select_summary_lines(lines))
        blocked_signals = [phrase for phrase in self.BLOCKED_PHRASES if phrase in snapshot.lower()]
        title = headings[0] if headings else (clickable_elements[0].label if clickable_elements else None)
        return PageState(
            current_url=current_url,
            title=title,
            visible_text_summary=visible_summary[:2500],
            headings=headings[:20],
            clickable_elements=_dedupe_elements(clickable_elements)[:40],
            table_like_lines=_dedupe_strings(table_like_lines)[:30],
            row_signal_count=sum(1 for line in lines if TABLE_SIGNAL_RE.search(line) or LINE_REF_RE.search(line)),
            blocked_signals=blocked_signals,
            raw_line_count=len(lines),
        )

    def _select_summary_lines(
        self,
        lines: list[str],
        *,
        max_lines: int = 30,
        max_chars: int = 2500,
        min_score: int = 3,
    ) -> list[str]:
        """Prefer high-signal content across the full snapshot over the first viewport lines."""
        scored: list[tuple[int, int, str]] = []
        for index, line in enumerate(lines):
            cleaned = _clean_line_text(line)
            if not cleaned:
                continue

            lowered = cleaned.lower()
            if any(phrase in lowered for phrase in self.BLOCKED_PHRASES):
                continue
            if any(phrase in lowered for phrase in SUMMARY_BLOCKED_PHRASES):
                continue
            if line.lower().startswith(("button ", "textbox ", "searchbox ", "menuitem ")):
                continue

            score = 0
            if HEADING_LINE_RE.search(line):
                score += 5
            if TABLE_SIGNAL_RE.search(line):
                score += 4
            if LINK_LINE_RE.search(line):
                score += 3
            if LINE_REF_RE.search(line):
                score += 1
            if any(token.isdigit() for token in cleaned):
                score += 1
            if 6 <= len(cleaned.split()) <= 18:
                score += 1

            if score < min_score:
                continue
            scored.append((score, index, cleaned))

        if not scored:
            return []

        scored.sort(key=lambda item: (-item[0], item[1]))
        selected: list[tuple[int, str]] = []
        seen: set[str] = set()
        total_chars = 0
        for _, index, cleaned in scored:
            if cleaned in seen:
                continue
            next_chars = total_chars + len(cleaned) + (1 if selected else 0)
            if len(selected) >= max_lines or next_chars > max_chars:
                continue
            selected.append((index, cleaned))
            seen.add(cleaned)
            total_chars = next_chars

        selected.sort(key=lambda item: item[0])
        return [cleaned for _, cleaned in selected]


def _clean_line_text(line: str) -> str:
    text = LINE_REF_RE.sub("", line)
    text = URL_RE.sub("", text)
    text = re.sub(r"\[[^\]]*\]", "", text)
    text = re.sub(r'^(?:link|heading|listitem|row|cell|button)\s+', "", text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', " ", text)
    return text.strip(" -|:\t\"'")


def _extract_url(line: str) -> str | None:
    match = URL_RE.search(line)
    return match.group(0) if match else None


def _infer_role(line: str) -> str:
    lowered = line.lower()
    for role in ("link", "button", "textbox", "heading", "row", "cell", "listitem"):
        if lowered.startswith(role):
            return role
    return "element"


def _dedupe_elements(elements: list[PageElement]) -> list[PageElement]:
    deduped: list[PageElement] = []
    seen: set[tuple[str, str]] = set()
    for element in elements:
        key = (element.ref, element.label)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(element)
    return deduped


def _dedupe_strings(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped
