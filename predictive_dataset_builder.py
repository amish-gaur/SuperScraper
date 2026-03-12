"""General-purpose assembly of wide predictive datasets from public HTML tables."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from io import StringIO
import logging
import re
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import requests
import urllib3

from entity_resolver import EntityResolver
from goal_intent import infer_domain_intent, infer_entity_intent, infer_goal_cardinality
from source_adapters import adapter_urls_for_goal
from source_health import FailureReason, fetch_url, REGISTRY
from source_ranker import SourceRanker


LOGGER = logging.getLogger(__name__)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
IDENTITY_COLUMNS = {"name", "entity_name", "raw_entity_name", "source_url"}


class DataQualityError(RuntimeError):
    """Raised when a candidate dataset fails minimum ML-ready quality thresholds."""

PREDICTIVE_KEYWORDS = {
    "predict",
    "predictor",
    "prediction",
    "predictive",
    "estimate",
    "estimation",
    "forecast",
    "forecasting",
    "model",
    "modeling",
    "modelling",
    "classification",
    "regression",
    "feature",
    "features",
    "target",
    "label",
}
ENTITY_COLUMN_CANDIDATES = (
    "bank",
    "club",
    "club_name",
    "company",
    "team",
    "school",
    "name",
    "player",
    "company",
    "organization",
    "school_name",
    "team_name",
    "entity_name",
    "title",
)
DROP_COLUMNS = {"rk", "rank", "unnamed_0", "unnamed_1"}
NCAA_TEAM_STAT_ALLOWLIST = {
    "145",
    "146",
    "147",
    "148",
    "149",
    "150",
    "151",
    "152",
    "168",
    "214",
    "215",
    "216",
    "217",
    "474",
    "518",
    "519",
    "931",
    "932",
    "1288",
}
NCAA_TEAM_STAT_NAMES = {
    "145": "scoring_offense",
    "146": "scoring_defense",
    "147": "scoring_margin",
    "148": "field_goal_pct",
    "149": "field_goal_pct_defense",
    "150": "free_throw_pct",
    "151": "rebound_margin",
    "152": "three_point_pct",
    "168": "winning_pct",
    "214": "blocks_per_game",
    "215": "steals_per_game",
    "216": "assists_per_game",
    "217": "turnovers_per_game",
    "474": "assist_turnover_ratio",
    "518": "three_point_pct_defense",
    "519": "turnover_margin",
    "931": "turnovers_forced_per_game",
    "932": "rebounds_per_game",
    "1288": "effective_fg_pct",
}
TEAMRANKINGS_NBA_METRICS = (
    "points-per-game",
    "opponent-points-per-game",
    "offensive-efficiency",
    "defensive-efficiency",
    "three-point-pct",
    "free-throw-pct",
    "turnovers-per-game",
)


@dataclass(slots=True)
class PredictiveDatasetBuilder:
    """Build a wide feature table by fusing compatible public HTML tables."""

    goal: str
    dataset_name: str
    starting_urls: list[str]
    timeout_seconds: float = 10.0
    minimum_rows: int = 20
    minimum_columns: int = 8
    max_candidate_urls: int = 12
    target_field: str | None = None
    core_feature_fields: list[str] = field(default_factory=list)
    domain_blacklist: set[str] = field(default_factory=set)
    source_ranker: SourceRanker = field(default_factory=SourceRanker)
    entity_resolver: EntityResolver = field(default_factory=EntityResolver)

    def is_applicable(self) -> bool:
        tokens = set(re.findall(r"[a-z0-9]+", self.goal.lower()))
        lowered_goal = self.goal.lower()
        ncaa_team_stats_goal = (
            "ncaa" in lowered_goal
            and "basketball" in lowered_goal
            and "team statistics" in lowered_goal
        )
        if not (tokens & PREDICTIVE_KEYWORDS):
            if not ncaa_team_stats_goal:
                return False
        non_predictive_phrases = (
            "dataset of",
            "table of",
            "list of",
        )
        if any(phrase in lowered_goal for phrase in non_predictive_phrases) and not (
            {"predict", "prediction", "predictive", "target", "label"} & tokens
            or ncaa_team_stats_goal
        ):
            return False
        return True

    def build(self) -> "PredictiveBuildResult | None":
        """Return a merged dataframe plus provenance when enough tabular signal is available."""
        prepared_frames: list[pd.DataFrame] = []
        numeric_rich_frame_seen = False
        for table_frame in self._candidate_frames():
            keyed = self._prepare_frame(table_frame)
            if keyed is None:
                continue
            prepared_frames.append(keyed)
            if self._numeric_feature_count(keyed) >= 3:
                numeric_rich_frame_seen = True

        if numeric_rich_frame_seen:
            prepared_frames = [
                frame
                for frame in prepared_frames
                if (
                    self._numeric_feature_count(frame) >= 3
                    or self._frame_has_target_signal(frame)
                    or self._frame_goal_alignment_score(frame) >= 8
                )
            ]

        primary_frame = self._best_single_frame(prepared_frames)
        merged = primary_frame.copy() if primary_frame is not None else None
        merged_provenance = dict(getattr(primary_frame, "attrs", {}).get("provenance_map", {})) if primary_frame is not None else {}
        if merged is not None:
            for frame in prepared_frames:
                if frame is primary_frame:
                    continue
                if self._frame_has_merge_coverage(merged, frame):
                    merged = self._merge_frames(merged, frame, how="left")
                    merged_provenance.update(getattr(frame, "attrs", {}).get("provenance_map", {}))
                    continue
                if self._frames_are_row_compatible(merged, frame):
                    merged = self._concat_frames(merged, frame)
                    merged_provenance.update(getattr(frame, "attrs", {}).get("provenance_map", {}))

        if merged is None:
            return None

        merged = self._finalize_frame(merged)
        self._enforce_fill_rate(merged)
        if (
            len(merged) >= self.minimum_rows
            and len(merged.columns) >= self.minimum_columns
            and self._frame_row_count_is_reasonable(merged)
        ):
            merged_provenance = self._finalize_provenance(merged, merged_provenance)
            return PredictiveBuildResult(dataframe=merged, provenance_map=merged_provenance)

        best_frame = self._best_single_frame(prepared_frames)
        if best_frame is None:
            return None
        best_frame = self._finalize_frame(best_frame)
        if (
            len(best_frame) < self.minimum_rows
            or len(best_frame.columns) < self.minimum_columns
            or not self._frame_row_count_is_reasonable(best_frame)
        ):
            return None
        self._enforce_fill_rate(best_frame)
        best_provenance = self._finalize_provenance(
            best_frame,
            dict(getattr(best_frame, "attrs", {}).get("provenance_map", {})),
        )
        return PredictiveBuildResult(dataframe=best_frame, provenance_map=best_provenance)

    def _candidate_frames(self) -> list[pd.DataFrame]:
        frames: list[pd.DataFrame] = []
        seen_urls: set[str] = set()
        for ranked_source in self._ranked_urls():
            url = ranked_source.url
            if url in seen_urls:
                continue
            seen_urls.add(url)
            extracted = self._extract_tables(url)
            if not extracted:
                continue
            frames.extend(extracted)
        return frames

    def _ranked_urls(self) -> list[Any]:
        return self.source_ranker.rank(self.goal, self._expand_urls())[: self.max_candidate_urls]

    def _expand_urls(self) -> list[str]:
        expanded = list(self.starting_urls)
        if not expanded:
            expanded.extend(adapter_urls_for_goal(self.goal))
            expanded.extend(self._goal_supplemental_urls())
        for url in list(expanded):
            expanded.extend(self._derived_companion_urls(url))
        return [
            url
            for url in expanded
            if (
                url.startswith(("http://", "https://"))
                and _root_domain(url) not in self.domain_blacklist
                and not REGISTRY.should_cooldown(url)
            )
        ]

    def _goal_supplemental_urls(self) -> list[str]:
        lowered = self.goal.lower()
        year = self._infer_season_year(lowered)
        if infer_domain_intent(self.goal) == "nba" and infer_entity_intent(self.goal) == "team":
            return self._nba_team_stat_urls(year)
        if infer_domain_intent(self.goal) == "nba" and infer_entity_intent(self.goal) == "player":
            return self._nba_player_stat_urls()
        if infer_domain_intent(self.goal) == "startup":
            return self._startup_company_urls()
        if infer_domain_intent(self.goal) == "soccer" and infer_entity_intent(self.goal) == "club":
            return self._soccer_club_urls()
        if "basketball" in lowered and any(token in lowered for token in ("ncaa", "college", "division i")):
            discovered = self._discover_ncaa_team_stat_urls(
                "https://www.ncaa.com/stats/basketball-men/d1/current/team/145"
            )
            return discovered or [
                "https://www.ncaa.com/stats/basketball-men/d1/current/team/145",
                "https://www.ncaa.com/stats/basketball-men/d1/current/team/146",
                "https://www.ncaa.com/stats/basketball-men/d1/current/team/147",
                "https://www.ncaa.com/stats/basketball-men/d1/current/team/148",
                "https://www.ncaa.com/stats/basketball-men/d1/current/team/149",
                "https://www.ncaa.com/stats/basketball-men/d1/current/team/150",
                "https://www.ncaa.com/stats/basketball-men/d1/current/team/151",
                "https://www.ncaa.com/stats/basketball-men/d1/current/team/152",
                "https://www.ncaa.com/stats/basketball-men/d1/current/team/153",
                "https://www.ncaa.com/stats/basketball-men/d1/current/team/168",
                "https://www.ncaa.com/stats/basketball-men/d1/current/team/214",
                "https://www.ncaa.com/stats/basketball-men/d1/current/team/215",
                "https://www.ncaa.com/stats/basketball-men/d1/current/team/216",
                "https://www.ncaa.com/stats/basketball-men/d1/current/team/217",
                "https://www.ncaa.com/stats/basketball-men/d1/current/team/474",
                "https://www.ncaa.com/stats/basketball-men/d1/current/team/518",
                "https://www.ncaa.com/stats/basketball-men/d1/current/team/519",
                "https://www.ncaa.com/stats/basketball-men/d1/current/team/625",
                "https://www.ncaa.com/stats/basketball-men/d1/current/team/633",
                "https://www.ncaa.com/stats/basketball-men/d1/current/team/638",
                "https://www.ncaa.com/stats/basketball-men/d1/current/team/857",
                "https://www.ncaa.com/stats/basketball-men/d1/current/team/859",
                "https://www.ncaa.com/stats/basketball-men/d1/current/team/931",
                "https://www.ncaa.com/stats/basketball-men/d1/current/team/932",
                f"https://www.sports-reference.com/cbb/seasons/{year}-school-stats.html",
                f"https://www.sports-reference.com/cbb/seasons/{year}-advanced-school-stats.html",
                f"https://www.sports-reference.com/cbb/seasons/{year}-opponent-stats.html",
                "https://en.wikipedia.org/wiki/List_of_NCAA_Division_I_men%27s_basketball_programs",
            ]
        return []

    def _derived_companion_urls(self, url: str) -> list[str]:
        if "ncaa.com/stats/basketball-men/d1/current/team/" in url:
            if "ncaa" in self.goal.lower() and "basketball" in self.goal.lower() and "team statistics" in self.goal.lower():
                return []
            return self._discover_ncaa_team_stat_urls(url)
        if "teamrankings.com/nba/stat/" in url:
            return self._nba_team_stat_urls(self._infer_season_year(self.goal.lower()))
        if "espn.com/nba/stats/player" in url or "hoopshype.com/salaries/players" in url:
            return self._nba_player_stat_urls()
        if "transfermarkt.com/" in url and "/transfers/wettbewerb/" in url:
            return self._soccer_club_urls()
        match = re.search(r"/cbb/seasons/(\d{4})-[a-z-]+\.html", url)
        if not match:
            return []
        year = match.group(1)
        return [
            f"https://www.sports-reference.com/cbb/seasons/{year}-school-stats.html",
            f"https://www.sports-reference.com/cbb/seasons/{year}-advanced-school-stats.html",
            f"https://www.sports-reference.com/cbb/seasons/{year}-opponent-stats.html",
        ]

    def _extract_tables(self, url: str) -> list[pd.DataFrame]:
        if "ncaa.com/stats/basketball-men/d1/current/team/" in url:
            return self._extract_paginated_ncaa_tables(url)
        if "transfermarkt.com/" in url and "/transfers/wettbewerb/" in url:
            return self._extract_transfermarkt_club_summary(url)

        html = self._fetch_html(url)
        if not html:
            REGISTRY.record_extraction(url, records_extracted=0, success=False, reason=FailureReason.EMPTY_CONTENT)
            return []

        try:
            tables = pd.read_html(StringIO(html))
        except Exception as exc:
            LOGGER.warning("Failed to extract tables from %s: %s", url, exc)
            REGISTRY.record_extraction(url, records_extracted=0, success=False, reason=FailureReason.NO_TABLES)
            return []

        extracted: list[pd.DataFrame] = []
        for table in self._combine_related_tables(tables):
            frame = self._flatten_columns(table)
            frame = self._prefix_metric_columns(frame, url)
            if len(frame) < self.minimum_rows:
                continue
            frame = frame.copy()
            frame["_source_url"] = url
            extracted.append(frame)
        if extracted:
            REGISTRY.record_extraction(
                url,
                records_extracted=sum(len(frame) for frame in extracted),
                success=True,
                reason=FailureReason.SUCCESS,
            )
        else:
            REGISTRY.record_extraction(url, records_extracted=0, success=False, reason=FailureReason.NO_TABLES)
        return extracted

    def _extract_paginated_ncaa_tables(self, url: str) -> list[pd.DataFrame]:
        first_html = self._fetch_html(url)
        if not first_html:
            REGISTRY.record_extraction(url, records_extracted=0, success=False, reason=FailureReason.EMPTY_CONTENT)
            return []

        page_numbers = [
            int(match)
            for match in re.findall(r"/stats/basketball-men/d1/current/team/\d+/p(\d+)", first_html)
        ]
        max_page = max(page_numbers, default=1)
        tables: list[pd.DataFrame] = []
        for page in range(1, max_page + 1):
            page_url = url if page == 1 else f"{url}/p{page}"
            html = first_html if page == 1 else self._fetch_html(page_url)
            if not html:
                continue
            try:
                page_tables = pd.read_html(StringIO(html))
            except Exception:
                continue
            if not page_tables:
                continue
            frame = self._flatten_columns(page_tables[0])
            frame = self._prefix_ncaa_metric_columns(frame, url)
            frame = self._prefix_metric_columns(frame, page_url)
            if len(frame) == 0:
                continue
            frame = frame.copy()
            frame["_source_url"] = page_url
            tables.append(frame)

        if not tables:
            REGISTRY.record_extraction(url, records_extracted=0, success=False, reason=FailureReason.NO_TABLES)
            return []
        combined = pd.concat(tables, ignore_index=True)
        REGISTRY.record_extraction(url, records_extracted=len(combined), success=True, reason=FailureReason.SUCCESS)
        return [combined]

    def _fetch_html(self, url: str) -> str:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (compatible; PredictiveDatasetBuilder/1.0; "
                "+https://example.com/web-scraper)"
            )
        }
        outcome = fetch_url(
            url,
            headers=headers,
            timeout_seconds=self.timeout_seconds,
            verify=False,
        )
        if outcome.reason in {FailureReason.HTTP_403, FailureReason.HTTP_429, FailureReason.ANTI_BOT}:
            self._blacklist_domain(url)
            LOGGER.warning("Blocking detected while fetching %s; blacklisting %s", url, _root_domain(url))
        if not outcome.ok:
            LOGGER.warning("Failed to fetch %s: %s", url, outcome.detail or outcome.reason.value)
            return ""
        return outcome.text

    def _discover_ncaa_team_stat_urls(self, url: str) -> list[str]:
        outcome = fetch_url(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (compatible; PredictiveDatasetBuilder/1.0; "
                    "+https://example.com/web-scraper)"
                )
            },
            timeout_seconds=self.timeout_seconds,
            verify=False,
        )
        if outcome.reason in {FailureReason.HTTP_403, FailureReason.HTTP_429, FailureReason.ANTI_BOT}:
            self._blacklist_domain(url)
        if not outcome.ok:
            return []

        discovered = re.findall(
            r'<option value="(/stats/basketball-men/d1/current/team/\d+)">',
            outcome.text,
            flags=re.IGNORECASE,
        )
        filtered: list[str] = []
        for path in dict.fromkeys(discovered):
            stat_id = path.rsplit("/", 1)[-1]
            if stat_id not in NCAA_TEAM_STAT_ALLOWLIST:
                continue
            filtered.append(f"https://www.ncaa.com{path}")
        return filtered

    def _flatten_columns(self, frame: pd.DataFrame) -> pd.DataFrame:
        normalized = self._promote_header_row(frame.copy())
        if isinstance(normalized.columns, pd.MultiIndex):
            normalized.columns = [
                " ".join(str(part) for part in column if str(part) not in {"nan", ""}).strip()
                for column in normalized.columns
            ]
        normalized.columns = [self._normalize_column_name(str(column)) for column in normalized.columns]
        return normalized

    def _promote_header_row(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Promote the first row to headers when the parser left header cells in the body."""
        current_columns = [str(column).strip() for column in frame.columns]
        if not all(column.isdigit() or column.lower().startswith("unnamed") for column in current_columns):
            return frame
        if frame.empty:
            return frame

        header_candidates = [str(value).strip() for value in frame.iloc[0].tolist()]
        informative = [value for value in header_candidates if value and value.lower() not in {"nan", "none"}]
        if len(informative) < max(2, len(header_candidates) // 2):
            return frame
        if not any(
            any(token in value.lower() for token in {"name", "player", "team", "salary", "company", "bank", "state", "club"})
            for value in informative
        ):
            return frame

        promoted = frame.iloc[1:].reset_index(drop=True).copy()
        promoted.columns = header_candidates
        return promoted

    def _prepare_frame(self, frame: pd.DataFrame) -> pd.DataFrame | None:
        key_column = self._detect_entity_column(frame)
        if key_column is None:
            return None

        working = self._clean_table_rows(frame.copy())
        working = working[working[key_column].notna()]
        working[key_column] = working[key_column].astype(str).str.strip()
        working = working[~working[key_column].isin({"", key_column})]
        if len(working) < self.minimum_rows:
            return None

        working = working.drop_duplicates(subset=[key_column])
        if key_column != "entity_name":
            working["entity_name"] = working[key_column].map(self._normalize_entity_name)
        else:
            working["entity_name"] = working["entity_name"].astype(str).map(self._normalize_entity_name)
        working = self.entity_resolver.resolve_frame(working, entity_column="entity_name")

        renamed: dict[str, str] = {}
        for column in working.columns:
            if column == key_column:
                continue
            if column == "_source_url":
                renamed[column] = "source_url"
                continue
            if column in DROP_COLUMNS:
                continue
            renamed[column] = column

        selected_columns = ["entity_name"]
        if key_column != "entity_name":
            selected_columns.append(key_column)
        selected_columns.extend(col for col in renamed if col not in DROP_COLUMNS)
        trimmed = working.loc[:, ~working.columns.duplicated()][selected_columns].copy()
        if key_column not in {"entity_name", "raw_entity_name"}:
            trimmed = trimmed.rename(columns={key_column: "raw_entity_name"})
        trimmed = trimmed.rename(columns=renamed)
        trimmed = trimmed.loc[:, ~trimmed.columns.duplicated()]
        trimmed = self._convert_numeric_columns(trimmed)
        trimmed = self._drop_low_value_columns(trimmed)
        source_ref = self._frame_source_reference(working)
        trimmed.attrs["source_ref"] = source_ref
        trimmed.attrs["entity_column"] = key_column
        trimmed.attrs["provenance_map"] = {
            column: source_ref
            for column in trimmed.columns
            if column not in {"entity_name", "raw_entity_name"}
        }
        return trimmed

    def _merge_frames(self, left: pd.DataFrame, right: pd.DataFrame, *, how: str = "outer") -> pd.DataFrame:
        left = left.copy()
        right = right.copy()

        for identifier in ("raw_entity_name", "source_url", "name"):
            if identifier in left.columns and identifier in right.columns:
                right = right.drop(columns=[identifier])

        overlapping = [
            column for column in right.columns if column in left.columns and column != "entity_name"
        ]
        if overlapping:
            renamed_overlap = {
                column: f"{column}__rhs"
                for column in overlapping
            }
            right = right.rename(columns=renamed_overlap)
        merged = left.merge(right, on="entity_name", how=how)
        if "raw_entity_name_x" in merged.columns and "raw_entity_name_y" in merged.columns:
            merged["raw_entity_name"] = merged["raw_entity_name_x"].fillna(merged["raw_entity_name_y"])
            merged = merged.drop(columns=["raw_entity_name_x", "raw_entity_name_y"])
        for column in overlapping:
            rhs_column = f"{column}__rhs"
            if rhs_column not in merged.columns:
                continue
            if column in merged.columns:
                merged[column] = merged[column].combine_first(merged[rhs_column])
                merged = merged.drop(columns=[rhs_column])
            else:
                merged = merged.rename(columns={rhs_column: column})
        return merged

    def _finalize_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        finalized = frame.copy()
        if "raw_entity_name" in finalized.columns:
            finalized = finalized.rename(columns={"raw_entity_name": "name"})
        finalized = finalized.drop_duplicates(subset=["entity_name"])
        finalized = finalized.rename(columns=self._final_column_label_map(finalized.columns))

        ordered = self._order_final_columns(finalized)
        finalized = finalized[[column for column in ordered if column in finalized.columns]]
        finalized = finalized.loc[:, ~finalized.columns.duplicated()]

        null_threshold = max(1, min(int(len(finalized) * 0.10), 25))
        keep_columns = [
            column
            for column in finalized.columns
            if (
                column in {"name", "entity_name"}
                or (
                    finalized[column].notna().sum() >= null_threshold
                    and not self._is_low_signal_column(column, finalized[column])
                )
            )
        ]
        finalized = finalized[keep_columns]
        finalized = self._trim_excess_columns(finalized)
        return finalized.reset_index(drop=True)

    def _final_column_label_map(self, columns: pd.Index) -> dict[str, str]:
        rename_map: dict[str, str] = {}
        seen: set[str] = set()
        for column in columns:
            cleaned = self._normalize_final_column_label(str(column))
            candidate = cleaned
            suffix = 2
            while candidate in seen:
                candidate = f"{cleaned}_{suffix}"
                suffix += 1
            seen.add(candidate)
            rename_map[str(column)] = candidate
        return rename_map

    def _normalize_final_column_label(self, column: str) -> str:
        if column == "source_url":
            return "source"
        parts = [part for part in str(column).split("_") if part]
        if not parts:
            return column
        cleaned_parts: list[str] = []
        for part in parts:
            if cleaned_parts and part == cleaned_parts[-1]:
                continue
            cleaned_parts.append(part)
        if len(cleaned_parts) >= 2 and cleaned_parts[-1] == "value":
            cleaned_parts = cleaned_parts[:-1]
        return "_".join(cleaned_parts) or column

    def _order_final_columns(self, frame: pd.DataFrame) -> list[str]:
        preferred = ["name", "entity_name"]
        columns = [column for column in frame.columns if column not in preferred]
        goal_tokens = set(re.findall(r"[a-z0-9]+", self.goal.lower()))

        def priority(column: str) -> tuple[int, int, str]:
            lowered = column.lower()
            if lowered in {"source", "source_url"}:
                return (4, 0, lowered)
            if any(token in lowered for token in ("target", "label", "outcome", "salary", "valuation", "revenue", "income", "gdp", "growth")):
                return (1, 0, lowered)
            if any(token in lowered for token in goal_tokens):
                return (2, 0, lowered)
            if pd.api.types.is_numeric_dtype(frame[column]):
                return (2, 1, lowered)
            return (3, 0, lowered)

        return preferred + sorted(columns, key=priority)

    def _trim_excess_columns(self, frame: pd.DataFrame) -> pd.DataFrame:
        max_columns = max(self.minimum_columns, min(36, max(12, len(frame) * 2)))
        if len(frame.columns) <= max_columns:
            return frame

        protected = {"name", "entity_name"}
        ranked_columns = []
        for column in frame.columns:
            if column in protected:
                continue
            ranked_columns.append((self._final_column_score(frame, column), column))

        keep = list(protected)
        keep.extend(column for _, column in sorted(ranked_columns, reverse=True))
        keep = keep[:max_columns]
        keep_set = set(keep)
        return frame[[column for column in frame.columns if column in keep_set]]

    def _final_column_score(
        self,
        frame: pd.DataFrame,
        column: str,
    ) -> tuple[float, float, int, int, str]:
        series = frame[column]
        coverage = float(series.notna().mean())
        unique = int(series.nunique(dropna=True))
        numeric_bonus = 1 if pd.api.types.is_numeric_dtype(series) else 0
        target_bonus = 1 if any(
            token in column.lower()
            for token in ("target", "label", "outcome", "salary", "valuation", "revenue", "income", "gdp", "growth")
        ) else 0
        return (target_bonus, coverage, numeric_bonus, unique, column)

    def _enforce_fill_rate(self, frame: pd.DataFrame) -> None:
        target_column = self._resolve_quality_column(frame, [self.target_field] if self.target_field else [])
        feature_columns = self._resolve_quality_columns(frame, self.core_feature_fields)
        if not feature_columns:
            feature_columns = self._infer_core_feature_columns(frame)

        if target_column is not None:
            target_missing_rate = self._missing_rate(frame[target_column])
            if target_missing_rate > 0.10:
                raise DataQualityError(
                    f"Dataset rejected due to low fill rate: target '{target_column}' missing in {target_missing_rate:.1%} of rows"
                )

        low_fill_features = [
            column
            for column in feature_columns
            if self._missing_rate(frame[column]) > 0.40
        ]
        if low_fill_features:
            raise DataQualityError(
                "Dataset rejected due to low fill rate: core features below threshold "
                f"({', '.join(sorted(low_fill_features))})"
            )

    def _resolve_quality_columns(self, frame: pd.DataFrame, field_names: list[str]) -> list[str]:
        resolved: list[str] = []
        seen: set[str] = set()
        for field_name in field_names:
            column = self._resolve_quality_column(frame, [field_name])
            if column is None or column in seen:
                continue
            seen.add(column)
            resolved.append(column)
        return resolved

    def _resolve_quality_column(self, frame: pd.DataFrame, field_names: list[str]) -> str | None:
        if not field_names:
            return None
        columns = list(map(str, frame.columns))
        normalized_columns = {column: self._normalize_column_name(column) for column in columns}
        for field_name in field_names:
            if not field_name:
                continue
            if field_name in frame.columns:
                return field_name
            normalized_field = self._normalize_column_name(field_name)
            for column, normalized_column in normalized_columns.items():
                if normalized_column == normalized_field:
                    return column
            for column, normalized_column in normalized_columns.items():
                if normalized_column.startswith(normalized_field) or normalized_field in normalized_column:
                    return column
        return None

    def _infer_core_feature_columns(self, frame: pd.DataFrame) -> list[str]:
        target_column = self._resolve_quality_column(frame, [self.target_field] if self.target_field else [])
        candidate_columns = [
            column
            for column in frame.columns
            if column not in IDENTITY_COLUMNS and column != target_column
        ]
        scored = sorted(
            candidate_columns,
            key=lambda column: self._final_column_score(frame, str(column)),
            reverse=True,
        )
        return [str(column) for column in scored[: min(4, len(scored))]]

    def _missing_rate(self, series: pd.Series) -> float:
        if len(series) == 0:
            return 1.0
        missing = series.isna()
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            normalized = series.fillna("").astype(str).str.strip().str.casefold()
            missing = missing | normalized.isin({"", "unknown", "n/a", "na", "none", "null"})
        return float(missing.mean())

    def _drop_low_value_columns(self, frame: pd.DataFrame) -> pd.DataFrame:
        keep = ["entity_name"]
        if "raw_entity_name" in frame.columns:
            keep.append("raw_entity_name")
        if "source_url" in frame.columns:
            keep.append("source_url")

        for column in frame.columns:
            if column in keep:
                continue
            series = frame[column]
            if series.notna().sum() < self.minimum_rows // 2:
                continue
            if series.nunique(dropna=True) <= 1:
                continue
            keep.append(column)
        return frame[keep]

    def _clean_table_rows(self, frame: pd.DataFrame) -> pd.DataFrame:
        cleaned = frame.copy()
        if cleaned.empty:
            return cleaned

        repeated_header_mask = pd.Series(False, index=cleaned.index)
        normalized_columns = [str(column).strip().lower() for column in cleaned.columns]
        for index, row in cleaned.iterrows():
            normalized_values = [str(value).strip().lower() for value in row.tolist()]
            if not normalized_values:
                continue
            if normalized_values == normalized_columns[: len(normalized_values)]:
                repeated_header_mask.loc[index] = True
                continue
            informative_pairs = sum(
                1
                for column, value in zip(normalized_columns, normalized_values)
                if column and value and column == value
            )
            if informative_pairs >= max(2, len(normalized_columns) // 2):
                repeated_header_mask.loc[index] = True

        if repeated_header_mask.any():
            cleaned = cleaned.loc[~repeated_header_mask].reset_index(drop=True)
        cleaned = cleaned.loc[:, ~cleaned.columns.duplicated()]
        return cleaned

    def _numeric_feature_count(self, frame: pd.DataFrame) -> int:
        count = 0
        for column in frame.columns:
            if column in IDENTITY_COLUMNS:
                continue
            if pd.api.types.is_numeric_dtype(frame[column]):
                count += 1
        return count

    def _best_single_frame(self, frames: list[pd.DataFrame]) -> pd.DataFrame | None:
        if not frames:
            return None
        return max(
            frames,
            key=lambda frame: (
                self._row_count_score(len(frame)),
                self._frame_goal_alignment_score(frame),
                self._numeric_feature_count(frame),
                len(frame.columns),
                len(frame),
            ),
        )

    def _row_count_score(self, row_count: int) -> float:
        cardinality = infer_goal_cardinality(self.goal)
        if cardinality is None or cardinality.count <= 0:
            return 0.0
        diff = abs(row_count - cardinality.count)
        if cardinality.exact:
            return -float(diff)
        return -(diff / max(cardinality.count, 1))

    def _frame_row_count_is_reasonable(self, frame: pd.DataFrame) -> bool:
        cardinality = infer_goal_cardinality(self.goal)
        if cardinality is None or cardinality.count <= 0:
            return True

        row_count = len(frame)
        if cardinality.exact:
            lower_bound = min(self.minimum_rows, max(3, int(cardinality.count * 0.5)))
            upper_bound = max(cardinality.count + 2, int(cardinality.count * 1.5))
            return lower_bound <= row_count <= upper_bound

        lower_bound = min(self.minimum_rows, max(3, int(cardinality.count * 0.4)))
        upper_bound = max(cardinality.count + 5, int(cardinality.count * 2.0))
        return lower_bound <= row_count <= upper_bound

    def _frame_goal_alignment_score(self, frame: pd.DataFrame) -> int:
        goal_tokens = set(re.findall(r"[a-z0-9]+", self.goal.lower()))
        entity_intent = infer_entity_intent(self.goal)
        entity_column = str(getattr(frame, "attrs", {}).get("entity_column", "")).lower()
        source_ref = str(getattr(frame, "attrs", {}).get("source_ref", "")).lower()
        score = 0

        if entity_intent == "team" or "team" in goal_tokens:
            if entity_column == "team":
                score += 8
            if "team" in source_ref or "/teams/" in source_ref:
                score += 4
            if "season" in source_ref:
                score -= 5

        if entity_intent == "club":
            if entity_column in {"club", "team", "name"}:
                score += 8
            if "transfermarkt" in source_ref or "club" in source_ref:
                score += 4

        if entity_intent == "player":
            if entity_column in {"player", "name"}:
                score += 8
            if "player" in source_ref or "/players" in source_ref:
                score += 4
            if "salary" in source_ref:
                score += 4
            if "teamrankings.com/nba/stat/" in source_ref:
                score -= 12

        if "company" in goal_tokens and entity_column == "company":
            score += 8
        if "bank" in goal_tokens and entity_column == "bank":
            score += 8
        if "school" in goal_tokens and entity_column == "school":
            score += 8

        sample_names = [
            str(value).strip().lower()
            for value in frame.get("raw_entity_name", pd.Series(dtype="object")).dropna().astype(str).head(10)
            if str(value).strip()
        ]
        if sample_names:
            numeric_like = sum(1 for value in sample_names if re.fullmatch(r"(19|20|21)\d{2}(?:[–-]\d{2,4})?", value))
            if numeric_like >= max(1, len(sample_names) // 2):
                score -= 6

        return score

    def _frame_has_target_signal(self, frame: pd.DataFrame) -> bool:
        lowered_columns = {str(column).lower() for column in frame.columns}
        lowered_source = str(getattr(frame, "attrs", {}).get("source_ref", "")).lower()
        target_markers = {
            token
            for token in re.findall(r"[a-z0-9]+", self.goal.lower())
            if token in {"salary", "valuation", "revenue", "income", "spend", "transfer", "outcome", "label"}
        }
        if "salary" in target_markers and any("salary" in column for column in lowered_columns):
            return True
        if "valuation" in target_markers and any("valuation" in column for column in lowered_columns):
            return True
        if {"transfer", "spend"} & target_markers and any(
            token in column for column in lowered_columns for token in {"fee", "spend", "transfer"}
        ):
            return True
        return any(marker in lowered_source for marker in target_markers)

    def _frame_has_merge_coverage(self, left: pd.DataFrame, right: pd.DataFrame) -> bool:
        left_keys = set(left["entity_name"].dropna().astype(str))
        right_keys = set(right["entity_name"].dropna().astype(str))
        overlap = len(left_keys & right_keys)
        smaller_frame_size = min(len(left_keys), len(right_keys))
        if smaller_frame_size <= 5:
            minimum_overlap = 1
        else:
            minimum_overlap = max(3, smaller_frame_size // 30)
        return overlap >= minimum_overlap

    def _frames_are_row_compatible(self, left: pd.DataFrame, right: pd.DataFrame) -> bool:
        left_keys = set(left["entity_name"].dropna().astype(str))
        right_keys = set(right["entity_name"].dropna().astype(str))
        if left_keys & right_keys:
            return False
        identity_columns = IDENTITY_COLUMNS
        left_columns = {column for column in left.columns if column not in identity_columns}
        right_columns = {column for column in right.columns if column not in identity_columns}
        shared = left_columns & right_columns
        return bool(shared) and len(shared) >= min(len(left_columns), len(right_columns)) // 2

    def _concat_frames(self, left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
        columns = list(dict.fromkeys([*left.columns, *right.columns]))
        combined = pd.concat(
            [
                left.reindex(columns=columns),
                right.reindex(columns=columns),
            ],
            ignore_index=True,
        )
        return combined.drop_duplicates(subset=["entity_name"])

    def _frame_source_reference(self, frame: pd.DataFrame) -> str:
        if "_source_url" in frame.columns:
            source_values = [str(value).strip() for value in frame["_source_url"].dropna().astype(str).unique() if str(value).strip()]
            if source_values:
                if len(source_values) == 1:
                    return source_values[0]
                return urlparse(source_values[0]).netloc or source_values[0]
        return "unknown_source"

    def _finalize_provenance(self, frame: pd.DataFrame, provenance_map: dict[str, str]) -> dict[str, str]:
        return {
            column: provenance_map.get(column, "derived")
            for column in frame.columns
        }

    def _detect_entity_column(self, frame: pd.DataFrame) -> str | None:
        columns = frame.columns
        entity_intent = infer_entity_intent(self.goal)
        candidate_order = list(ENTITY_COLUMN_CANDIDATES)
        if entity_intent == "player":
            candidate_order = [
                "player", "name", "player_name", "entity_name", "team", "company", "organization", "title"
            ] + [candidate for candidate in ENTITY_COLUMN_CANDIDATES if candidate not in {"player", "name", "team", "company", "organization", "title"}]
        elif entity_intent == "company":
            candidate_order = [
                "company", "organization", "name", "entity_name", "title"
            ] + [candidate for candidate in ENTITY_COLUMN_CANDIDATES if candidate not in {"company", "organization", "name", "entity_name", "title"}]
        elif entity_intent == "bank":
            candidate_order = [
                "bank", "name", "organization", "company", "entity_name", "title"
            ] + [candidate for candidate in ENTITY_COLUMN_CANDIDATES if candidate not in {"bank", "name", "organization", "company", "entity_name", "title"}]
        elif entity_intent in {"team", "club"}:
            candidate_order = [
                "team", "school", "name", "entity_name", "title"
            ] + [candidate for candidate in ENTITY_COLUMN_CANDIDATES if candidate not in {"team", "school", "name", "entity_name", "title"}]

        for candidate in candidate_order:
            if candidate in columns:
                return candidate
        best_column: str | None = None
        best_score = -1
        for column in columns:
            parts = [
                part
                for part in str(column).split("_")
                if part not in {"unnamed", "level"} and not part.isdigit()
            ]
            if not parts:
                continue
            score = 0
            if "bank" in parts:
                score = 7
            elif "company" in parts:
                score = 6
            elif "school" in parts:
                score = 5
            elif "team" in parts:
                score = 4
            elif "name" in parts:
                score = 3
            elif any(token in parts for token in {"player", "company", "organization", "title"}):
                score = 2

            if any(token in parts for token in {"rank", "rk", "no", "number", "position"}):
                score -= 5

            sample_values = [
                str(value).strip()
                for value in frame[column].dropna().astype(str).head(20)
                if str(value).strip()
            ]
            if sample_values:
                numeric_like = sum(1 for value in sample_values if re.fullmatch(r"[\d,.]+", value))
                if numeric_like >= max(1, len(sample_values) * 0.7):
                    score -= 6
                alpha_like = sum(1 for value in sample_values if re.search(r"[A-Za-z]", value))
                if alpha_like >= max(1, len(sample_values) * 0.7):
                    score += 2

            if score > best_score:
                best_score = score
                best_column = str(column)
        return best_column

    def _convert_numeric_columns(self, frame: pd.DataFrame) -> pd.DataFrame:
        converted = frame.copy()
        for column in converted.columns:
            if column in {"entity_name", "raw_entity_name", "source_url"}:
                continue
            series = converted[column].map(self._normalize_numeric_like)
            numeric = pd.to_numeric(series, errors="coerce")
            if numeric.notna().sum() >= max(len(series) * 0.7, self.minimum_rows // 2):
                converted[column] = numeric
        return converted

    def _prefix_ncaa_metric_columns(self, frame: pd.DataFrame, url: str) -> pd.DataFrame:
        stat_id = url.rstrip("/").split("/")[-1]
        prefix = NCAA_TEAM_STAT_NAMES.get(stat_id)
        if not prefix:
            return frame

        rename_map: dict[str, str] = {}
        for column in frame.columns:
            if column in {"rank", "team"}:
                continue
            rename_map[column] = f"{prefix}_{column}"
        return frame.rename(columns=rename_map)

    def _prefix_metric_columns(self, frame: pd.DataFrame, url: str) -> pd.DataFrame:
        """Namespace generic stat-page columns so multiple sources can merge cleanly."""
        prefix = self._metric_prefix_from_url(url)
        if not prefix:
            return frame

        identity_columns = {
            "rank",
            "team",
            "name",
            "player",
            "school",
            "club",
            "company",
            "bank",
            "state",
            "organization",
            "entity_name",
            "raw_entity_name",
        }
        rename_map: dict[str, str] = {}
        for column in frame.columns:
            if column in identity_columns or column.startswith(f"{prefix}_"):
                continue
            rename_map[column] = f"{prefix}_{column}"
        return frame.rename(columns=rename_map)

    def _metric_prefix_from_url(self, url: str) -> str | None:
        if "teamrankings.com/nba/stat/" in url:
            slug = urlparse(url).path.rstrip("/").rsplit("/", 1)[-1]
            return self._normalize_column_name(slug)
        if "hoopshype.com/salaries/players" in url:
            return "salary"
        if "espn.com/nba/salaries" in url:
            return "salary"
        return None

    def _nba_team_stat_urls(self, year: int) -> list[str]:
        season_label = f"{year - 1}-{str(year)[-2:]}"
        urls = [f"https://www.teamrankings.com/nba/stat/{metric}" for metric in TEAMRANKINGS_NBA_METRICS]
        urls.extend(
            [
                f"https://www.nba.com/stats/teams/traditional?Season={season_label}",
                f"https://en.wikipedia.org/wiki/{year - 1}%E2%80%93{str(year)[-2:]}_NBA_season",
                "https://en.wikipedia.org/wiki/National_Basketball_Association",
            ]
        )
        return urls

    def _nba_player_stat_urls(self) -> list[str]:
        return [
            "https://www.espn.com/nba/stats/player",
            "https://www.espn.com/nba/salaries",
            "https://hoopshype.com/salaries/players/",
            "https://www.nba.com/players",
        ]

    def _startup_company_urls(self) -> list[str]:
        return [
            "https://en.wikipedia.org/wiki/List_of_unicorn_startup_companies",
        ]

    def _soccer_club_urls(self) -> list[str]:
        season = self._infer_season_year(self.goal.lower()) - 1
        return [
            f"https://www.transfermarkt.com/premier-league/transfers/wettbewerb/GB1/plus/?saison_id={season}&s_w=s&leihe=1&intern=0",
            f"https://www.transfermarkt.com/laliga/transfers/wettbewerb/ES1/plus/?saison_id={season}&s_w=s&leihe=1&intern=0",
            f"https://www.transfermarkt.com/serie-a/transfers/wettbewerb/IT1/plus/?saison_id={season}&s_w=s&leihe=1&intern=0",
            f"https://www.transfermarkt.com/ligue-1/transfers/wettbewerb/FR1/plus/?saison_id={season}&s_w=s&leihe=1&intern=0",
            f"https://www.transfermarkt.com/bundesliga/transfers/wettbewerb/L1/plus/?saison_id={season}&s_w=s&leihe=1&intern=0",
        ]

    def _combine_related_tables(self, tables: list[pd.DataFrame]) -> list[pd.DataFrame]:
        """Combine split-name/stat tables that appear as side-by-side HTML tables."""
        combined: list[pd.DataFrame] = []
        index = 0
        while index < len(tables):
            current = tables[index]
            if index + 1 < len(tables):
                merged = self._maybe_merge_parallel_tables(current, tables[index + 1])
                if merged is not None:
                    combined.append(merged)
                    index += 2
                    continue
            combined.append(current)
            index += 1
        return combined

    def _maybe_merge_parallel_tables(self, left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame | None:
        if len(left) == 0 or len(right) == 0 or len(left) != len(right):
            return None
        left_columns = {self._normalize_column_name(str(column)) for column in left.columns}
        right_columns = {self._normalize_column_name(str(column)) for column in right.columns}
        if not left_columns.intersection({"name", "player", "team", "company"}):
            return None
        if right_columns.intersection({"name", "player", "team", "company"}):
            return None
        if sum(_is_numeric_like_value(value) for value in right.head(5).to_numpy().ravel()) < 5:
            return None
        merged = pd.concat([left.reset_index(drop=True), right.reset_index(drop=True)], axis=1)
        return merged.loc[:, ~merged.columns.duplicated()]

    def _extract_transfermarkt_club_summary(self, url: str) -> list[pd.DataFrame]:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (compatible; PredictiveDatasetBuilder/1.0; "
                "+https://example.com/web-scraper)"
            )
        }
        outcome = fetch_url(
            url,
            headers=headers,
            timeout_seconds=max(self.timeout_seconds, 20.0),
            verify=False,
        )
        if outcome.reason in {FailureReason.HTTP_403, FailureReason.HTTP_429, FailureReason.ANTI_BOT}:
            self._blacklist_domain(url)
        if not outcome.ok or not outcome.text:
            REGISTRY.record_extraction(url, records_extracted=0, success=False, reason=FailureReason.EMPTY_CONTENT)
            return []
        html = outcome.text
        try:
            tables = pd.read_html(StringIO(html))
        except Exception as exc:
            LOGGER.warning("Failed to extract Transfermarkt tables from %s: %s", url, exc)
            REGISTRY.record_extraction(url, records_extracted=0, success=False, reason=FailureReason.NO_TABLES)
            return []

        club_names = list(
            dict.fromkeys(
                re.findall(r'href="#to-\d+"><img [^>]*title="([^"]+)"', html, flags=re.IGNORECASE)
            )
        )
        detail_tables = tables[1:]
        pair_count = min(len(club_names), len(detail_tables) // 2)
        rows: list[dict[str, object]] = []
        for index in range(pair_count):
            incoming = self._flatten_columns(detail_tables[index * 2])
            outgoing = self._flatten_columns(detail_tables[index * 2 + 1])
            club_name = club_names[index]
            rows.append(
                {
                    "club": club_name,
                    "incoming_player_count": len(incoming),
                    "outgoing_player_count": len(outgoing),
                    "incoming_transfer_spend": self._sum_money_column(incoming, "fee"),
                    "outgoing_transfer_income": self._sum_money_column(outgoing, "fee"),
                    "incoming_market_value_total": self._sum_money_column(incoming, "market_value"),
                    "outgoing_market_value_total": self._sum_money_column(outgoing, "market_value"),
                    "_source_url": url,
                }
            )
        if not rows:
            REGISTRY.record_extraction(url, records_extracted=0, success=False, reason=FailureReason.NO_TABLES)
            return []
        frame = pd.DataFrame(rows)
        REGISTRY.record_extraction(url, records_extracted=len(frame), success=True, reason=FailureReason.SUCCESS)
        return [frame]

    def _sum_money_column(self, frame: pd.DataFrame, column: str) -> float:
        if column not in frame.columns:
            return 0.0
        total = 0.0
        for value in frame[column].tolist():
            total += _parse_money_value(value)
        return round(total, 4)

    def _normalize_column_name(self, value: str) -> str:
        normalized = value.strip().lower()
        normalized = normalized.replace("%", "_pct")
        normalized = normalized.replace("/", "_per_")
        normalized = normalized.replace("tm.", "team")
        normalized = normalized.replace("opp.", "opponent")
        normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
        normalized = re.sub(r"_+", "_", normalized).strip("_")
        return normalized or "column"

    def _normalize_entity_name(self, value: str) -> str:
        return self.entity_resolver.canonical_key(value)

    def _is_low_signal_column(self, column: str, series: pd.Series) -> bool:
        normalized = str(column).strip().lower()
        if normalized.isdigit() or normalized in {"column", "unnamed", "unnamed_0", "unnamed_1"}:
            return True
        if normalized in {"rk", "rank"} or normalized.endswith("_rk") or "_rank" in normalized:
            return True
        if re.fullmatch(r"\d+", normalized):
            return True
        if normalized.startswith("unnamed"):
            return True
        if pd.api.types.is_numeric_dtype(series):
            values = series.dropna().tolist()
            if len(values) >= 5:
                try:
                    numeric_values = [float(value) for value in values[: min(len(values), 25)]]
                except (TypeError, ValueError):
                    numeric_values = []
                if numeric_values:
                    expected = list(range(1, len(numeric_values) + 1))
                    if numeric_values == expected:
                        return True
        return False

    def _normalize_numeric_like(self, value: Any) -> str:
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none"}:
            return text
        text = text.replace(",", "")
        text = text.replace("$", "")
        text = text.replace("%", "")
        text = re.sub(r"^[^\d\-\.]+", "", text)
        multiplier = Decimal("1")
        lowered = text.lower()
        if lowered.endswith("bn"):
            multiplier = Decimal("1000000000")
            text = text[:-2]
        elif lowered.endswith("b"):
            multiplier = Decimal("1000000000")
            text = text[:-1]
        elif lowered.endswith("m"):
            multiplier = Decimal("1000000")
            text = text[:-1]
        elif lowered.endswith("k"):
            multiplier = Decimal("1000")
            text = text[:-1]
        try:
            return str(float(Decimal(text) * multiplier))
        except (InvalidOperation, ValueError):
            return text

    def _infer_season_year(self, lowered_goal: str) -> int:
        years = [int(match) for match in re.findall(r"\b(19\d{2}|20\d{2}|21\d{2})\b", lowered_goal)]
        if years:
            return years[-1]
        if "ncaa" in lowered_goal or ("college" in lowered_goal and "basketball" in lowered_goal):
            return 2025
        return 2026

    def _response_indicates_blocking(self, response: requests.Response) -> bool:
        if response.status_code in {403, 429}:
            return True
        lowered = response.text.lower()
        return any(token in lowered for token in ("captcha", "verify you are human", "cloudflare", "access denied"))

    def _blacklist_domain(self, url: str) -> None:
        domain = _root_domain(url)
        if domain:
            self.domain_blacklist.add(domain)


@dataclass(slots=True)
class PredictiveBuildResult:
    """Final predictive dataset artifact with provenance."""

    dataframe: pd.DataFrame
    provenance_map: dict[str, str]

    @property
    def records(self) -> list[dict[str, Any]]:
        return self.dataframe.to_dict(orient="records")


def _root_domain(url: str) -> str:
    hostname = (urlparse(url).hostname or "").lower().strip(".")
    if not hostname:
        return ""
    parts = hostname.split(".")
    if len(parts) <= 2:
        return hostname
    if parts[-2] in {"co", "com", "org", "gov", "ac"} and len(parts) >= 3:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def _is_numeric_like_value(value: Any) -> bool:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return False
    text = text.replace(",", "").replace("$", "").replace("%", "")
    text = text.lstrip("P")
    return bool(re.fullmatch(r"-?\d+(?:\.\d+)?", text))


def _parse_money_value(value: Any) -> float:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "-", "?", "free transfer", "end of loan"}:
        return 0.0
    text = text.replace(",", "")
    text = text.replace("€", "")
    text = text.replace("$", "")
    text = text.replace("£", "")
    text = re.sub(r"loan fee:?", "", text, flags=re.IGNORECASE)
    text = text.strip()
    multiplier = 1.0
    lowered = text.lower()
    if lowered.endswith("bn"):
        multiplier = 1_000_000_000.0
        text = text[:-2]
    elif lowered.endswith("m"):
        multiplier = 1_000_000.0
        text = text[:-1]
    elif lowered.endswith("k"):
        multiplier = 1_000.0
        text = text[:-1]
    try:
        return float(text) * multiplier
    except ValueError:
        return 0.0
