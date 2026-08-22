"""Build reader-facing Part 1 handoff documents from verified analysis outputs.

The module deliberately contains no analysis constants masquerading as findings.
Every title and number in the generated documents is derived from the frozen
snapshot metadata, the data-quality report, or the temporal result tables.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SEVERITY_COLOURS = {
    "Felony": "#0072B2",
    "Misdemeanor": "#E69F00",
    "Violation": "#CC79A7",
    "Other or missing": "#9AA3AA",
}

REQUIRED_TEMPORAL_FILES = {
    "daily": "daily_arrests.csv",
    "monthly": "monthly_arrests.csv",
    "weekday": "weekday_arrests.csv",
}

REQUIRED_FIGURES = {
    "missingness": "missingness_overview.png",
    "daily": "daily_arrests_rolling.png",
    "monthly": "monthly_average_daily_arrests.png",
    "weekday": "weekday_average_arrests.png",
}

DELIVERABLE_NAMES = (
    "visual_style_guide.md",
    "chart_contracts.md",
    "slide_plan.md",
    "slide_notes.md",
    "team_handoff.md",
    "ai_usage_note.md",
)


class DeliverableInputError(ValueError):
    """Raised when an upstream artifact cannot support truthful deliverables."""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise DeliverableInputError(f"Required JSON file does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DeliverableInputError(f"Could not read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DeliverableInputError(f"Expected a JSON object in {path}")
    return value


def _read_csv(path: Path, required_columns: Sequence[str]) -> pd.DataFrame:
    if not path.is_file():
        raise DeliverableInputError(f"Required CSV file does not exist: {path}")
    try:
        frame = pd.read_csv(path)
    except Exception as exc:  # pandas exposes several parser/IO exception types
        raise DeliverableInputError(f"Could not read {path}: {exc}") from exc
    missing = sorted(set(required_columns).difference(frame.columns))
    if missing:
        raise DeliverableInputError(f"{path.name} is missing required columns: {missing}")
    if frame.empty:
        raise DeliverableInputError(f"Required CSV is empty: {path}")
    return frame


def _first(mapping: Mapping[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    for key in keys:
        if key in mapping and mapping[key] not in (None, ""):
            return mapping[key]
    return default


def _find_first(value: Any, keys: Sequence[str]) -> Any:
    """Find the first non-empty value for a key, preferring shallower objects."""

    wanted = {key.casefold() for key in keys}
    frontier = [value]
    while frontier:
        next_frontier: list[Any] = []
        for item in frontier:
            if isinstance(item, Mapping):
                for key, child in item.items():
                    if str(key).casefold() in wanted and child not in (None, ""):
                        return child
                next_frontier.extend(item.values())
            elif isinstance(item, list):
                next_frontier.extend(item)
        frontier = next_frontier
    return None


def _as_int(value: Any, label: str) -> int:
    try:
        result = int(float(value))
    except (TypeError, ValueError) as exc:
        raise DeliverableInputError(f"Missing or invalid integer for {label}: {value!r}") from exc
    if result < 0:
        raise DeliverableInputError(f"Negative value is invalid for {label}: {result}")
    return result


def _as_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise DeliverableInputError(f"Missing or invalid number for {label}: {value!r}") from exc
    if not math.isfinite(result):
        raise DeliverableInputError(f"Non-finite value is invalid for {label}: {value!r}")
    return result


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    return str(value).strip().casefold() in {"1", "true", "yes", "y"}


def _format_date(value: Any) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise DeliverableInputError(f"Invalid date value in an upstream artifact: {value!r}")
    return parsed.strftime("%d %b %Y")


def _format_count(value: int) -> str:
    return f"{value:,}"


def _format_rate(value: float) -> str:
    return f"{value:,.1f}"


def _parse_log_count(text: str, labels: Sequence[str]) -> int | None:
    for label in labels:
        match = re.search(
            rf"(?im)^\s*{re.escape(label)}\s*:\s*([0-9][0-9,]*)\s*$",
            text,
        )
        if match:
            return int(match.group(1).replace(",", ""))
    return None


def _extract_missingness(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Normalise common nested missingness-report shapes without guessing values."""

    missing_root = report.get("missingness", {})
    rows: dict[str, dict[str, Any]] = {}

    def visit(value: Any, implied_name: str | None = None) -> None:
        if isinstance(value, list):
            for item in value:
                visit(item)
            return
        if not isinstance(value, Mapping):
            return

        name = _first(value, ("column_name", "column", "field", "variable"), implied_name)
        pct = _first(
            value,
            (
                "effective_missing_percentage",
                "missing_percentage",
                "null_percentage",
                "missing_pct",
                "null_pct",
            ),
        )
        count = _first(
            value,
            ("effective_missing_count", "missing_count", "null_count"),
        )
        if name is not None and pct is not None:
            try:
                pct_value = float(pct)
            except (TypeError, ValueError):
                pct_value = math.nan
            if math.isfinite(pct_value):
                rows[str(name)] = {
                    "column": str(name),
                    "missing_percentage": pct_value,
                    "missing_count": count,
                }

        for key, child in value.items():
            child_name = str(key) if isinstance(child, Mapping) else None
            visit(child, child_name)

    visit(missing_root)
    return sorted(rows.values(), key=lambda row: row["missing_percentage"], reverse=True)


def _extract_schema_columns(report: Mapping[str, Any]) -> list[str]:
    schema = report.get("schema_validation", {})
    candidates = (
        _find_first(schema, ("available_columns", "actual_columns", "column_names", "columns")),
        _find_first(
            report.get("dataset", {}),
            ("standardized_column_names", "column_names", "columns"),
        ),
    )
    for candidate in candidates:
        if isinstance(candidate, list) and all(isinstance(item, str) for item in candidate):
            return [item.upper() for item in candidate]
        if isinstance(candidate, Mapping):
            return [str(item).upper() for item in candidate]
    return []


def _quality_count(
    report: Mapping[str, Any], section: str, aliases: Sequence[str]
) -> int | None:
    value = _find_first(report.get(section, {}), aliases)
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    try:
        result = int(float(value))
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None


def _normalise_inputs(
    metadata: Mapping[str, Any],
    quality: Mapping[str, Any],
    cleaning_log: str,
    daily: pd.DataFrame,
    monthly: pd.DataFrame,
    weekday: pd.DataFrame,
) -> dict[str, Any]:
    raw_rows = _as_int(
        _first(metadata, ("row_count", "downloaded_rows", "downloaded_row_count")),
        "raw row count",
    )
    columns = _as_int(
        _first(metadata, ("column_count", "downloaded_columns")),
        "column count",
    )
    retrieval_value = _first(
        metadata, ("retrieval_date", "snapshot_date", "download_date")
    )
    min_date_value = _first(metadata, ("min_arrest_date", "date_min", "min_date"))
    max_date_value = _first(metadata, ("max_arrest_date", "date_max", "max_date"))
    retrieval_timestamp = pd.to_datetime(retrieval_value, errors="coerce")
    max_date_timestamp = pd.to_datetime(max_date_value, errors="coerce")
    if pd.isna(retrieval_timestamp) or pd.isna(max_date_timestamp):
        raise DeliverableInputError("Metadata contains an invalid retrieval or maximum arrest date")
    freshness_gap_days = int((retrieval_timestamp.normalize() - max_date_timestamp.normalize()).days)
    if freshness_gap_days < 0:
        raise DeliverableInputError("Maximum arrest date occurs after the retrieval date")
    retrieval_date = _format_date(retrieval_value)
    min_date = _format_date(min_date_value)
    max_date = _format_date(max_date_value)

    processed_rows_value = _find_first(
        quality.get("cleaning", {}),
        ("final_processed_rows", "final_rows", "processed_rows", "output_rows"),
    )
    if processed_rows_value is None:
        processed_rows_value = _parse_log_count(
            cleaning_log, ("Final processed rows", "Final rows", "Processed rows")
        )
    processed_rows = _as_int(processed_rows_value, "processed row count")
    derived_columns_value = _find_first(
        quality.get("cleaning", {}), ("derived_temporal_columns",)
    )
    if not isinstance(derived_columns_value, list) or not all(
        isinstance(column, str) for column in derived_columns_value
    ):
        raise DeliverableInputError(
            "data_quality_report.json must list cleaning.derived_temporal_columns"
        )
    derived_columns = [str(column).upper() for column in derived_columns_value]
    source_columns = _extract_schema_columns(quality)
    processed_columns = (
        len(set(source_columns).union(derived_columns))
        if source_columns
        else columns + len([column for column in derived_columns if column not in source_columns])
    )

    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    daily["arrest_count"] = pd.to_numeric(daily["arrest_count"], errors="coerce")
    daily["rolling_7d_mean"] = pd.to_numeric(daily["rolling_7d_mean"], errors="coerce")
    if daily[["date", "arrest_count"]].isna().any().any():
        raise DeliverableInputError("daily_arrests.csv contains invalid dates or counts")
    rolling = daily.dropna(subset=["rolling_7d_mean"])
    if rolling.empty:
        raise DeliverableInputError("daily_arrests.csv has no valid 7-day rolling values")

    monthly = monthly.copy()
    monthly["month_start"] = pd.to_datetime(monthly["month_start"], errors="coerce")
    monthly["avg_arrests_per_calendar_day"] = pd.to_numeric(
        monthly["avg_arrests_per_calendar_day"], errors="coerce"
    )
    if monthly[["month_start", "avg_arrests_per_calendar_day"]].isna().any().any():
        raise DeliverableInputError("monthly_arrests.csv contains invalid months or averages")
    monthly["_is_partial"] = monthly["is_partial_month"].map(_truthy)
    partial_rows = monthly.loc[monthly["_is_partial"]]
    if len(partial_rows) > 1:
        raise DeliverableInputError("monthly_arrests.csv marks more than one partial month")
    is_partial = not partial_rows.empty
    comparison_months = monthly.loc[~monthly["_is_partial"]]
    if comparison_months.empty:
        comparison_months = monthly

    weekday = weekday.copy()
    weekday["weekday_num"] = pd.to_numeric(weekday["weekday_num"], errors="coerce")
    weekday["mean_arrests_per_occurrence"] = pd.to_numeric(
        weekday["mean_arrests_per_occurrence"], errors="coerce"
    )
    if weekday[["weekday_num", "mean_arrests_per_occurrence"]].isna().any().any():
        raise DeliverableInputError("weekday_arrests.csv contains invalid ordering or averages")
    weekday = weekday.sort_values("weekday_num")
    if weekday["weekday"].astype(str).nunique() != 7:
        raise DeliverableInputError("weekday_arrests.csv must contain all seven weekdays")

    rolling_high = rolling.loc[rolling["rolling_7d_mean"].idxmax()]
    rolling_low = rolling.loc[rolling["rolling_7d_mean"].idxmin()]
    daily_high = daily.loc[daily["arrest_count"].idxmax()]
    monthly_high = comparison_months.loc[
        comparison_months["avg_arrests_per_calendar_day"].idxmax()
    ]
    monthly_low = comparison_months.loc[
        comparison_months["avg_arrests_per_calendar_day"].idxmin()
    ]
    weekday_high = weekday.loc[weekday["mean_arrests_per_occurrence"].idxmax()]
    weekday_low = weekday.loc[weekday["mean_arrests_per_occurrence"].idxmin()]
    weekday_low_value = _as_float(
        weekday_low["mean_arrests_per_occurrence"], "lowest weekday average"
    )
    weekday_high_value = _as_float(
        weekday_high["mean_arrests_per_occurrence"], "highest weekday average"
    )
    weekday_relative_to_low_pct = (
        ((weekday_high_value - weekday_low_value) / weekday_low_value) * 100
        if weekday_low_value > 0
        else math.nan
    )
    overall_daily_mean = _as_float(daily["arrest_count"].mean(), "overall daily mean")
    weekday_spread_pct = (
        ((weekday_high_value - weekday_low_value) / overall_daily_mean) * 100
        if overall_daily_mean > 0
        else math.nan
    )

    partial_label = None
    partial_reason = None
    if is_partial:
        partial_row = partial_rows.iloc[0]
        partial_label = str(partial_row["month_label"])
        partial_reason_value = partial_row.get("partial_reason")
        if partial_reason_value is not None and not pd.isna(partial_reason_value):
            partial_reason = str(partial_reason_value).strip() or None

    expected_rows = _first(
        metadata,
        (
            "api_expected_rows_after",
            "api_expected_rows",
            "expected_rows",
            "server_row_count",
        ),
    )
    expected_rows_int = _as_int(expected_rows, "API expected rows") if expected_rows is not None else None
    api_matches = (
        expected_rows_int == raw_rows if expected_rows_int is not None else None
    )

    return {
        "raw_rows": raw_rows,
        "processed_rows": processed_rows,
        "processed_columns": processed_columns,
        "derived_columns": derived_columns,
        "columns": columns,
        "retrieval_date": retrieval_date,
        "min_date": min_date,
        "max_date": max_date,
        "freshness_gap_days": freshness_gap_days,
        "observed_months": int(len(monthly)),
        "all_observed_months_complete": not is_partial,
        "source": str(_first(metadata, ("source",), "NYC Open Data / NYPD")),
        "dataset_name": str(
            _first(metadata, ("dataset_name",), "NYPD Arrest Data, Year to Date")
        ),
        "dataset_id": str(_first(metadata, ("dataset_id",), "uip8-fykc")),
        "raw_file": str(_first(metadata, ("raw_file", "raw_path"), "Not recorded")),
        "expected_rows": expected_rows_int,
        "api_matches": api_matches,
        "rolling_high_date": _format_date(rolling_high["date"]),
        "rolling_high": _as_float(rolling_high["rolling_7d_mean"], "rolling high"),
        "rolling_low_date": _format_date(rolling_low["date"]),
        "rolling_low": _as_float(rolling_low["rolling_7d_mean"], "rolling low"),
        "daily_high_date": _format_date(daily_high["date"]),
        "daily_high": _as_int(daily_high["arrest_count"], "daily high"),
        "monthly_high_label": str(monthly_high["month_label"]),
        "monthly_high": _as_float(
            monthly_high["avg_arrests_per_calendar_day"], "monthly high"
        ),
        "monthly_low_label": str(monthly_low["month_label"]),
        "monthly_low": _as_float(
            monthly_low["avg_arrests_per_calendar_day"], "monthly low"
        ),
        "weekday_high_label": str(weekday_high["weekday"]),
        "weekday_high": weekday_high_value,
        "weekday_low_label": str(weekday_low["weekday"]),
        "weekday_low": weekday_low_value,
        "weekday_spread_pct": weekday_spread_pct,
        "weekday_relative_to_low_pct": weekday_relative_to_low_pct,
        "is_partial": is_partial,
        "partial_label": partial_label,
        "partial_reason": partial_reason,
    }


def _slide_titles(stats: Mapping[str, Any]) -> list[str]:
    title_1 = (
        f"The Frozen NYPD Snapshot Contains {_format_count(stats['raw_rows'])} "
        "Recorded Arrest Events"
    )
    title_2 = (
        "The 7-Day Arrest Average Reached Its Observed High on "
        f"{stats['rolling_high_date']}"
    )
    title_3 = (
        f"{stats['monthly_high_label']} Had the Highest Average Daily Arrest "
        "Activity Among Comparable Months"
    )
    spread = stats["weekday_spread_pct"]
    if math.isfinite(spread) and spread <= 10:
        title_4 = (
            f"Average Arrest Activity Varied by {spread:.1f}% Across Weekdays"
        )
    else:
        title_4 = (
            f"Average Arrest Activity Was Highest on {stats['weekday_high_label']}"
        )
    return [title_1, title_2, title_3, title_4]


def _word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def _render_slide_note(
    title: str,
    finding: str,
    interpretation: str,
    design: str,
    limitation: str,
) -> str:
    note = (
        f"### {title}\n\n"
        f"**Finding**\n\n{finding}\n\n"
        f"**Interpretation**\n\n{interpretation}\n\n"
        f"**Design Rationale**\n\n{design}\n\n"
        f"**Limitation / Caveat**\n\n{limitation}"
    )
    count = _word_count(note)
    if not 120 <= count <= 200:
        raise AssertionError(f"Slide note must contain 120-200 English words; got {count}: {title}")
    return note


def _quality_issues(
    quality: Mapping[str, Any], stats: Mapping[str, Any]
) -> list[str]:
    issues: list[str] = []
    exact_duplicates = _quality_count(
        quality,
        "duplicates",
        (
            "exact_duplicate_rows_beyond_first",
            "exact_duplicate_rows",
            "exact_duplicates",
            "duplicate_rows",
        ),
    )
    duplicate_keys = _quality_count(
        quality,
        "duplicates",
        (
            "rows_with_duplicated_arrest_key",
            "duplicate_arrest_key_rows",
            "duplicate_key_rows",
            "duplicate_arrest_keys",
        ),
    )
    invalid_dates = _quality_count(
        quality,
        "date_validation",
        ("invalid_date_count", "parse_failure_count", "invalid_dates"),
    )
    if exact_duplicates is not None:
        issues.append(f"Exact duplicate rows detected in raw data: {_format_count(exact_duplicates)}.")
    if duplicate_keys is not None:
        issues.append(
            f"Rows associated with duplicated ARREST_KEY values: {_format_count(duplicate_keys)}; "
            "see the audit evidence before any downstream removal."
        )
    if invalid_dates is not None:
        issues.append(f"Unparseable or invalid ARREST_DATE values: {_format_count(invalid_dates)}.")

    positive_missing = [
        item for item in _extract_missingness(quality) if item["missing_percentage"] > 0
    ]
    if positive_missing:
        top = positive_missing[0]
        issues.append(
            f"Highest reported field-level missingness: {top['column']} "
            f"at {top['missing_percentage']:.2f}%."
        )
    categorical_columns = quality.get("categorical_validation", {}).get("columns", {})
    law_category = (
        categorical_columns.get("LAW_CAT_CD", {})
        if isinstance(categorical_columns, Mapping)
        else {}
    )
    if isinstance(law_category, Mapping):
        core_count_value = law_category.get("core_severity_count")
        core_coverage_value = law_category.get(
            "core_severity_coverage_percentage_of_all_rows"
        )
        if core_count_value is not None and core_coverage_value is not None:
            core_count = _as_int(core_count_value, "F/M/V severity count")
            non_core_count = stats["raw_rows"] - core_count
            true_nulls = _as_int(
                law_category.get("missing_count", 0), "LAW_CAT_CD true nulls"
            )
            known_non_core = law_category.get("known_non_core_value_counts", {})
            unrecognised = law_category.get("unrecognised_non_core_value_counts", {})
            sentinels = law_category.get("source_missing_sentinel_counts", {})
            retained_parts = [f"{_format_count(true_nulls)} true nulls"]
            for label, value in (
                ("rows coded `I`", known_non_core.get("I", 0) if isinstance(known_non_core, Mapping) else 0),
                ("rows coded `9`", unrecognised.get("9", 0) if isinstance(unrecognised, Mapping) else 0),
                (
                    "source `(null)` sentinels",
                    sentinels.get("(null)", 0) if isinstance(sentinels, Mapping) else 0,
                ),
            ):
                count = _as_int(value, f"LAW_CAT_CD {label} count")
                if count:
                    retained_parts.append(f"{_format_count(count)} {label}")
            issues.append(
                f"F/M/V severity codes cover {float(core_coverage_value):.2f}% "
                f"({_format_count(core_count)} rows). The remaining "
                f"{_format_count(non_core_count)} rows were retained, including "
                + ", ".join(retained_parts)
                + "."
            )
    if stats["freshness_gap_days"] > 0:
        issues.append(
            f"The latest observed arrest date precedes retrieval by "
            f"{stats['freshness_gap_days']} days; the snapshot is not activity through "
            f"{stats['retrieval_date']}."
        )
    if stats["is_partial"]:
        reason = f" ({stats['partial_reason']})" if stats["partial_reason"] else ""
        issues.append(
            f"{stats['partial_label']} is a partial month{reason}; its monthly value must remain labelled."
        )
    if not issues:
        issues.append(
            "No specific non-zero issue is asserted here; use data_quality_report.json for the "
            "complete field-level audit."
        )
    return issues


def _cleaning_actions(quality: Mapping[str, Any], cleaning_log: str) -> list[str]:
    removed = _find_first(
        quality.get("cleaning", {}),
        (
            "exact_duplicate_rows_removed",
            "exact_duplicates_removed",
            "duplicate_rows_removed",
        ),
    )
    if removed is None:
        removed = _parse_log_count(
            cleaning_log, ("Exact duplicates removed", "Duplicate rows removed")
        )
    invalid_removed = _find_first(
        quality.get("cleaning", {}),
        ("invalid_date_rows_removed", "rows_removed_for_invalid_dates"),
    )
    if invalid_removed is None:
        invalid_removed = _parse_log_count(
            cleaning_log, ("Rows removed for invalid dates", "Invalid-date rows removed")
        )
    invalid_retained = _find_first(
        quality.get("cleaning", {}),
        ("invalid_date_values_retained_as_nat", "invalid_dates_retained_as_nat"),
    )

    actions = [
        "Standardised column names, parsed ARREST_DATE, and created the shared temporal fields.",
        "Trimmed accidental whitespace in string categories without imputing absent values.",
        "Coerced coordinates to numeric while retaining rows with missing coordinates.",
    ]
    if removed is not None:
        actions.append(
            f"Removed {_format_count(int(removed))} confirmed exact duplicate rows."
        )
    else:
        actions.append("Handled exact duplicates exactly as documented in cleaning_log.md.")
    if invalid_removed is not None and invalid_retained is not None:
        actions.append(
            f"Date parsing found {_format_count(int(invalid_retained))} invalid values; the "
            f"policy retains them as NaT and removed {_format_count(int(invalid_removed))} rows."
        )
    return actions


def _build_visual_style_guide(stats: Mapping[str, Any]) -> str:
    partial = ""
    if stats["is_partial"]:
        partial = (
            f"\n- **Partial-month encoding:** {stats['partial_label']} uses lower opacity, a `///` "
            "hatch, and a direct `Partial month` label. Never rely on colour alone.\n"
        )
    return f"""# Part 1 Visual Style Guide

## Language and evidence boundary

- Describe the dataset as **recorded arrest activity**, **arrest events**, or **arrest counts**.
- Do not substitute arrest activity for underlying crime incidence or a crime rate.
- Treat every pattern as descriptive within {stats['min_date']} to {stats['max_date']}; do not attach an unsupported cause.

## Visual system

- Canvas: white (`#FFFFFF`); primary text: charcoal (`#1F2937`); supporting text: slate (`#4B5563`).
- Primary arrest series and complete-month bars: blue (`#0072B2`).
- Daily background marks: cool grey (`#9CA3AF`) at low opacity; the 7-day mean receives the strongest line weight.
- Reference lines and restrained gridlines: light grey (`#D9DEE3`). Keep only gridlines that support value lookup.
- Use direct labels for highlighted values. Legends are reserved for charts with more than one data series.
{partial}
## Stable severity mapping

| LAW_CAT_CD display label | Hex | Use |
|---|---:|---|
| Felony | `{SEVERITY_COLOURS['Felony']}` | Consistent categorical identity |
| Misdemeanor | `{SEVERITY_COLOURS['Misdemeanor']}` | Consistent categorical identity |
| Violation | `{SEVERITY_COLOURS['Violation']}` | Consistent categorical identity |
| Other or missing | `{SEVERITY_COLOURS['Other or missing']}` | Retained, de-emphasised category |

The palette is colourblind-aware and avoids a red-versus-green-only comparison. In monochrome, direct labels, ordering, line weight, and hatch carry meaning in addition to hue.

## Perception and layout rules

- Prefer position and length on a common baseline over angle or area; use bars rather than pie charts.
- Keep chronological axes in chronological order, including Monday through Sunday for weekday categories.
- Use alignment, proximity, and consistent spacing to group titles, annotations, plots, and source notes.
- Use sentence-case, finding-driven titles. Keep slide text short and place detailed caveats in speaker notes.
- Export on a 16:9 canvas with readable presentation typography. Do not use 3D marks, gradients, decorative icons, or rainbow scales.
"""


def _build_chart_contracts(stats: Mapping[str, Any], has_severity: bool) -> str:
    partial_contract = ""
    if stats["is_partial"]:
        partial_contract = (
            f"\n- Snapshot-specific condition: `{stats['partial_label']}` has `is_partial_month=true`; "
            "render it with lower opacity, hatch, and a direct warning label. Exclude it from "
            "complete-month ranking language."
        )
    severity_status = (
        "Generated because the severity admission gate passed and the validated figure is present."
        if has_severity
        else "Not generated because the severity admission gate was not met or its validated figure is absent."
    )
    return f"""# Part 1 Chart Contracts

Every chart is descriptive of the frozen snapshot only. Titles and annotations must say **arrest activity**, never imply underlying crime incidence, and never assign a cause.

## Figure 1 — Dataset completeness overview

- Asset: `figures/part1/{REQUIRED_FIGURES['missingness']}` and matching SVG.
- Source: `outputs/part1/missingness_summary.csv` / `data_quality_report.json`.
- Mark and encoding: horizontal bars; x = missing percentage, y = selected key field, sorted by missing percentage.
- Hierarchy: one accent colour for analytically important missingness; neutral bars elsewhere. Label percentages directly.
- Constraint: show a readable key-field subset rather than a dense screenshot of every column.

## Figure 2 — Daily activity and 7-day mean

- Asset: `figures/part1/{REQUIRED_FIGURES['daily']}` and matching SVG.
- Source: `outputs/part1/daily_arrests.csv` (`date`, `arrest_count`, `rolling_7d_mean`).
- Mark and encoding: thin low-opacity daily line behind a heavier blue 7-day rolling line; x = date, y = arrests.
- Annotation: observed 7-day high of {_format_rate(stats['rolling_high'])} on {stats['rolling_high_date']}.
- Constraint: preserve the calendar sequence, include zero-count dates, and use a non-truncated count axis.

## Figure 3 — Average daily arrests by month

- Asset: `figures/part1/{REQUIRED_FIGURES['monthly']}` and matching SVG.
- Source: `outputs/part1/monthly_arrests.csv` (`month_label`, `avg_arrests_per_calendar_day`, `is_partial_month`).
- Mark and encoding: ordered vertical bars on a common baseline; y = arrests per calendar day in scope.
- Annotation: highest comparable month is {stats['monthly_high_label']} at {_format_rate(stats['monthly_high'])} per day.{partial_contract}

## Figure 4 — Average arrests by weekday occurrence

- Asset: `figures/part1/{REQUIRED_FIGURES['weekday']}` and matching SVG.
- Source: `outputs/part1/weekday_arrests.csv` (`weekday_num`, `weekday`, `mean_arrests_per_occurrence`).
- Mark and encoding: ordered bars, Monday through Sunday; y = mean arrests per occurrence of that weekday.
- Annotation: {stats['weekday_high_label']} is highest at {_format_rate(stats['weekday_high'])}; {stats['weekday_low_label']} is lowest at {_format_rate(stats['weekday_low'])}.
- Constraint: never alphabetise weekdays and do not infer a behavioural cause.

## Optional Figure 5 — Monthly severity composition

- Asset when admitted: `figures/part1/monthly_severity_composition.png` and matching SVG.
- Source: `outputs/part1/monthly_severity_composition.csv`.
- Mark and encoding: 100% stacked monthly bars using `share_of_monthly_arrests_pct` and the stable severity mapping in the style guide; y = share of all valid-date arrests that month.
- Admission: {severity_status}
- Constraint: show classified coverage and retain the explicit Other or missing category. Do not describe severity composition as crime severity.
"""


def _build_slide_plan(stats: Mapping[str, Any], titles: Sequence[str]) -> str:
    partial_bullet = ""
    if stats["is_partial"]:
        partial_bullet = (
            f"\n  - Label {stats['partial_label']} as a partial month and keep it out of the "
            "complete-month rank."
        )
    return f"""# Part 1 Four-Slide Plan

## P1-1 — {titles[0]}

- **Purpose:** establish source, scope, and analytical reliability.
- **On-slide facts:** {stats['source']}; retrieved {stats['retrieval_date']}; {_format_count(stats['raw_rows'])} rows × {stats['columns']} columns; {stats['min_date']} to {stats['max_date']}; latest observation is {stats['freshness_gap_days']} days before retrieval.
- **Key variable groups:** temporal; offence and severity; borough and coordinates; age, sex, and race.
- **Visual:** `figures/part1/{REQUIRED_FIGURES['missingness']}` with a compact snapshot fact strip.
- **Takeaway:** this is an official, frozen event-level dataset, but it measures recorded arrest activity rather than underlying crime incidence and is not activity through the retrieval date.

## P1-2 — {titles[1]}

- **Purpose:** show the observed daily path without letting day-to-day noise dominate.
- **On-slide facts:** 7-day mean ranged from {_format_rate(stats['rolling_low'])} to {_format_rate(stats['rolling_high'])}; observed high ended {stats['rolling_high_date']}.
- **Visual:** `figures/part1/{REQUIRED_FIGURES['daily']}`.
- **Takeaway:** describe the timing and size of observed movement only; do not supply an external cause.

## P1-3 — {titles[2]}

- **Purpose:** compare months fairly after accounting for calendar days observed.
- **On-slide facts:** {stats['monthly_high_label']} averaged {_format_rate(stats['monthly_high'])} arrests per day versus {_format_rate(stats['monthly_low'])} in {stats['monthly_low_label']} among comparable months.{partial_bullet}
- **Visual:** `figures/part1/{REQUIRED_FIGURES['monthly']}`.
- **Takeaway:** average daily activity is the comparison measure; raw totals are supporting context only.

## P1-4 — {titles[3]}

- **Purpose:** compare like-for-like weekday occurrences.
- **On-slide facts:** {stats['weekday_high_label']} averaged {_format_rate(stats['weekday_high'])}; {stats['weekday_low_label']} averaged {_format_rate(stats['weekday_low'])} arrests per occurrence.
- **Visual:** `figures/part1/{REQUIRED_FIGURES['weekday']}`.
- **Takeaway:** weekday differences are descriptive and do not establish behaviour, exposure, or enforcement causes.
"""


def _build_slide_notes(stats: Mapping[str, Any], titles: Sequence[str]) -> str:
    partial_finding = ""
    partial_design = ""
    partial_limitation = ""
    month_scope_sentence = (
        f"All {stats['observed_months']} observed months are calendar-complete, but the "
        f"latest observation is {stats['freshness_gap_days']} days before retrieval."
    )
    if stats["is_partial"]:
        month_scope_sentence = (
            f"The observed range includes an incomplete boundary month, and the latest "
            f"observation is {stats['freshness_gap_days']} days before retrieval."
        )
        partial_finding = (
            f" {stats['partial_label']} is incomplete and excluded from that ranking."
        )
        partial_design = (
            " Its bar uses hatch and a direct label, not colour alone."
        )
        partial_limitation = (
            " The incomplete month covers only observed days and is not a finished total."
        )

    notes = [
        _render_slide_note(
            titles[0],
            (
                f"The {stats['dataset_name']} snapshot was retrieved on "
                f"{stats['retrieval_date']}. It contains {_format_count(stats['raw_rows'])} "
                f"raw rows and {stats['columns']} columns covering {stats['min_date']} through "
                f"{stats['max_date']}; the pipeline retains "
                f"{_format_count(stats['processed_rows'])} rows. {month_scope_sentence}"
            ),
            (
                "The snapshot provides event dates and classification fields for the team's "
                "later temporal, spatial, demographic, and modelling work. The audit identifies "
                "where missingness or validation conditions require explicit handling rather "
                "than silent imputation."
            ),
            (
                "A horizontal missingness chart uses position and length on a common scale, "
                "which supports more accurate comparison than pie slices. A compact fact strip "
                "creates visual hierarchy, while direct percentage labels reduce lookup effort. "
                "Neutral colours keep attention on material quality issues, and aligned groups "
                "separate snapshot scope from field completeness."
            ),
            (
                "These records measure recorded arrest activity, not underlying crime incidence "
                "or a population-adjusted crime rate. Despite the source dataset's Year-to-Date "
                "name, this frozen snapshot is not activity through the retrieval date. It cannot "
                "establish a long-term trend or explain why a pattern occurred."
            ),
        ),
        _render_slide_note(
            titles[1],
            (
                f"Across the observed window, the 7-day average ranged from "
                f"{_format_rate(stats['rolling_low'])} arrests on {stats['rolling_low_date']} to "
                f"{_format_rate(stats['rolling_high'])} on {stats['rolling_high_date']}. The "
                f"largest single-day count was {_format_count(stats['daily_high'])} on "
                f"{stats['daily_high_date']}."
            ),
            (
                "The series shows when recorded arrest activity was relatively higher or lower "
                "inside this snapshot. The smoothed path helps distinguish sustained movement "
                "from isolated daily variation, but the chart remains descriptive and does not "
                "identify an external driver."
            ),
            (
                "Daily counts appear as a thin, low-opacity grey context line, while the blue "
                "7-day mean receives greater weight and contrast. This hierarchy directs the eye "
                "to the stable signal without hiding the underlying observations. Dates remain "
                "chronological, gridlines are restrained, and the observed high is annotated "
                "directly to avoid a separate legend search."
            ),
            (
                "A trailing rolling mean smooths short-lived variation and is less informative "
                "at the boundary of the series. The observed high is not evidence of seasonality, "
                "policy effects, individual risk, or a change in crime incidence."
            ),
        ),
        _render_slide_note(
            titles[2],
            (
                f"Among comparable months, {stats['monthly_high_label']} recorded the highest "
                f"average daily arrest activity at {_format_rate(stats['monthly_high'])}, while "
                f"{stats['monthly_low_label']} recorded the lowest at "
                f"{_format_rate(stats['monthly_low'])} arrests per calendar day in scope."
                f"{partial_finding}"
            ),
            (
                "Dividing by calendar days in scope makes month lengths comparable and avoids "
                "rewarding a 31-day month simply for containing more days. The differences show "
                "variation within the observed Year-to-Date period; they do not by themselves "
                "demonstrate a recurring seasonal pattern."
            ),
            (
                "Ordered bars use length from a shared zero baseline, a perceptually accurate "
                "encoding for comparison. A single blue hue keeps category identity consistent, "
                "and the highest comparable value is labelled directly. Month labels follow "
                "calendar order rather than sorting by magnitude, preserving the temporal story."
                f"{partial_design}"
            ),
            (
                "The metric averages recorded arrests over observed calendar days and has no "
                "population or exposure denominator. It cannot support a crime-rate statement or "
                "a causal explanation for month-to-month differences."
                f"{partial_limitation}"
            ),
        ),
        _render_slide_note(
            titles[3],
            (
                f"{stats['weekday_high_label']} had the highest mean at "
                f"{_format_rate(stats['weekday_high'])} arrests per occurrence, compared with "
                f"{_format_rate(stats['weekday_low'])} on {stats['weekday_low_label']}. "
                f"The {stats['weekday_high_label']} average was "
                f"{stats['weekday_relative_to_low_pct']:.1f}% higher than the "
                f"{stats['weekday_low_label']} average within this descriptive metric."
            ),
            (
                "Averaging by the number of each weekday observed avoids bias when the snapshot "
                "contains unequal counts of Mondays, Tuesdays, or other weekdays. The resulting "
                "profile describes timing in recorded arrest activity, not an explanation for "
                "people's behaviour or police operations."
            ),
            (
                "Bars use position and length on one baseline, and weekdays remain in the familiar "
                "Monday-to-Sunday sequence rather than an alphabetical or rank order. Consistent "
                "blue marks support similarity; direct labels reduce legend dependence; light "
                "gridlines aid value lookup without competing with the data."
            ),
            (
                "The comparison is not adjusted for population, mobility, events, enforcement "
                "deployment, or exposure. It reflects only this Year-to-Date snapshot and cannot "
                "establish that weekday alone produced the observed difference or that underlying "
                "crime followed the same profile."
            ),
        ),
    ]
    return "# Part 1 Speaker Notes\n\n" + "\n\n---\n\n".join(notes) + "\n"


def _build_team_handoff(
    stats: Mapping[str, Any],
    quality: Mapping[str, Any],
    cleaning_log: str,
) -> str:
    expected_row_line = "Not available from the API metadata."
    if stats["expected_rows"] is not None:
        match_text = "matches" if stats["api_matches"] else "does not match"
        expected_row_line = (
            f"{_format_count(stats['expected_rows'])}; this {match_text} the downloaded "
            f"count of {_format_count(stats['raw_rows'])}."
        )

    issues = "\n".join(f"- {item}" for item in _quality_issues(quality, stats))
    actions = "\n".join(
        f"- {item}" for item in _cleaning_actions(quality, cleaning_log)
    )
    derived = "\n".join(f"- `{column}`" for column in stats["derived_columns"])
    schema_columns = set(_extract_schema_columns(quality))
    preferred_fields = [
        "ARREST_KEY",
        "ARREST_DATE",
        "ARREST_BORO",
        "LAW_CAT_CD",
        "AGE_GROUP",
        "PERP_SEX",
        "PERP_RACE",
        "LATITUDE",
        "LONGITUDE",
    ]
    available_fields = [field for field in preferred_fields if not schema_columns or field in schema_columns]
    field_text = ", ".join(f"`{field}`" for field in available_fields)
    partial_warning = ""
    if stats["is_partial"]:
        partial_warning = (
            f"\n- **{stats['partial_label']} is a partial month.** Preserve its label and do not "
            "rank it as a complete monthly total."
        )
    else:
        partial_warning = (
            f"\n- All {stats['observed_months']} observed months are complete calendar months; "
            "this does not make the snapshot complete through the retrieval date."
        )

    return f"""# Part 1 Team Handoff

## Dataset snapshot

| Item | Verified value |
|---|---|
| Dataset | {stats['dataset_name']} (`{stats['dataset_id']}`) |
| Source | {stats['source']} |
| Retrieval date | {stats['retrieval_date']} |
| Raw file | `{stats['raw_file']}` |
| Raw size | {_format_count(stats['raw_rows'])} rows × {stats['columns']} columns |
| Processed size | {_format_count(stats['processed_rows'])} rows × {stats['processed_columns']} columns |
| Arrest-date range | {stats['min_date']} to {stats['max_date']} |
| Observed months | {stats['observed_months']} |
| Latest-record to retrieval gap | {stats['freshness_gap_days']} calendar days |
| API expected rows | {expected_row_line} |

## Important quality issues

{issues}

The complete machine-readable evidence is in `outputs/part1/data_quality_report.json`; missing values were not automatically imputed or discarded.

## Cleaning performed

{actions}

The complete action counts and rationales are in `outputs/part1/cleaning_log.md`.

## Derived columns

{derived}

## Files other members should use

- **Shared baseline:** `data/processed/nypd_arrests_clean.csv`
- **Snapshot provenance:** `outputs/part1/dataset_snapshot_metadata.json`
- **Quality evidence:** `outputs/part1/data_quality_report.json`, `schema_summary.csv`, and `missingness_summary.csv`
- **Temporal evidence:** `daily_arrests.csv`, `monthly_arrests.csv`, `weekday_arrests.csv`, and `key_findings.json` under `outputs/part1/`
- **Important source and derived fields:** {field_text}, `YEAR`, `MONTH`, `MONTH_NAME`, `DAY_OF_WEEK`, and `DAY_OF_WEEK_NUM`

## Important warnings

- **Do not interpret arrests as crime incidence or a crime rate.** These records reflect recorded arrest and enforcement activity.
- Do not infer a cause from the descriptive temporal patterns without external evidence.
- Do not impute absent demographic, offence, or coordinate values without a separately documented analytical reason.
- Despite the source dataset's Year-to-Date name, this snapshot ends on {stats['max_date']} and is not activity through {stats['retrieval_date']}.
- This snapshot is not evidence of a long-term trend.{partial_warning}
"""


def _build_ai_usage_note() -> str:
    return """# Generative AI Usage Note

Generative AI was used to assist with official API retrieval code, data-validation logic, conservative cleaning code, temporal aggregation, Python visualisation implementation, reproducibility checks, and preparation of draft slide-ready materials.

All reported values, analytical choices, generated outputs, and conclusions were checked against the frozen official dataset and machine-readable analysis artifacts before inclusion. AI assistance was not treated as an independent factual source, and no causal interpretation was accepted without supporting evidence.
"""


def _validate_language(documents: Mapping[str, str], is_partial: bool) -> None:
    combined = "\n".join(documents.values())
    prohibited = {
        r"\bhigher crime rate\b": "unsupported crime-rate language",
        r"\blower crime rate\b": "unsupported crime-rate language",
        r"\bcaused by\b": "unsupported causal language",
        r"\bresulted from\b": "unsupported causal language",
    }
    for pattern, reason in prohibited.items():
        if re.search(pattern, combined, flags=re.IGNORECASE):
            raise AssertionError(f"Generated deliverables contain {reason}: {pattern}")
    if not is_partial and re.search(r"partial[- ]month", combined, flags=re.IGNORECASE):
        raise AssertionError("A partial-month statement was emitted for a complete-month snapshot")


def build_deliverables(
    project_root: str | Path = PROJECT_ROOT,
    *,
    metadata_path: str | Path | None = None,
    data_quality_path: str | Path | None = None,
    cleaning_log_path: str | Path | None = None,
    temporal_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    figures_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Generate all six Part 1 reader-facing Markdown deliverables.

    Inputs are validated before any output is written. The function returns a
    mapping from deliverable filename to its absolute path.
    """

    root = Path(project_root).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve() if output_dir else root / "outputs" / "part1"
    temporal = Path(temporal_dir).expanduser().resolve() if temporal_dir else output
    figures = Path(figures_dir).expanduser().resolve() if figures_dir else root / "figures" / "part1"
    metadata_file = (
        Path(metadata_path).expanduser().resolve()
        if metadata_path
        else output / "dataset_snapshot_metadata.json"
    )
    quality_file = (
        Path(data_quality_path).expanduser().resolve()
        if data_quality_path
        else output / "data_quality_report.json"
    )
    log_file = (
        Path(cleaning_log_path).expanduser().resolve()
        if cleaning_log_path
        else output / "cleaning_log.md"
    )

    metadata = _read_json(metadata_file)
    quality = _read_json(quality_file)
    key_findings = _read_json(temporal / "key_findings.json")
    if not log_file.is_file():
        raise DeliverableInputError(f"Required cleaning log does not exist: {log_file}")
    cleaning_log = log_file.read_text(encoding="utf-8")
    if not cleaning_log.strip():
        raise DeliverableInputError(f"Cleaning log is empty: {log_file}")

    daily = _read_csv(
        temporal / REQUIRED_TEMPORAL_FILES["daily"],
        ("date", "arrest_count", "rolling_7d_mean"),
    )
    monthly = _read_csv(
        temporal / REQUIRED_TEMPORAL_FILES["monthly"],
        ("month_start", "month_label", "avg_arrests_per_calendar_day", "is_partial_month"),
    )
    weekday = _read_csv(
        temporal / REQUIRED_TEMPORAL_FILES["weekday"],
        ("weekday_num", "weekday", "mean_arrests_per_occurrence"),
    )

    for figure_name in REQUIRED_FIGURES.values():
        figure_path = figures / figure_name
        if not figure_path.is_file() or figure_path.stat().st_size == 0:
            raise DeliverableInputError(f"Required figure is missing or empty: {figure_path}")

    severity_file = temporal / "monthly_severity_composition.csv"
    has_severity_table = False
    if severity_file.is_file() and severity_file.stat().st_size > 0:
        severity = _read_csv(
            severity_file,
            (
                "month_label",
                "severity",
                "share_of_monthly_arrests_pct",
                "classified_coverage_pct",
            ),
        )
        has_severity_table = not severity.empty

    severity_assessment = key_findings.get("severity_analysis", {})
    severity_eligible = (
        _truthy(severity_assessment.get("eligible"))
        if isinstance(severity_assessment, Mapping)
        else False
    )
    has_severity = (
        has_severity_table
        and severity_eligible
        and (figures / "monthly_severity_composition.png").is_file()
    )

    # Reading key_findings is mandatory provenance validation. Its source/window
    # fields are cross-checked when present; all prose numbers still come from CSV.
    finding_source_payload = key_findings.get("source")
    finding_source = (
        finding_source_payload.get("source")
        if isinstance(finding_source_payload, Mapping)
        else finding_source_payload
    )
    if (
        finding_source
        and str(metadata.get("source", ""))
        and str(finding_source) != str(metadata["source"])
    ):
        raise DeliverableInputError("key_findings.json source does not match snapshot metadata")

    stats = _normalise_inputs(metadata, quality, cleaning_log, daily, monthly, weekday)
    titles = _slide_titles(stats)
    documents = {
        "visual_style_guide.md": _build_visual_style_guide(stats),
        "chart_contracts.md": _build_chart_contracts(stats, has_severity),
        "slide_plan.md": _build_slide_plan(stats, titles),
        "slide_notes.md": _build_slide_notes(stats, titles),
        "team_handoff.md": _build_team_handoff(stats, quality, cleaning_log),
        "ai_usage_note.md": _build_ai_usage_note(),
    }
    if tuple(documents) != DELIVERABLE_NAMES:
        raise AssertionError("Internal deliverable set does not match the declared contract")
    _validate_language(documents, stats["is_partial"])

    output.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name, content in documents.items():
        destination = output / name
        destination.write_text(content.rstrip() + "\n", encoding="utf-8")
        written[name] = destination.resolve()
    return written


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build CA6002 Part 1 reader-facing Markdown deliverables."
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--data-quality", type=Path)
    parser.add_argument("--cleaning-log", type=Path)
    parser.add_argument("--temporal-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--figures-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    written = build_deliverables(
        project_root=args.project_root,
        metadata_path=args.metadata,
        data_quality_path=args.data_quality,
        cleaning_log_path=args.cleaning_log,
        temporal_dir=args.temporal_dir,
        output_dir=args.output_dir,
        figures_dir=args.figures_dir,
    )
    for path in written.values():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
