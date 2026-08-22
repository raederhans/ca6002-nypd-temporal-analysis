"""Reproducible temporal analysis and static figures for CA6002 Part 1.

The module reads the shared processed NYPD arrest snapshot together with the
dataset audit and snapshot metadata.  It writes auditable aggregate tables,
machine-readable findings, and presentation-ready PNG/SVG figures.

Arrest records describe recorded enforcement activity.  They are not a
measure of underlying crime incidence, and this distinction is repeated on
every analytical figure produced here.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle


LOGGER = logging.getLogger(__name__)

DEFAULT_PROCESSED_CSV = Path("data/processed/nypd_arrests_clean.csv")
DEFAULT_AUDIT_CSV = Path("outputs/part1/missingness_summary.csv")
DEFAULT_METADATA_JSON = Path("outputs/part1/dataset_snapshot_metadata.json")
DEFAULT_OUTPUTS_DIR = Path("outputs/part1")
DEFAULT_FIGURES_DIR = Path("figures/part1")

WEEKDAY_NAMES = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
SEVERITY_ORDER = ["Felony", "Misdemeanor", "Violation"]
SEVERITY_DISPLAY_ORDER = [*SEVERITY_ORDER, "Other or missing"]
SEVERITY_CODE_MAP = {
    "F": "Felony",
    "FELONY": "Felony",
    "M": "Misdemeanor",
    "MISDEMEANOR": "Misdemeanor",
    "V": "Violation",
    "VIOLATION": "Violation",
}
SEVERITY_MIN_CLASSIFIED_COVERAGE = 0.90
SEVERITY_MIN_MONTHLY_COVERAGE = 0.80

# Okabe-Ito-derived categorical roots plus restrained neutral scaffolding.
PALETTE = {
    "ink": "#24313A",
    "muted": "#68737D",
    "grid": "#D9E0E5",
    "daily": "#A8B4BE",
    "blue": "#0072B2",
    "blue_dark": "#174A6E",
    "blue_light": "#D7EAF4",
    "gold": "#E69F00",
    "pink": "#CC79A7",
    "white": "#FFFFFF",
    "panel": "#F8FAFB",
}
SEVERITY_COLORS = {
    "Felony": PALETTE["blue"],
    "Misdemeanor": PALETTE["gold"],
    "Violation": PALETTE["pink"],
    "Other or missing": "#9AA3AA",
}
SEVERITY_HATCHES = {
    "Felony": "",
    "Misdemeanor": "///",
    "Violation": "...",
    "Other or missing": "xx",
}

RC_PARAMS = {
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.labelcolor": PALETTE["ink"],
    "axes.edgecolor": PALETTE["ink"],
    "axes.linewidth": 0.8,
    "axes.titlecolor": PALETTE["ink"],
    "xtick.color": PALETTE["muted"],
    "ytick.color": PALETTE["muted"],
    "figure.facecolor": PALETTE["white"],
    "axes.facecolor": PALETTE["white"],
    "savefig.facecolor": PALETTE["white"],
    "svg.hashsalt": "ca6002-part1-temporal",
}

KEY_MISSINGNESS_FIELDS = [
    "ARREST_DATE",
    "LAW_CAT_CD",
    "ARREST_BORO",
    "AGE_GROUP",
    "PERP_SEX",
    "PERP_RACE",
    "OFNS_DESC",
    "PD_DESC",
    "LATITUDE",
    "LONGITUDE",
]


def _count_word(value: int) -> str:
    """Use compact English words for small chart-caption counts."""

    words = {
        0: "zero",
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine",
        10: "ten",
        11: "eleven",
        12: "twelve",
    }
    return words.get(value, str(value))


def _normalise_name(value: object) -> str:
    """Return a comparison-safe field name without changing source data."""

    return str(value).strip().upper()


def _resolve_column(frame: pd.DataFrame, canonical_name: str) -> str:
    """Resolve a column case-insensitively and fail clearly if it is absent."""

    lookup = {_normalise_name(column): str(column) for column in frame.columns}
    key = _normalise_name(canonical_name)
    if key not in lookup:
        raise KeyError(
            f"Required column {canonical_name!r} was not found. "
            f"Available columns: {', '.join(map(str, frame.columns))}"
        )
    return lookup[key]


def _optional_column(frame: pd.DataFrame, canonical_name: str) -> str | None:
    lookup = {_normalise_name(column): str(column) for column in frame.columns}
    return lookup.get(_normalise_name(canonical_name))


def _python_scalar(value: Any) -> Any:
    """Convert pandas/numpy scalars into deterministic JSON-compatible values."""

    if value is None or value is pd.NA:
        return None
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _project_root_from_outputs(outputs_dir: Path) -> Path:
    resolved = outputs_dir.resolve()
    if resolved.name == "part1" and resolved.parent.name == "outputs":
        return resolved.parent.parent
    return Path.cwd().resolve()


def _portable_project_path(path: Path | str, project_root: Path) -> str:
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return candidate.as_posix()


def _round(value: Any, digits: int = 2) -> float | None:
    scalar = _python_scalar(value)
    if scalar is None:
        return None
    return round(float(scalar), digits)


def load_metadata(path: Path | str) -> dict[str, Any]:
    """Load snapshot metadata, using explicit official-source defaults if absent."""

    metadata_path = Path(path)
    if not metadata_path.exists():
        LOGGER.warning(
            "Snapshot metadata not found at %s; using source defaults and "
            "deriving temporal coverage from the processed data.",
            metadata_path,
        )
        return {
            "dataset_name": "NYPD Arrest Data, Year to Date",
            "dataset_id": "uip8-fykc",
            "source": "NYC Open Data / NYPD",
            "retrieval_date": None,
        }
    with metadata_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Metadata must be a JSON object: {metadata_path}")
    return payload


def load_processed_data(path: Path | str) -> pd.DataFrame:
    """Read the shared processed CSV without mutating or overwriting it."""

    processed_path = Path(path)
    if not processed_path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {processed_path}")
    frame = pd.read_csv(processed_path, low_memory=False)
    if frame.empty:
        raise ValueError(f"Processed dataset is empty: {processed_path}")
    _resolve_column(frame, "ARREST_DATE")
    return frame


def prepare_temporal_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return valid-date analysis rows and a transparent date parsing audit."""

    date_column = _resolve_column(frame, "ARREST_DATE")
    parsed_dates = pd.to_datetime(frame[date_column], errors="coerce")
    valid_mask = parsed_dates.notna()
    prepared = frame.loc[valid_mask].copy()
    prepared["__analysis_date"] = parsed_dates.loc[valid_mask].dt.normalize()
    if prepared.empty:
        raise ValueError("ARREST_DATE contains no parseable dates; temporal analysis cannot run.")
    audit = {
        "input_rows": int(len(frame)),
        "valid_date_rows": int(valid_mask.sum()),
        "invalid_date_rows_excluded_from_temporal_analysis": int((~valid_mask).sum()),
    }
    return prepared, audit


def calculate_daily_arrests(prepared: pd.DataFrame) -> pd.DataFrame:
    """Count records per calendar day and reindex the complete observed range."""

    if "__analysis_date" not in prepared.columns:
        raise KeyError("prepare_temporal_data must be called before daily aggregation.")
    daily_counts = prepared.groupby("__analysis_date", sort=True).size()
    full_range = pd.date_range(
        daily_counts.index.min(), daily_counts.index.max(), freq="D", name="date"
    )
    daily = daily_counts.reindex(full_range, fill_value=0).rename("arrest_count").to_frame()
    daily["rolling_7d_mean"] = daily["arrest_count"].rolling(
        window=7, min_periods=7
    ).mean()
    daily["is_zero_count_day"] = daily["arrest_count"].eq(0)
    daily = daily.reset_index()
    daily["arrest_count"] = daily["arrest_count"].astype("int64")
    return daily[["date", "arrest_count", "rolling_7d_mean", "is_zero_count_day"]]


def _partial_reason(
    month_start: pd.Timestamp,
    analysis_start: pd.Timestamp,
    analysis_end: pd.Timestamp,
) -> str:
    month_end = month_start + pd.offsets.MonthEnd(0)
    reasons: list[str] = []
    if analysis_start > month_start and analysis_start <= month_end:
        reasons.append(f"starts {analysis_start:%Y-%m-%d}")
    if analysis_end < month_end and analysis_end >= month_start:
        reasons.append(f"ends {analysis_end:%Y-%m-%d}")
    return "; ".join(reasons)


def calculate_monthly_arrests(daily: pd.DataFrame) -> pd.DataFrame:
    """Aggregate by month using calendar days in the analysis range as denominator."""

    if daily.empty:
        raise ValueError("Daily arrest table is empty.")
    working = daily.copy()
    working["month_start"] = working["date"].dt.to_period("M").dt.to_timestamp()
    monthly = (
        working.groupby("month_start", sort=True)
        .agg(
            arrest_count=("arrest_count", "sum"),
            calendar_days_in_scope=("date", "size"),
        )
        .reset_index()
    )
    monthly["calendar_days_in_month"] = monthly["month_start"].dt.days_in_month
    monthly["avg_arrests_per_calendar_day"] = (
        monthly["arrest_count"] / monthly["calendar_days_in_scope"]
    )
    monthly["is_partial_month"] = (
        monthly["calendar_days_in_scope"] < monthly["calendar_days_in_month"]
    )
    analysis_start = pd.Timestamp(working["date"].min())
    analysis_end = pd.Timestamp(working["date"].max())
    monthly["partial_reason"] = monthly["month_start"].map(
        lambda month: _partial_reason(month, analysis_start, analysis_end)
    )
    monthly["month_label"] = monthly["month_start"].dt.strftime("%b %Y")
    monthly["arrest_count"] = monthly["arrest_count"].astype("int64")
    monthly["calendar_days_in_scope"] = monthly["calendar_days_in_scope"].astype("int64")
    monthly["calendar_days_in_month"] = monthly["calendar_days_in_month"].astype("int64")
    return monthly[
        [
            "month_start",
            "month_label",
            "arrest_count",
            "calendar_days_in_scope",
            "calendar_days_in_month",
            "avg_arrests_per_calendar_day",
            "is_partial_month",
            "partial_reason",
        ]
    ]


def calculate_weekday_arrests(daily: pd.DataFrame) -> pd.DataFrame:
    """Compute mean arrests for each weekday using calendar occurrences."""

    if daily.empty:
        raise ValueError("Daily arrest table is empty.")
    working = daily.copy()
    working["weekday_num"] = working["date"].dt.dayofweek
    grouped = (
        working.groupby("weekday_num", sort=True)
        .agg(
            arrest_count=("arrest_count", "sum"),
            calendar_occurrences=("date", "size"),
        )
        .reindex(range(7), fill_value=0)
        .reset_index()
    )
    grouped["weekday"] = grouped["weekday_num"].map(dict(enumerate(WEEKDAY_NAMES)))
    grouped["mean_arrests_per_occurrence"] = np.where(
        grouped["calendar_occurrences"].gt(0),
        grouped["arrest_count"] / grouped["calendar_occurrences"],
        np.nan,
    )
    grouped["arrest_count"] = grouped["arrest_count"].astype("int64")
    grouped["calendar_occurrences"] = grouped["calendar_occurrences"].astype("int64")
    return grouped[
        [
            "weekday_num",
            "weekday",
            "arrest_count",
            "calendar_occurrences",
            "mean_arrests_per_occurrence",
        ]
    ]


def calculate_monthly_severity(
    prepared: pd.DataFrame,
    monthly: pd.DataFrame,
    *,
    min_classified_coverage: float = SEVERITY_MIN_CLASSIFIED_COVERAGE,
    min_monthly_coverage: float = SEVERITY_MIN_MONTHLY_COVERAGE,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build long-form monthly F/M/V composition and assess figure eligibility.

    The chart share uses all valid-date arrests as the denominator and exposes
    missing/unrecognised values as ``Other or missing``.  A second classified
    share retains the within-F/M/V denominator for audit and reuse.
    """

    law_column = _optional_column(prepared, "LAW_CAT_CD")
    month_lookup = monthly.set_index("month_start")
    month_starts = pd.DatetimeIndex(monthly["month_start"])
    if law_column is None:
        empty = pd.DataFrame(
            columns=[
                "month_start",
                "month_label",
                "severity",
                "arrest_count",
                "share_of_monthly_arrests_pct",
                "share_of_classified_pct",
                "month_classified_count",
                "month_total_arrests",
                "classified_coverage_pct",
                "is_partial_month",
            ]
        )
        return empty, {
            "eligible": False,
            "figure_generated": False,
            "reason": "LAW_CAT_CD is absent from the processed dataset.",
            "classification_coverage_pct": 0.0,
            "minimum_monthly_classification_coverage_pct": 0.0,
            "recognized_rows": 0,
            "unclassified_rows": int(len(prepared)),
            "non_missing_unrecognized_rows": 0,
            "categories_present": [],
        }

    raw_codes = prepared[law_column].astype("string").str.strip().str.upper()
    severity = raw_codes.map(SEVERITY_CODE_MAP)
    severity_frame = pd.DataFrame(
        {
            "month_start": prepared["__analysis_date"].dt.to_period("M").dt.to_timestamp(),
            "severity": severity,
        },
        index=prepared.index,
    )
    total_by_month = severity_frame.groupby("month_start", sort=True).size().reindex(
        month_starts, fill_value=0
    )
    classified_by_month = (
        severity_frame.dropna(subset=["severity"])
        .groupby("month_start", sort=True)
        .size()
        .reindex(month_starts, fill_value=0)
    )
    counts = (
        severity_frame.dropna(subset=["severity"])
        .groupby(["month_start", "severity"], sort=True)
        .size()
        .unstack(fill_value=0)
        .reindex(index=month_starts, columns=SEVERITY_ORDER, fill_value=0)
    )

    rows: list[dict[str, Any]] = []
    for month_start in month_starts:
        total = int(total_by_month.loc[month_start])
        classified = int(classified_by_month.loc[month_start])
        recognised_counts = {
            category: int(counts.loc[month_start, category])
            for category in SEVERITY_ORDER
        }
        display_counts = {
            **recognised_counts,
            "Other or missing": total - classified,
        }
        for category in SEVERITY_DISPLAY_ORDER:
            count = int(display_counts[category])
            rows.append(
                {
                    "month_start": month_start,
                    "month_label": str(month_lookup.loc[month_start, "month_label"]),
                    "severity": category,
                    "arrest_count": count,
                    "share_of_monthly_arrests_pct": (
                        100.0 * count / total if total else np.nan
                    ),
                    "share_of_classified_pct": (
                        100.0 * count / classified
                        if classified and category in SEVERITY_ORDER
                        else np.nan
                    ),
                    "month_classified_count": classified,
                    "month_total_arrests": total,
                    "classified_coverage_pct": 100.0 * classified / total if total else np.nan,
                    "is_partial_month": bool(
                        month_lookup.loc[month_start, "is_partial_month"]
                    ),
                }
            )
    composition = pd.DataFrame(rows)

    recognized_rows = int(severity.notna().sum())
    non_missing_rows = int(raw_codes.notna().sum())
    non_missing_unrecognized_rows = int((raw_codes.notna() & severity.isna()).sum())
    overall_coverage = recognized_rows / len(prepared) if len(prepared) else 0.0
    monthly_coverage = classified_by_month.div(total_by_month.replace(0, np.nan))
    minimum_monthly_coverage = float(monthly_coverage.min()) if monthly_coverage.notna().any() else 0.0
    categories_present = [
        category for category in SEVERITY_ORDER if int((severity == category).sum()) > 0
    ]

    criteria = {
        "overall_classified_coverage_at_least": min_classified_coverage,
        "each_month_classified_coverage_at_least": min_monthly_coverage,
        "minimum_distinct_categories": 2,
        "minimum_months": 2,
    }
    failures: list[str] = []
    if overall_coverage < min_classified_coverage:
        failures.append(
            f"overall F/M/V coverage is {overall_coverage:.1%}, below {min_classified_coverage:.0%}"
        )
    if minimum_monthly_coverage < min_monthly_coverage:
        failures.append(
            "minimum monthly F/M/V coverage is "
            f"{minimum_monthly_coverage:.1%}, below {min_monthly_coverage:.0%}"
        )
    if len(categories_present) < 2:
        failures.append("fewer than two recognised severity categories are present")
    if len(month_starts) < 2:
        failures.append("fewer than two months are available for a temporal comparison")

    eligible = not failures
    assessment: dict[str, Any] = {
        "eligible": eligible,
        "figure_generated": False,
        "reason": (
            "Severity composition meets the stated completeness criteria."
            if eligible
            else "; ".join(failures) + "."
        ),
        "criteria": criteria,
        "classification_coverage_pct": round(100.0 * overall_coverage, 2),
        "minimum_monthly_classification_coverage_pct": round(
            100.0 * minimum_monthly_coverage, 2
        ),
        "recognized_rows": recognized_rows,
        "unclassified_rows": int(len(prepared) - recognized_rows),
        "non_missing_law_category_rows": non_missing_rows,
        "non_missing_unrecognized_rows": non_missing_unrecognized_rows,
        "categories_present": categories_present,
        "chart_share_denominator": "All valid-date arrest records within each month",
        "classified_share_denominator": "Recognised F/M/V arrest records within each month",
        "coverage_denominator": "All valid-date arrest records within each month",
    }
    return composition, assessment


def load_missingness_summary(
    audit_csv: Path | str,
    processed: pd.DataFrame,
    *,
    max_fields: int = 12,
) -> tuple[pd.DataFrame, str, str]:
    """Return a compact key-field missingness table from audit or processed data."""

    audit_path = Path(audit_csv)
    source_label = "processed dataset fallback"
    if audit_path.exists():
        audit = pd.read_csv(audit_path)
        column_name_col = _optional_column(audit, "column_name")
        candidates = [
            (
                "effective_missing_count",
                "effective_missing_percentage",
                "effective_missing",
            ),
            ("missing_count", "missing_percentage", "missing"),
            ("null_count", "null_percentage", "null"),
        ]
        count_col: str | None = None
        percentage_col: str | None = None
        metric = "processed_null_fallback"
        for candidate_count, candidate_percentage, candidate_metric in candidates:
            resolved_percentage = _optional_column(audit, candidate_percentage)
            if resolved_percentage is not None:
                count_col = _optional_column(audit, candidate_count)
                percentage_col = resolved_percentage
                metric = candidate_metric
                break
        if column_name_col is not None and percentage_col is not None:
            missingness = pd.DataFrame(
                {
                    "column_name": audit[column_name_col].astype("string").str.strip(),
                    "null_percentage": pd.to_numeric(
                        audit[percentage_col], errors="coerce"
                    ),
                    "null_count": (
                        pd.to_numeric(audit[count_col], errors="coerce")
                        if count_col is not None
                        else np.nan
                    ),
                }
            )
            missingness = missingness.dropna(subset=["column_name", "null_percentage"])
            source_label = str(audit_path.as_posix())
        else:
            LOGGER.warning(
                "Audit %s lacks column_name/null_percentage; deriving missingness "
                "from the processed dataset.",
                audit_path,
            )
            missingness = _missingness_from_frame(processed)
            metric = "processed_null_fallback"
    else:
        LOGGER.warning(
            "Schema audit not found at %s; deriving missingness from processed data.",
            audit_path,
        )
        missingness = _missingness_from_frame(processed)
        metric = "processed_null_fallback"

    missingness["__key"] = missingness["column_name"].map(_normalise_name)
    priority = {name: position for position, name in enumerate(KEY_MISSINGNESS_FIELDS)}
    # Preserve every field with measured missingness before filling the compact
    # view with important zero-missing reference fields.  This prevents small
    # but real percentages from being rounded away or omitted by a priority list.
    selected = missingness[missingness["null_percentage"].gt(0)].copy()
    selected = selected.sort_values(
        ["null_percentage", "column_name"], ascending=[False, True]
    ).head(max_fields)

    if len(selected) < min(max_fields, len(missingness)):
        remaining = missingness[~missingness.index.isin(selected.index)].copy()
        remaining["__priority"] = remaining["__key"].map(priority).fillna(len(priority))
        remaining = remaining.sort_values(
            ["__priority", "column_name"], ascending=[True, True]
        )
        selected = pd.concat([selected, remaining.head(max_fields - len(selected))])
    selected = selected.head(max_fields).copy()
    selected["null_percentage"] = selected["null_percentage"].clip(lower=0, upper=100)
    selected = selected.sort_values(
        ["null_percentage", "column_name"], ascending=[True, True]
    ).reset_index(drop=True)
    return selected[["column_name", "null_count", "null_percentage"]], source_label, metric


def _missingness_from_frame(frame: pd.DataFrame) -> pd.DataFrame:
    counts = frame.isna().sum()
    return pd.DataFrame(
        {
            "column_name": counts.index.astype(str),
            "null_count": counts.to_numpy(dtype="int64"),
            "null_percentage": counts.to_numpy(dtype="float64") / len(frame) * 100.0,
        }
    )


def _source_line(metadata: Mapping[str, Any]) -> str:
    source = metadata.get("source") or "NYC Open Data / NYPD"
    dataset_id = metadata.get("dataset_id") or "uip8-fykc"
    retrieval = metadata.get("retrieval_date")
    retrieval_text = f"; frozen snapshot retrieved {retrieval}" if retrieval else ""
    return f"Source: {source}, dataset {dataset_id}{retrieval_text}."


def _new_figure(title: str, subtitle: str) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(16, 9), dpi=100)
    fig.subplots_adjust(left=0.095, right=0.965, bottom=0.18, top=0.81)
    fig.text(
        0.07,
        0.94,
        title,
        ha="left",
        va="top",
        fontsize=24,
        fontweight="bold",
        color=PALETTE["ink"],
    )
    fig.text(
        0.07,
        0.885,
        subtitle,
        ha="left",
        va="top",
        fontsize=13,
        color=PALETTE["muted"],
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def _footer(
    fig: plt.Figure,
    metadata: Mapping[str, Any],
    caveat: str,
) -> None:
    fig.text(
        0.07,
        0.075,
        _source_line(metadata),
        ha="left",
        va="bottom",
        fontsize=10.5,
        color=PALETTE["muted"],
    )
    fig.text(
        0.07,
        0.045,
        f"Note: {caveat}",
        ha="left",
        va="bottom",
        fontsize=10.5,
        color=PALETTE["muted"],
    )


def _save_figure(fig: plt.Figure, base_path: Path, title: str) -> list[str]:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    png_path = base_path.with_suffix(".png")
    svg_path = base_path.with_suffix(".svg")
    fig.savefig(
        png_path,
        dpi=100,
        format="png",
        metadata={"Software": "Matplotlib", "Title": title},
    )
    fig.savefig(
        svg_path,
        format="svg",
        metadata={
            "Creator": "CA6002 Part 1 temporal analysis",
            "Title": title,
            "Date": None,
        },
    )
    plt.close(fig)
    return [png_path.as_posix(), svg_path.as_posix()]


def plot_missingness(
    missingness: pd.DataFrame,
    metadata: Mapping[str, Any],
    output_base: Path,
    *,
    metric: str,
) -> list[str]:
    effective = metric == "effective_missing"
    title = (
        "Effective missingness across analysis-relevant fields"
        if effective
        else "Missingness across analysis-relevant fields"
    )
    subtitle = (
        "Null, blank-string and documented sentinel share in the frozen snapshot; "
        "selected temporal, classification, demographic, offence and location fields"
        if effective
        else "Null share in the frozen snapshot; selected temporal, classification, "
        "demographic, offence and location fields"
    )
    fig, ax = _new_figure(title, subtitle)
    labels = missingness["column_name"].astype(str).str.replace("_", " ", regex=False)
    values = missingness["null_percentage"].astype(float)
    bars = ax.barh(
        np.arange(len(missingness)),
        values,
        color=PALETTE["blue_light"],
        edgecolor=PALETTE["blue_dark"],
        linewidth=1.0,
        height=0.66,
    )
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_xlabel("Missing values (%)")
    ax.set_ylabel("")
    upper = max(5.0, min(100.0, math.ceil(max(values.max(), 1.0) * 1.22)))
    ax.set_xlim(0, upper)
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=100, decimals=0))
    for bar, value in zip(bars, values):
        x = min(value + upper * 0.012, upper * 0.965)
        alignment = "left" if value < upper * 0.9 else "right"
        if alignment == "right":
            x = value - upper * 0.012
        if value == 0:
            value_label = "0.0%"
        elif value < 0.1:
            value_label = f"{value:.3f}%"
        else:
            value_label = f"{value:.1f}%"
        ax.text(
            x,
            bar.get_y() + bar.get_height() / 2,
            value_label,
            va="center",
            ha=alignment,
            fontsize=11,
            fontweight="bold" if value == values.max() and value > 0 else "normal",
            color=PALETTE["ink"],
        )
    _footer(
        fig,
        metadata,
        (
            "Effective missingness includes nulls, blank strings and documented sentinel values; records remain retained unless a justified cleaning rule states otherwise."
            if effective
            else "Missing values are retained unless the documented cleaning pipeline has a specific, justified rule."
        ),
    )
    return _save_figure(fig, output_base, title)


def _date_axis(ax: plt.Axes, start: pd.Timestamp, end: pd.Timestamp) -> None:
    span = max(1, int((end - start).days))
    if span <= 45:
        locator: mdates.DateLocator = mdates.WeekdayLocator(interval=1)
        formatter = mdates.DateFormatter("%d %b")
    elif span <= 180:
        locator = mdates.MonthLocator(interval=1)
        formatter = mdates.DateFormatter("%b %Y")
    else:
        interval = 1 if span <= 370 else 2
        locator = mdates.MonthLocator(interval=interval)
        formatter = mdates.DateFormatter("%b %Y")
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.tick_params(axis="x", rotation=0)


def plot_daily_activity(
    daily: pd.DataFrame,
    metadata: Mapping[str, Any],
    output_base: Path,
) -> list[str]:
    start = pd.Timestamp(daily["date"].min())
    end = pd.Timestamp(daily["date"].max())
    title = "Daily recorded arrests and 7-day rolling mean"
    subtitle = (
        f"{start:%d %b %Y} to {end:%d %b %Y}; every calendar date is represented, "
        "and the rolling line begins after seven days"
    )
    fig, ax = _new_figure(title, subtitle)
    ax.plot(
        daily["date"],
        daily["arrest_count"],
        color=PALETTE["daily"],
        linewidth=1.0,
        alpha=0.55,
        label="Daily count",
        zorder=2,
    )
    ax.plot(
        daily["date"],
        daily["rolling_7d_mean"],
        color=PALETTE["blue_dark"],
        linewidth=3.0,
        label="7-day rolling mean",
        zorder=3,
    )
    rolling = daily.dropna(subset=["rolling_7d_mean"])
    if not rolling.empty:
        high = rolling.loc[rolling["rolling_7d_mean"].idxmax()]
        low = rolling.loc[rolling["rolling_7d_mean"].idxmin()]
        annotations = [
            (high, "Highest 7-day mean", (14, 28)),
            (low, "Lowest 7-day mean", (14, -38)),
        ]
        if pd.Timestamp(high["date"]) == pd.Timestamp(low["date"]):
            annotations = annotations[:1]
        for point, label, offset in annotations:
            ax.scatter(
                [point["date"]],
                [point["rolling_7d_mean"]],
                s=45,
                color=PALETTE["blue_dark"],
                edgecolor=PALETTE["white"],
                linewidth=0.8,
                zorder=4,
            )
            ax.annotate(
                f"{label}\n{point['rolling_7d_mean']:.1f} ending {point['date']:%d %b}",
                xy=(point["date"], point["rolling_7d_mean"]),
                xytext=offset,
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=10.5,
                color=PALETTE["ink"],
                arrowprops={
                    "arrowstyle": "-",
                    "color": PALETTE["muted"],
                    "linewidth": 0.9,
                },
                bbox={
                    "boxstyle": "round,pad=0.3",
                    "facecolor": PALETTE["white"],
                    "edgecolor": PALETTE["grid"],
                    "alpha": 0.95,
                },
                zorder=5,
            )
    ax.set_ylabel("Recorded arrests per calendar day")
    ax.set_xlabel("")
    ax.set_ylim(bottom=0)
    ax.set_xlim(start, end)
    _date_axis(ax, start, end)
    ax.legend(
        loc="upper left",
        frameon=False,
        ncol=2,
        bbox_to_anchor=(0, 1.015),
        borderaxespad=0,
    )
    _footer(
        fig,
        metadata,
        "Peaks and troughs are descriptive only; arrests reflect recorded enforcement activity, not crime incidence.",
    )
    return _save_figure(fig, output_base, title)


def plot_monthly_activity(
    monthly: pd.DataFrame,
    metadata: Mapping[str, Any],
    output_base: Path,
) -> list[str]:
    title = "Average recorded arrests per calendar day by month"
    partial_count = int(monthly["is_partial_month"].sum())
    if partial_count:
        subtitle = (
            "Monthly totals divided by calendar days within the analysed date range; "
            "partial boundary months use open, hatched bars and explicit labels"
        )
        caveat = (
            "Calendar-day normalisation improves month comparability; partial-month "
            "estimates remain provisional. Arrests are not crime rates."
        )
    else:
        complete_count = len(monthly)
        subtitle = (
            f"All {_count_word(complete_count)} observed months are complete calendar "
            "months; totals are divided by each month's calendar days"
        )
        caveat = (
            "Calendar-day normalisation improves month comparability; all observed "
            "months are complete. Arrests are not crime rates."
        )
    fig, ax = _new_figure(title, subtitle)
    x = np.arange(len(monthly))
    values = monthly["avg_arrests_per_calendar_day"].to_numpy(dtype=float)
    bars = []
    for position, (_, row) in enumerate(monthly.iterrows()):
        partial = bool(row["is_partial_month"])
        bar = ax.bar(
            position,
            float(row["avg_arrests_per_calendar_day"]),
            width=0.66,
            color=PALETTE["white"] if partial else PALETTE["blue"],
            edgecolor=PALETTE["blue_dark"],
            linewidth=1.3,
            hatch="///" if partial else "",
            zorder=3,
        )[0]
        bars.append(bar)
    ax.set_xticks(x, monthly["month_label"], rotation=0)
    ax.set_ylabel("Average recorded arrests per calendar day")
    ax.set_xlabel("")
    ax.set_ylim(0, max(values.max() * 1.25, 1.0))
    for bar, (_, row) in zip(bars, monthly.iterrows()):
        value = float(row["avg_arrests_per_calendar_day"])
        label = f"{value:.1f}"
        if bool(row["is_partial_month"]):
            label += f"\nPartial · {int(row['calendar_days_in_scope'])} days"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + ax.get_ylim()[1] * 0.018,
            label,
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="bold" if bool(row["is_partial_month"]) else "normal",
            color=PALETTE["ink"],
        )
    _footer(
        fig,
        metadata,
        caveat,
    )
    return _save_figure(fig, output_base, title)


def plot_weekday_activity(
    weekday: pd.DataFrame,
    metadata: Mapping[str, Any],
    output_base: Path,
) -> list[str]:
    title = "Average recorded arrests per weekday occurrence"
    subtitle = (
        "Monday-to-Sunday order; totals divided by the number of each weekday in "
        "the complete analysed calendar range"
    )
    fig, ax = _new_figure(title, subtitle)
    available = weekday[weekday["calendar_occurrences"].gt(0)].copy()
    x = np.arange(len(available))
    values = available["mean_arrests_per_occurrence"].to_numpy(dtype=float)
    bars = ax.bar(
        x,
        values,
        width=0.66,
        color=PALETTE["blue"],
        edgecolor=PALETTE["blue_dark"],
        linewidth=1.0,
        zorder=3,
    )
    ax.set_xticks(x, available["weekday"].str.slice(0, 3))
    ax.set_ylabel("Average recorded arrests per occurrence")
    ax.set_xlabel("")
    ax.set_ylim(0, max(values.max() * 1.20, 1.0))
    for bar, (_, row) in zip(bars, available.iterrows()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ax.get_ylim()[1] * 0.018,
            f"{row['mean_arrests_per_occurrence']:.1f}\n(n={int(row['calendar_occurrences'])} days)",
            ha="center",
            va="bottom",
            fontsize=10.5,
            color=PALETTE["ink"],
        )
    _footer(
        fig,
        metadata,
        "Different weekday frequencies are explicitly normalised; the pattern describes recorded arrests, not underlying crime incidence.",
    )
    return _save_figure(fig, output_base, title)


def plot_severity_composition(
    composition: pd.DataFrame,
    metadata: Mapping[str, Any],
    output_base: Path,
) -> list[str]:
    title = "Monthly composition of recorded arrest severity categories"
    subtitle = (
        "100% stacked shares of all valid-date arrests; grey marks missing or "
        "unrecognised LAW_CAT_CD values, and patterns preserve distinction beyond colour"
    )
    fig, ax = _new_figure(title, subtitle)
    pivot = composition.pivot(
        index="month_label", columns="severity", values="share_of_monthly_arrests_pct"
    ).reindex(columns=SEVERITY_DISPLAY_ORDER)
    month_order = composition[["month_start", "month_label"]].drop_duplicates().sort_values(
        "month_start"
    )
    pivot = pivot.reindex(month_order["month_label"])
    x = np.arange(len(pivot))
    bottoms = np.zeros(len(pivot), dtype=float)
    legend_handles: list[Patch] = []
    for category in SEVERITY_DISPLAY_ORDER:
        values = pivot[category].fillna(0).to_numpy(dtype=float)
        bars = ax.bar(
            x,
            values,
            bottom=bottoms,
            width=0.70,
            color=SEVERITY_COLORS[category],
            edgecolor=PALETTE["ink"],
            linewidth=0.7,
            hatch=SEVERITY_HATCHES[category],
            zorder=3,
        )
        for bar, value, bottom in zip(bars, values, bottoms):
            if value >= 7.0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bottom + value / 2,
                    f"{value:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=9.5,
                    fontweight="bold",
                    color=(
                        PALETTE["white"]
                        if category in {"Felony", "Other or missing"}
                        else PALETTE["ink"]
                    ),
                    zorder=4,
                )
        legend_handles.append(
            Patch(
                facecolor=SEVERITY_COLORS[category],
                edgecolor=PALETTE["ink"],
                hatch=SEVERITY_HATCHES[category],
                label=category,
            )
        )
        bottoms += values

    partial_lookup = (
        composition[["month_label", "is_partial_month"]]
        .drop_duplicates()
        .set_index("month_label")["is_partial_month"]
    )
    partial_count = int(partial_lookup.sum())
    for position, month_label in enumerate(pivot.index):
        if bool(partial_lookup.loc[month_label]):
            ax.add_patch(
                Rectangle(
                    (position - 0.39, -1.5),
                    0.78,
                    103.0,
                    fill=False,
                    linestyle="--",
                    linewidth=1.4,
                    edgecolor=PALETTE["ink"],
                    clip_on=False,
                    zorder=5,
                )
            )
            ax.text(
                position,
                103.0,
                "Partial",
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
                color=PALETTE["ink"],
            )
    ax.set_xticks(x, pivot.index)
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=100, decimals=0))
    ax.set_ylabel("Share of monthly recorded arrests")
    ax.set_xlabel("")
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        frameon=False,
        ncol=4,
        bbox_to_anchor=(0, 1.015),
        borderaxespad=0,
    )
    _footer(
        fig,
        metadata,
        (
            "Grey segments retain missing or unrecognised LAW_CAT_CD values in the "
            "denominator; partial months are outlined and labelled."
            if partial_count
            else "Grey segments retain missing or unrecognised LAW_CAT_CD values in the "
            f"denominator. All {_count_word(len(pivot))} observed months are complete "
            "calendar months."
        ),
    )
    return _save_figure(fig, output_base, title)


def _write_csv(frame: pd.DataFrame, path: Path, *, date_columns: Sequence[str]) -> None:
    output = frame.copy()
    for column in date_columns:
        if column in output.columns:
            output[column] = pd.to_datetime(output[column]).dt.strftime("%Y-%m-%d")
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False, encoding="utf-8", float_format="%.6f", na_rep="")


def _extreme_record(
    frame: pd.DataFrame,
    value_column: str,
    label_column: str,
    mode: str,
    *,
    digits: int = 2,
) -> dict[str, Any] | None:
    eligible = frame.dropna(subset=[value_column])
    if eligible.empty:
        return None
    index = eligible[value_column].idxmax() if mode == "max" else eligible[value_column].idxmin()
    row = eligible.loc[index]
    return {
        "label": _python_scalar(row[label_column]),
        "value": _round(row[value_column], digits),
    }


def build_key_findings(
    *,
    metadata: Mapping[str, Any],
    parse_audit: Mapping[str, int],
    missingness: pd.DataFrame,
    missingness_source: str,
    missingness_metric: str,
    daily: pd.DataFrame,
    monthly: pd.DataFrame,
    weekday: pd.DataFrame,
    composition: pd.DataFrame,
    severity_assessment: Mapping[str, Any],
    generated_files: Mapping[str, Any],
) -> dict[str, Any]:
    """Create exact, human-readable values for slides and independent review."""

    start = pd.Timestamp(daily["date"].min())
    end = pd.Timestamp(daily["date"].max())
    retrieval_value = metadata.get("retrieval_date")
    retrieval_timestamp = pd.to_datetime(retrieval_value, errors="coerce")
    if pd.notna(retrieval_timestamp):
        retrieval_timestamp = pd.Timestamp(retrieval_timestamp)
        if retrieval_timestamp.tzinfo is not None:
            retrieval_timestamp = retrieval_timestamp.tz_localize(None)
        retrieval_timestamp = retrieval_timestamp.normalize()
        retrieval_lag_days: int | None = int((retrieval_timestamp - end).days)
    else:
        retrieval_lag_days = None
    daily_for_json = daily.copy()
    daily_for_json["date_label"] = daily_for_json["date"].dt.strftime("%Y-%m-%d")
    rolling = daily_for_json.dropna(subset=["rolling_7d_mean"])
    monthly_partial = [
        {
            "month": row.month_label,
            "calendar_days_in_scope": int(row.calendar_days_in_scope),
            "calendar_days_in_month": int(row.calendar_days_in_month),
            "reason": row.partial_reason,
        }
        for row in monthly.loc[monthly["is_partial_month"]].itertuples(index=False)
    ]
    weekday_available = weekday[weekday["calendar_occurrences"].gt(0)].copy()
    overall_mean = float(daily["arrest_count"].mean())
    weekday_high = _extreme_record(
        weekday_available,
        "mean_arrests_per_occurrence",
        "weekday",
        "max",
    )
    weekday_low = _extreme_record(
        weekday_available,
        "mean_arrests_per_occurrence",
        "weekday",
        "min",
    )
    if weekday_high and weekday_low and overall_mean:
        weekday_spread_pct = (
            (weekday_high["value"] - weekday_low["value"]) / overall_mean * 100.0
        )
    else:
        weekday_spread_pct = None

    severity_payload = dict(severity_assessment)
    if not composition.empty:
        share_ranges = (
            composition.groupby("severity", sort=False)["share_of_monthly_arrests_pct"]
            .agg(lambda values: float(values.max() - values.min()))
            .reindex(SEVERITY_DISPLAY_ORDER)
            .dropna()
        )
        if not share_ranges.empty:
            category = str(share_ranges.idxmax())
            severity_payload["largest_monthly_share_range"] = {
                "severity": category,
                "percentage_points": round(float(share_ranges.loc[category]), 2),
            }

    missing_rows = [
        {
            "field": str(row.column_name),
            "reported_missing_count": (
                None if pd.isna(row.null_count) else int(row.null_count)
            ),
            "reported_missing_pct": round(float(row.null_percentage), 4),
            "effective_missing_count": (
                None
                if missingness_metric != "effective_missing" or pd.isna(row.null_count)
                else int(row.null_count)
            ),
            "effective_missing_pct": (
                None
                if missingness_metric != "effective_missing"
                else round(float(row.null_percentage), 4)
            ),
        }
        for row in missingness.itertuples(index=False)
    ]
    highest_missing = (
        max(missing_rows, key=lambda row: row["reported_missing_pct"])
        if missing_rows
        else None
    )

    findings: dict[str, Any] = {
        "source": {
            "dataset_name": metadata.get("dataset_name", "NYPD Arrest Data, Year to Date"),
            "dataset_id": metadata.get("dataset_id", "uip8-fykc"),
            "source": metadata.get("source", "NYC Open Data / NYPD"),
            "retrieval_date": metadata.get("retrieval_date"),
            "raw_snapshot_row_count": metadata.get("row_count"),
            "raw_snapshot_column_count": metadata.get("column_count"),
            "dataset_updated_at": metadata.get("dataset_updated_at"),
            "data_freshness": {
                "latest_arrest_date": end.strftime("%Y-%m-%d"),
                "snapshot_retrieval_date": retrieval_value,
                "calendar_days_between_latest_record_and_retrieval": retrieval_lag_days,
            },
        },
        "analysis_window": {
            "start_date": start.strftime("%Y-%m-%d"),
            "end_date": end.strftime("%Y-%m-%d"),
            "calendar_days_in_scope": int(len(daily)),
            "months_in_scope": int(len(monthly)),
            **{key: int(value) for key, value in parse_audit.items()},
        },
        "data_quality": {
            "missingness_source": missingness_source,
            "missingness_metric": missingness_metric,
            "missingness_definition": (
                "Pandas nulls plus blank strings and documented sentinel values."
                if missingness_metric == "effective_missing"
                else "Fallback metric reported by the selected audit source."
            ),
            "missingness_fields_displayed": missing_rows,
            "highest_missingness_among_displayed_fields": highest_missing,
        },
        "temporal_findings": {
            "daily": {
                "definition": "Arrest records per calendar date; absent dates between the minimum and maximum are reindexed to zero.",
                "total_arrests_with_valid_dates": int(daily["arrest_count"].sum()),
                "mean_arrests_per_calendar_day": round(overall_mean, 2),
                "median_arrests_per_calendar_day": round(
                    float(daily["arrest_count"].median()), 2
                ),
                "zero_count_calendar_days": int(daily["is_zero_count_day"].sum()),
                "highest_daily_count": _extreme_record(
                    daily_for_json, "arrest_count", "date_label", "max", digits=0
                ),
                "lowest_daily_count": _extreme_record(
                    daily_for_json, "arrest_count", "date_label", "min", digits=0
                ),
                "highest_7d_rolling_mean": _extreme_record(
                    rolling, "rolling_7d_mean", "date_label", "max"
                ),
                "lowest_7d_rolling_mean": _extreme_record(
                    rolling, "rolling_7d_mean", "date_label", "min"
                ),
            },
            "monthly": {
                "definition": "Monthly arrest count divided by calendar days inside the analysed date range.",
                "highest_average_daily_month": _extreme_record(
                    monthly,
                    "avg_arrests_per_calendar_day",
                    "month_label",
                    "max",
                ),
                "lowest_average_daily_month": _extreme_record(
                    monthly,
                    "avg_arrests_per_calendar_day",
                    "month_label",
                    "min",
                ),
                "complete_month_count": int((~monthly["is_partial_month"]).sum()),
                "partial_months": monthly_partial,
            },
            "weekday": {
                "definition": "Total arrests for each weekday divided by its calendar occurrences in the analysed range.",
                "highest_average_weekday": weekday_high,
                "lowest_average_weekday": weekday_low,
                "high_to_low_spread_as_pct_of_overall_daily_mean": _round(
                    weekday_spread_pct, 2
                ),
                "calendar_occurrences": {
                    str(row.weekday): int(row.calendar_occurrences)
                    for row in weekday_available.itertuples(index=False)
                },
            },
        },
        "severity_analysis": severity_payload,
        "generated_files": dict(generated_files),
        "caveats": [
            "Arrest records reflect recorded police enforcement activity, not the true incidence or rate of crime.",
            "The Year-to-Date snapshot provides a limited observation window and is not evidence of a long-term trend.",
            (
                f"The snapshot was retrieved {retrieval_lag_days} calendar days after "
                f"the latest arrest date; temporal coverage ends on {end:%Y-%m-%d}, "
                "not on the retrieval date."
                if retrieval_lag_days is not None and retrieval_lag_days > 0
                else "Temporal coverage is defined by the actual minimum and maximum arrest dates."
            ),
            "Monthly averages use calendar days in scope; any partial boundary month is explicitly identified.",
            "No causal explanation is assigned to temporal peaks or troughs without external evidence.",
        ],
    }
    return findings


def analyze_temporal(
    processed_csv: Path | str = DEFAULT_PROCESSED_CSV,
    audit_csv: Path | str = DEFAULT_AUDIT_CSV,
    metadata_json: Path | str = DEFAULT_METADATA_JSON,
    outputs_dir: Path | str = DEFAULT_OUTPUTS_DIR,
    figures_dir: Path | str = DEFAULT_FIGURES_DIR,
) -> dict[str, Any]:
    """Run all Part 1 temporal tables, findings, and static figure exports."""

    processed_csv = Path(processed_csv)
    audit_csv = Path(audit_csv)
    metadata_json = Path(metadata_json)
    outputs_dir = Path(outputs_dir)
    figures_dir = Path(figures_dir)
    project_root = _project_root_from_outputs(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(metadata_json)
    processed = load_processed_data(processed_csv)
    prepared, parse_audit = prepare_temporal_data(processed)
    daily = calculate_daily_arrests(prepared)
    monthly = calculate_monthly_arrests(daily)
    weekday = calculate_weekday_arrests(daily)
    composition, severity_assessment = calculate_monthly_severity(prepared, monthly)
    missingness, missingness_source, missingness_metric = load_missingness_summary(
        audit_csv, processed
    )
    if missingness_source != "processed dataset fallback":
        missingness_source = _portable_project_path(
            missingness_source, project_root
        )

    table_paths = {
        "daily_arrests_csv": outputs_dir / "daily_arrests.csv",
        "monthly_arrests_csv": outputs_dir / "monthly_arrests.csv",
        "weekday_arrests_csv": outputs_dir / "weekday_arrests.csv",
        "monthly_severity_composition_csv": outputs_dir
        / "monthly_severity_composition.csv",
    }
    _write_csv(daily, table_paths["daily_arrests_csv"], date_columns=["date"])
    _write_csv(monthly, table_paths["monthly_arrests_csv"], date_columns=["month_start"])
    _write_csv(weekday, table_paths["weekday_arrests_csv"], date_columns=[])
    _write_csv(
        composition,
        table_paths["monthly_severity_composition_csv"],
        date_columns=["month_start"],
    )

    figure_paths: dict[str, list[str] | None] = {}
    with mpl.rc_context(RC_PARAMS):
        figure_paths["missingness_overview"] = plot_missingness(
            missingness,
            metadata,
            figures_dir / "missingness_overview",
            metric=missingness_metric,
        )
        figure_paths["daily_arrests_rolling"] = plot_daily_activity(
            daily, metadata, figures_dir / "daily_arrests_rolling"
        )
        figure_paths["monthly_average_daily_arrests"] = plot_monthly_activity(
            monthly, metadata, figures_dir / "monthly_average_daily_arrests"
        )
        figure_paths["weekday_average_arrests"] = plot_weekday_activity(
            weekday, metadata, figures_dir / "weekday_average_arrests"
        )
        if severity_assessment["eligible"]:
            figure_paths["monthly_severity_composition"] = plot_severity_composition(
                composition, metadata, figures_dir / "monthly_severity_composition"
            )
            severity_assessment["figure_generated"] = True
        else:
            figure_paths["monthly_severity_composition"] = None
            for suffix in (".png", ".svg"):
                stale_path = figures_dir / f"monthly_severity_composition{suffix}"
                if stale_path.exists():
                    stale_path.unlink()
            LOGGER.warning(
                "Severity figure omitted: %s", severity_assessment.get("reason")
            )

    portable_figure_paths = {
        key: (
            None
            if paths is None
            else [_portable_project_path(path, project_root) for path in paths]
        )
        for key, paths in figure_paths.items()
    }
    generated_files = {
        "tables": {
            key: _portable_project_path(path, project_root)
            for key, path in table_paths.items()
        },
        "figures": portable_figure_paths,
    }
    findings = build_key_findings(
        metadata=metadata,
        parse_audit=parse_audit,
        missingness=missingness,
        missingness_source=missingness_source,
        missingness_metric=missingness_metric,
        daily=daily,
        monthly=monthly,
        weekday=weekday,
        composition=composition,
        severity_assessment=severity_assessment,
        generated_files=generated_files,
    )
    findings_path = outputs_dir / "key_findings.json"
    findings["generated_files"]["key_findings_json"] = _portable_project_path(
        findings_path, project_root
    )
    with findings_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(findings, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")

    LOGGER.info(
        "Temporal analysis complete: %s valid-date arrests, %s calendar days, %s months.",
        parse_audit["valid_date_rows"],
        len(daily),
        len(monthly),
    )
    return findings


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate CA6002 Part 1 temporal aggregates, findings and static figures "
            "from the shared processed NYPD arrest snapshot."
        )
    )
    parser.add_argument("--processed-csv", type=Path, default=DEFAULT_PROCESSED_CSV)
    parser.add_argument(
        "--audit-csv",
        "--missingness-csv",
        dest="audit_csv",
        type=Path,
        default=DEFAULT_AUDIT_CSV,
    )
    parser.add_argument("--metadata-json", type=Path, default=DEFAULT_METADATA_JSON)
    parser.add_argument("--outputs-dir", type=Path, default=DEFAULT_OUTPUTS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(message)s",
    )
    findings = analyze_temporal(
        processed_csv=args.processed_csv,
        audit_csv=args.audit_csv,
        metadata_json=args.metadata_json,
        outputs_dir=args.outputs_dir,
        figures_dir=args.figures_dir,
    )
    window = findings["analysis_window"]
    print(
        "Temporal analysis complete: "
        f"{window['valid_date_rows']:,} valid-date records, "
        f"{window['start_date']} to {window['end_date']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
