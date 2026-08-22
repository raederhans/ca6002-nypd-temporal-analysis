"""Audit and conservatively clean a frozen NYPD Arrests YTD snapshot.

The module deliberately keeps acquisition and cleaning separate: this code never
writes to the raw snapshot.  It can be imported by the Part 1 notebook or run as
a command-line program from any working directory.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from datetime import date, datetime, timezone
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED_RELATIVE_PATH = Path("data/processed/nypd_arrests_clean.csv")
DEFAULT_OUTPUT_RELATIVE_DIR = Path("outputs/part1")

EXPECTED_COLUMNS = (
    "ARREST_KEY",
    "ARREST_DATE",
    "PD_CD",
    "PD_DESC",
    "KY_CD",
    "OFNS_DESC",
    "LAW_CODE",
    "LAW_CAT_CD",
    "ARREST_BORO",
    "ARREST_PRECINCT",
    "JURISDICTION_CODE",
    "AGE_GROUP",
    "PERP_SEX",
    "PERP_RACE",
    "X_COORD_CD",
    "Y_COORD_CD",
    "LATITUDE",
    "LONGITUDE",
    "GEOCODED_COLUMN",
)

NUMERIC_CODED_CATEGORICAL_COLUMNS = (
    "PD_CD",
    "KY_CD",
    "ARREST_PRECINCT",
    "JURISDICTION_CODE",
)

TEXTUAL_IDENTIFIER_AND_CODE_COLUMNS = (
    "ARREST_KEY",
    *NUMERIC_CODED_CATEGORICAL_COLUMNS,
)

CONTINUOUS_NUMERIC_COLUMNS = (
    "LATITUDE",
    "LONGITUDE",
    "X_COORD_CD",
    "Y_COORD_CD",
)

CATEGORICAL_COLUMNS = (
    "ARREST_BORO",
    "LAW_CAT_CD",
    "AGE_GROUP",
    "PERP_SEX",
    "PERP_RACE",
)

CATEGORY_EXPECTATIONS: dict[str, set[str]] = {
    "ARREST_BORO": {"B", "K", "M", "Q", "S"},
    # I (infraction) is uncommon but is a documented law-category code.
    "LAW_CAT_CD": {"F", "M", "V", "I"},
    "AGE_GROUP": {"<18", "18-24", "25-44", "45-64", "65+"},
    # U is retained as a possible source-supplied unknown code.
    "PERP_SEX": {"F", "M", "U"},
    "PERP_RACE": {
        "AMERICAN INDIAN/ALASKAN NATIVE",
        "ASIAN / PACIFIC ISLANDER",
        "BLACK",
        "BLACK HISPANIC",
        "UNKNOWN",
        "WHITE",
        "WHITE HISPANIC",
    },
}

LAW_CATEGORY_CORE_SEVERITY = {"F", "M", "V"}
LAW_CATEGORY_KNOWN_NON_CORE = {"I"}
# Pandas already parses its standard NA spellings.  The frozen Socrata export
# additionally contains this literal token, verified in the source snapshot.
SOURCE_MISSING_SENTINELS = {"(NULL)"}

# Deliberately conservative NYC screening extent.  This is not a borough-level
# spatial inclusion rule; records outside it are reported and retained for the
# spatial team.
NYC_BOUNDS = {
    "latitude_min": 40.45,
    "latitude_max": 40.95,
    "longitude_min": -74.30,
    "longitude_max": -73.65,
}


def _normalise_column_name(value: object) -> str:
    """Return a stable uppercase snake-case representation of a column name."""

    name = str(value).strip().upper()
    name = re.sub(r"[^A-Z0-9]+", "_", name)
    return re.sub(r"_+", "_", name).strip("_")


def _project_path(path: str | Path, project_root: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path(project_root).expanduser() / candidate
    return candidate.resolve()


def _snapshot_date_from_path(path: Path) -> date | None:
    match = re.fullmatch(
        r"nypd_arrests_ytd_(\d{4}-\d{2}-\d{2})\.csv", path.name, re.IGNORECASE
    )
    if not match:
        return None
    try:
        return date.fromisoformat(match.group(1))
    except ValueError:
        return None


def resolve_raw_path(
    raw_path: str | Path | None = None,
    project_root: str | Path = PROJECT_ROOT,
) -> Path:
    """Resolve an explicit snapshot or discover the latest dated raw snapshot.

    Discovery is intentionally restricted to the assignment's frozen-snapshot
    naming convention so an unrelated CSV can never be selected silently.
    """

    root = Path(project_root).expanduser().resolve()
    if raw_path is not None:
        resolved = _project_path(raw_path, root)
        if not resolved.is_file():
            raise FileNotFoundError(f"Raw snapshot does not exist: {resolved}")
        return resolved

    raw_dir = root / "data" / "raw"
    candidates = [
        path
        for path in raw_dir.glob("nypd_arrests_ytd_*.csv")
        if path.is_file() and _snapshot_date_from_path(path) is not None
    ]
    if not candidates:
        raise FileNotFoundError(
            "No frozen raw snapshot found. Expected "
            f"{raw_dir / 'nypd_arrests_ytd_YYYY-MM-DD.csv'}"
        )
    return max(candidates, key=lambda path: (_snapshot_date_from_path(path), path.name))


def load_raw_data(
    raw_path: str | Path | None = None,
    project_root: str | Path = PROJECT_ROOT,
    preserve_code_text: bool = False,
) -> pd.DataFrame:
    """Load a frozen raw CSV without modifying or rewriting it.

    The default intentionally retains pandas' normal type inference for schema
    auditing.  ``preserve_code_text=True`` reads identifier/numeric-code fields
    with pandas' nullable string dtype so their original CSV spelling survives
    the raw-to-processed path.
    """

    resolved = resolve_raw_path(raw_path, project_root)
    dtype: dict[str, str] | None = None
    if preserve_code_text:
        source_columns = pd.read_csv(resolved, nrows=0).columns
        dtype = {
            str(column): "string"
            for column in source_columns
            if _normalise_column_name(column)
            in TEXTUAL_IDENTIFIER_AND_CODE_COLUMNS
        }
    frame = pd.read_csv(resolved, low_memory=False, dtype=dtype)
    if frame.empty:
        raise ValueError(f"Raw snapshot is empty: {resolved}")
    frame.attrs["source_path"] = str(resolved)
    frame.attrs["code_text_preserved_on_load"] = preserve_code_text
    return frame


def standardize_column_names(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with uppercase snake-case columns and collision checks."""

    original_columns = [str(column) for column in frame.columns]
    normalised_columns = [_normalise_column_name(column) for column in frame.columns]
    duplicates = sorted(
        {name for name in normalised_columns if normalised_columns.count(name) > 1}
    )
    if duplicates:
        raise ValueError(
            "Column-name standardisation would create collisions: "
            + ", ".join(duplicates)
        )
    result = frame.copy()
    result.columns = normalised_columns
    result.attrs.update(frame.attrs)
    result.attrs["original_column_names"] = original_columns
    result.attrs["column_name_mapping"] = dict(
        zip(original_columns, normalised_columns, strict=True)
    )
    return result


def _percentage(count: int, denominator: int) -> float:
    return round((100.0 * count / denominator), 4) if denominator else 0.0


def _blank_mask(series: pd.Series) -> pd.Series:
    """Identify zero-length/whitespace-only strings without coercing other values."""

    if not (
        pd.api.types.is_object_dtype(series.dtype)
        or pd.api.types.is_string_dtype(series.dtype)
    ):
        return pd.Series(False, index=series.index, dtype=bool)
    return series.map(lambda value: isinstance(value, str) and not value.strip())


def _nonblank_mask(series: pd.Series) -> pd.Series:
    return series.notna() & ~_blank_mask(series)


def _missing_sentinel_mask(series: pd.Series) -> pd.Series:
    """Identify explicit source strings that encode missingness."""

    if not (
        pd.api.types.is_object_dtype(series.dtype)
        or pd.api.types.is_string_dtype(series.dtype)
    ):
        return pd.Series(False, index=series.index, dtype=bool)
    normalised = series.astype("string").str.strip().str.upper()
    return series.notna() & normalised.isin(SOURCE_MISSING_SENTINELS)


def build_schema_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the required column-level schema and null profile."""

    rows = len(frame)
    records: list[dict[str, object]] = []
    for column in frame.columns:
        null_count = int(frame[column].isna().sum())
        records.append(
            {
                "column_name": str(column),
                "inferred_dtype": str(frame[column].dtype),
                "non_null_count": int(frame[column].notna().sum()),
                "null_count": null_count,
                "null_percentage": _percentage(null_count, rows),
                "unique_count": int(frame[column].nunique(dropna=True)),
            }
        )
    return pd.DataFrame.from_records(
        records,
        columns=(
            "column_name",
            "inferred_dtype",
            "non_null_count",
            "null_count",
            "null_percentage",
            "unique_count",
        ),
    )


def build_missingness_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Profile both parser-level nulls and source strings that are blank."""

    rows = len(frame)
    records: list[dict[str, object]] = []
    for column in frame.columns:
        missing_count = int(frame[column].isna().sum())
        blank_count = int(_blank_mask(frame[column]).sum())
        sentinel_count = int(_missing_sentinel_mask(frame[column]).sum())
        effective_missing = missing_count + blank_count + sentinel_count
        records.append(
            {
                "column_name": str(column),
                "missing_count": missing_count,
                "missing_percentage": _percentage(missing_count, rows),
                "blank_string_count": blank_count,
                "sentinel_missing_count": sentinel_count,
                "effective_missing_count": effective_missing,
                "effective_missing_percentage": _percentage(effective_missing, rows),
            }
        )
    summary = pd.DataFrame.from_records(records)
    if summary.empty:
        return pd.DataFrame(
            columns=(
                "column_name",
                "missing_count",
                "missing_percentage",
                "blank_string_count",
                "sentinel_missing_count",
                "effective_missing_count",
                "effective_missing_percentage",
            )
        )
    return summary.sort_values(
        ["effective_missing_percentage", "column_name"],
        ascending=[False, True],
        kind="stable",
        ignore_index=True,
    )


def _parse_arrest_dates(series: pd.Series) -> pd.Series:
    """Parse common Socrata/CSV date encodings without assuming one format."""

    values = series.astype("string").str.strip().replace("", pd.NA)
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    formats = (
        "%m/%d/%Y",
        "%Y-%m-%d",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
    )
    for date_format in formats:
        remaining = values.notna() & parsed.isna()
        if not remaining.any():
            break
        parsed.loc[remaining] = pd.to_datetime(
            values.loc[remaining], format=date_format, errors="coerce"
        )
    remaining = values.notna() & parsed.isna()
    if remaining.any():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            parsed.loc[remaining] = pd.to_datetime(
                values.loc[remaining], errors="coerce"
            )
    return parsed


def _coerce_snapshot_date(value: date | datetime | str | pd.Timestamp | None) -> date:
    if value is None:
        return date.today()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _build_date_validation(
    frame: pd.DataFrame, snapshot_date: date | datetime | str | pd.Timestamp | None
) -> dict[str, Any]:
    if "ARREST_DATE" not in frame.columns:
        return {
            "available": False,
            "reason": "ARREST_DATE is absent from the source schema.",
        }

    as_of = _coerce_snapshot_date(snapshot_date)
    raw = frame["ARREST_DATE"]
    provided = _nonblank_mask(raw)
    parsed = _parse_arrest_dates(raw)
    failures = provided & parsed.isna()
    valid = parsed.dropna()
    future = parsed.dt.date > as_of
    abnormal_year = parsed.notna() & parsed.dt.year.ne(as_of.year)

    if valid.empty:
        minimum = maximum = None
        coverage_days = unique_days = missing_calendar_days = 0
        year_counts: dict[str, int] = {}
        snapshot_to_latest_gap = None
        latest_month_calendar_complete = None
    else:
        normalised = valid.dt.normalize()
        minimum_timestamp = normalised.min()
        maximum_timestamp = normalised.max()
        minimum = minimum_timestamp.date().isoformat()
        maximum = maximum_timestamp.date().isoformat()
        coverage_days = int((maximum_timestamp - minimum_timestamp).days + 1)
        unique_days = int(normalised.nunique())
        missing_calendar_days = coverage_days - unique_days
        snapshot_to_latest_gap = (as_of - maximum_timestamp.date()).days
        latest_month_calendar_complete = bool(maximum_timestamp.is_month_end)
        year_counts = {
            str(int(year)): int(count)
            for year, count in valid.dt.year.value_counts().sort_index().items()
        }

    report = {
        "available": True,
        "snapshot_date": as_of.isoformat(),
        "expected_ytd_year": as_of.year,
        "source_missing_or_blank_count": int((~provided).sum()),
        "parse_failure_count": int(failures.sum()),
        "parse_failure_percentage_of_provided": _percentage(
            int(failures.sum()), int(provided.sum())
        ),
        "valid_date_count": int(parsed.notna().sum()),
        "min_date": minimum,
        "max_date": maximum,
        "calendar_coverage_days": coverage_days,
        "observed_unique_dates": unique_days,
        "unobserved_dates_within_range": missing_calendar_days,
        "snapshot_to_latest_arrest_gap_days": snapshot_to_latest_gap,
        "latest_observed_month_calendar_complete": latest_month_calendar_complete,
        "future_date_count": int(future.fillna(False).sum()),
        "abnormal_year_count": int(abnormal_year.fillna(False).sum()),
        "year_counts": year_counts,
        "ytd_caveat": (
            "Coverage is assessed only over the observed snapshot; a YTD dataset "
            "must not be treated as a complete twelve-month period."
        ),
    }
    return report


def _build_quality_findings(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Translate profiles into concise, risk-linked and evidence-backed findings."""

    findings: list[dict[str, Any]] = []
    schema = report["schema_validation"]
    if not schema["matches_expected_19_column_schema"]:
        findings.append(
            {
                "id": "schema-drift",
                "severity": "high",
                "confidence": "high",
                "what_failed": "The source does not match the expected 19-column schema.",
                "evidence": {
                    "missing_expected_columns": schema["missing_expected_columns"],
                    "unexpected_columns": schema["unexpected_columns"],
                },
                "why_it_matters": "Downstream code may omit required fields or read the wrong grain.",
                "likely_cause": "Upstream schema change or an incorrect input file; not established by this audit.",
                "suggested_remediation": "Resolve the source/schema difference before downstream analysis.",
            }
        )

    duplicate_validation = report["duplicates"]
    if duplicate_validation["duplicated_arrest_key_count"]:
        findings.append(
            {
                "id": "duplicate-arrest-keys",
                "severity": "high",
                "confidence": "high",
                "what_failed": "ARREST_KEY is not unique among nonblank keys.",
                "evidence": {
                    "duplicated_key_count": duplicate_validation[
                        "duplicated_arrest_key_count"
                    ],
                    "affected_rows": duplicate_validation[
                        "rows_with_duplicated_arrest_key"
                    ],
                },
                "why_it_matters": "Unresolved repeated identifiers can double-count event-level analysis.",
                "likely_cause": "Could be exact duplication or conflicting record versions; inspect the evidence file.",
                "suggested_remediation": "Remove only exact copies and retain differing versions until adjudicated.",
            }
        )

    dates = report["date_validation"]
    if dates.get("available") and (
        dates["parse_failure_count"]
        or dates["future_date_count"]
        or dates["abnormal_year_count"]
    ):
        findings.append(
            {
                "id": "date-validity",
                "severity": "high",
                "confidence": "high",
                "what_failed": "One or more arrest dates failed YTD validity checks.",
                "evidence": {
                    "parse_failures": dates["parse_failure_count"],
                    "future_dates": dates["future_date_count"],
                    "abnormal_year_rows": dates["abnormal_year_count"],
                },
                "why_it_matters": "Invalid dates can distort temporal aggregation and derived features.",
                "likely_cause": "Source entry or encoding issue; not established by this audit.",
                "suggested_remediation": "Retain raw rows, exclude invalid dates only from date-dependent calculations, and report coverage.",
            }
        )

    if dates.get("available") and dates.get("max_date"):
        findings.append(
            {
                "id": "ytd-temporal-coverage",
                "severity": "medium",
                "confidence": "high",
                "what_failed": "The snapshot is not a complete twelve-month observation period.",
                "evidence": {
                    "min_date": dates["min_date"],
                    "max_date": dates["max_date"],
                    "snapshot_date": dates["snapshot_date"],
                    "snapshot_to_latest_arrest_gap_days": dates[
                        "snapshot_to_latest_arrest_gap_days"
                    ],
                    "latest_observed_month_calendar_complete": dates[
                        "latest_observed_month_calendar_complete"
                    ],
                },
                "why_it_matters": "Observed-period patterns cannot be presented as annual or long-term trends.",
                "likely_cause": "The source is explicitly Year-to-Date; publication cadence may also affect recency, but no causal explanation was verified.",
                "suggested_remediation": "Label the exact observed range and assess each month for calendar completeness before comparison.",
            }
        )

    law = report["categorical_validation"]["columns"].get("LAW_CAT_CD")
    if law:
        special_count = (
            int(law["missing_count"])
            + sum(law.get("source_missing_sentinel_counts", {}).values())
            + sum(law.get("known_non_core_value_counts", {}).values())
            + sum(law.get("unrecognised_non_core_value_counts", {}).values())
        )
        if special_count:
            findings.append(
                {
                    "id": "law-category-coverage",
                    "severity": "low",
                    "confidence": "high",
                    "what_failed": "LAW_CAT_CD is not exclusively populated by the core F/M/V severity categories.",
                    "evidence": {
                        "core_fmv_count": law["core_severity_count"],
                        "core_fmv_coverage_percentage": law[
                            "core_severity_coverage_percentage_of_all_rows"
                        ],
                        "true_null_count": law["missing_count"],
                        "source_missing_sentinel_counts": law[
                            "source_missing_sentinel_counts"
                        ],
                        "known_non_core_value_counts": law[
                            "known_non_core_value_counts"
                        ],
                        "unrecognised_non_core_value_counts": law[
                            "unrecognised_non_core_value_counts"
                        ],
                    },
                    "why_it_matters": "Severity-composition denominators will be inconsistent if special values are silently mixed into F/M/V.",
                    "likely_cause": "The source contains nulls, an explicit missing sentinel, and non-core codes; their upstream causes were not established.",
                    "suggested_remediation": "For severity composition, explicitly filter to F/M/V and report retained coverage; preserve all source values in the baseline dataset.",
                }
            )

    missingness = report["missingness"]
    if missingness["total_effective_missing_cells"]:
        findings.append(
            {
                "id": "field-missingness",
                "severity": "low",
                "confidence": "high",
                "what_failed": "Some source fields contain nulls or explicit missing sentinels.",
                "evidence": {
                    "total_effective_missing_cells": missingness[
                        "total_effective_missing_cells"
                    ],
                    "highest_effective_missingness": missingness[
                        "highest_effective_missingness"
                    ],
                },
                "why_it_matters": "Usable sample sizes can vary by field and should be stated for each analysis.",
                "likely_cause": "Upstream completeness varies by field; no cause was established by this snapshot-only audit.",
                "suggested_remediation": "Do not impute without a defensible rule; report analysis-specific denominators.",
            }
        )

    coordinates = report["coordinate_validation"]
    if coordinates.get("available") and (
        coordinates["rows_with_any_source_coordinate_missing_or_blank"]
        or coordinates["latitude"]["non_numeric_count"]
        or coordinates["longitude"]["non_numeric_count"]
        or coordinates["latitude"]["zero_count"]
        or coordinates["longitude"]["zero_count"]
        or coordinates["latitude"]["impossible_global_range_count"]
        or coordinates["longitude"]["impossible_global_range_count"]
        or coordinates["outside_approximate_nyc_bounds_count"]
    ):
        findings.append(
            {
                "id": "coordinate-validity",
                "severity": "medium",
                "confidence": "high",
                "what_failed": "One or more coordinate records failed screening checks.",
                "evidence": coordinates,
                "why_it_matters": "Invalid coordinates can misplace records in downstream spatial analysis.",
                "likely_cause": "Source geocoding or encoding issue; not established by this audit.",
                "suggested_remediation": "Retain the baseline rows and apply an explicit spatial-use filter downstream.",
            }
        )

    return findings


def _build_categorical_validation(frame: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {
        "checked_columns": [
            column for column in CATEGORICAL_COLUMNS if column in frame.columns
        ],
        "absent_requested_columns": [
            column for column in CATEGORICAL_COLUMNS if column not in frame.columns
        ],
        "columns": {},
        "treatment": (
            "Values are trimmed for validation. Unexpected values are reported "
            "and retained; no record is dropped or imputed."
        ),
    }
    for column in CATEGORICAL_COLUMNS:
        if column not in frame.columns:
            continue
        source = frame[column]
        trimmed = source.astype("string").str.strip()
        missing_count = int(source.isna().sum())
        blank_count = int((trimmed.eq("") & source.notna()).sum())
        observed = trimmed[source.notna() & trimmed.ne("")]
        counts = observed.value_counts(dropna=True)
        allowed = CATEGORY_EXPECTATIONS.get(column)
        normalised_observed = {str(value).upper() for value in counts.index}
        unexpected = sorted(normalised_observed - allowed) if allowed else []
        result["columns"][column] = {
            "missing_count": missing_count,
            "missing_percentage": _percentage(missing_count, len(frame)),
            "blank_string_count": blank_count,
            "distinct_non_missing_count": int(observed.nunique(dropna=True)),
            "unique_values": sorted(str(value) for value in counts.index),
            "value_counts": {
                str(value): int(count) for value, count in counts.items()
            },
            "expected_values": sorted(allowed) if allowed else None,
            "unexpected_values_after_uppercase_comparison": unexpected,
        }
        if column == "LAW_CAT_CD":
            upper = observed.str.upper()
            sentinel_mask = upper.isin(SOURCE_MISSING_SENTINELS)
            core_mask = upper.isin(LAW_CATEGORY_CORE_SEVERITY)
            known_non_core_mask = upper.isin(LAW_CATEGORY_KNOWN_NON_CORE)
            unrecognised_mask = ~(
                sentinel_mask | core_mask | known_non_core_mask
            )
            result["columns"][column].update(
                {
                    "core_severity_values": sorted(LAW_CATEGORY_CORE_SEVERITY),
                    "core_severity_count": int(core_mask.sum()),
                    "core_severity_coverage_percentage_of_all_rows": _percentage(
                        int(core_mask.sum()), len(frame)
                    ),
                    "known_non_core_values": sorted(LAW_CATEGORY_KNOWN_NON_CORE),
                    "known_non_core_value_counts": {
                        str(value): int(count)
                        for value, count in observed.loc[
                            known_non_core_mask
                        ].value_counts().items()
                    },
                    "source_missing_sentinel_counts": {
                        str(value): int(count)
                        for value, count in observed.loc[
                            sentinel_mask
                        ].value_counts().items()
                    },
                    "unrecognised_non_core_value_counts": {
                        str(value): int(count)
                        for value, count in observed.loc[unrecognised_mask]
                        .value_counts()
                        .items()
                    },
                    "severity_analysis_rule": (
                        "Use only F/M/V for severity composition after reporting "
                        "coverage. I, unrecognised codes, true missing values and "
                        "source missing sentinels remain in the processed dataset "
                        "and must not be silently recoded."
                    ),
                }
            )
    return result


def _numeric_coordinate_profile(
    series: pd.Series, lower: float, upper: float
) -> tuple[pd.Series, dict[str, Any]]:
    provided = _nonblank_mask(series)
    numeric = pd.to_numeric(series, errors="coerce")
    non_numeric = provided & numeric.isna()
    zero = numeric.eq(0)
    impossible = numeric.notna() & ~numeric.between(lower, upper, inclusive="both")
    return numeric, {
        "source_missing_or_blank_count": int((~provided).sum()),
        "non_numeric_count": int(non_numeric.sum()),
        "numeric_count": int(numeric.notna().sum()),
        "zero_count": int(zero.fillna(False).sum()),
        "impossible_global_range_count": int(impossible.fillna(False).sum()),
        "minimum_numeric_value": (
            float(numeric.min()) if numeric.notna().any() else None
        ),
        "maximum_numeric_value": (
            float(numeric.max()) if numeric.notna().any() else None
        ),
    }


def _build_coordinate_validation(frame: pd.DataFrame) -> dict[str, Any]:
    missing_columns = [
        column for column in ("LATITUDE", "LONGITUDE") if column not in frame.columns
    ]
    if missing_columns:
        return {
            "available": False,
            "missing_columns": missing_columns,
            "reason": "Latitude/longitude pair is incomplete in the source schema.",
        }

    latitude, latitude_profile = _numeric_coordinate_profile(
        frame["LATITUDE"], -90.0, 90.0
    )
    longitude, longitude_profile = _numeric_coordinate_profile(
        frame["LONGITUDE"], -180.0, 180.0
    )
    globally_valid_pair = (
        latitude.between(-90.0, 90.0, inclusive="both")
        & longitude.between(-180.0, 180.0, inclusive="both")
        & latitude.ne(0)
        & longitude.ne(0)
    )
    within_nyc = (
        latitude.between(
            NYC_BOUNDS["latitude_min"],
            NYC_BOUNDS["latitude_max"],
            inclusive="both",
        )
        & longitude.between(
            NYC_BOUNDS["longitude_min"],
            NYC_BOUNDS["longitude_max"],
            inclusive="both",
        )
    )
    outside_nyc = globally_valid_pair & ~within_nyc
    numeric_pair = latitude.notna() & longitude.notna()
    incomplete_pair = latitude.notna() ^ longitude.notna()
    source_latitude_provided = _nonblank_mask(frame["LATITUDE"])
    source_longitude_provided = _nonblank_mask(frame["LONGITUDE"])
    source_pair_missing = ~(source_latitude_provided & source_longitude_provided)

    latitude_profile["outside_approximate_nyc_range_count"] = int(
        (
            latitude.notna()
            & latitude.ne(0)
            & latitude.between(-90.0, 90.0, inclusive="both")
            & ~latitude.between(
                NYC_BOUNDS["latitude_min"],
                NYC_BOUNDS["latitude_max"],
                inclusive="both",
            )
        ).sum()
    )
    longitude_profile["outside_approximate_nyc_range_count"] = int(
        (
            longitude.notna()
            & longitude.ne(0)
            & longitude.between(-180.0, 180.0, inclusive="both")
            & ~longitude.between(
                NYC_BOUNDS["longitude_min"],
                NYC_BOUNDS["longitude_max"],
                inclusive="both",
            )
        ).sum()
    )
    return {
        "available": True,
        "approximate_nyc_bounds": NYC_BOUNDS,
        "latitude": latitude_profile,
        "longitude": longitude_profile,
        "numeric_coordinate_pair_count": int(numeric_pair.sum()),
        "rows_with_any_source_coordinate_missing_or_blank": int(
            source_pair_missing.sum()
        ),
        "incomplete_numeric_pair_count": int(incomplete_pair.sum()),
        "globally_valid_nonzero_pair_count": int(globally_valid_pair.sum()),
        "within_approximate_nyc_bounds_count": int(within_nyc.sum()),
        "outside_approximate_nyc_bounds_count": int(outside_nyc.sum()),
        "treatment": (
            "Coordinate anomalies are reported, not deleted. Approximate NYC "
            "bounds are a screening check and not a geographic adjudication."
        ),
    }


def build_duplicate_key_evidence(frame: pd.DataFrame) -> pd.DataFrame:
    """Return all rows belonging to a duplicated, nonblank ARREST_KEY."""

    working = standardize_column_names(frame)
    if "ARREST_KEY" not in working.columns:
        return pd.DataFrame()

    keys = working["ARREST_KEY"].astype("string").str.strip().replace("", pd.NA)
    duplicated_key = keys.notna() & keys.duplicated(keep=False)
    if not duplicated_key.any():
        return pd.DataFrame()

    evidence = working.loc[duplicated_key].copy()
    evidence_keys = keys.loc[duplicated_key]
    group_sizes = evidence_keys.value_counts(dropna=True)
    signatures = pd.util.hash_pandas_object(evidence, index=False)
    signature_counts = signatures.groupby(evidence_keys, sort=False).nunique()

    evidence.insert(0, "AUDIT_ARREST_KEY", evidence_keys)
    evidence.insert(
        1,
        "AUDIT_KEY_OCCURRENCE_COUNT",
        evidence_keys.map(group_sizes).astype("Int64"),
    )
    evidence.insert(
        2,
        "AUDIT_ALL_ROWS_IDENTICAL_FOR_KEY",
        evidence_keys.map(signature_counts.eq(1)).astype("boolean"),
    )
    evidence.insert(
        3,
        "AUDIT_ROW_PART_OF_EXACT_DUPLICATE",
        working.duplicated(keep=False).loc[duplicated_key].astype("boolean"),
    )
    return evidence.sort_values(
        ["AUDIT_ARREST_KEY"], kind="stable", ignore_index=True
    )


def _build_duplicate_validation(frame: pd.DataFrame) -> dict[str, Any]:
    exact_beyond_first = int(frame.duplicated(keep="first").sum())
    exact_involved = int(frame.duplicated(keep=False).sum())
    evidence = build_duplicate_key_evidence(frame)
    if evidence.empty:
        duplicate_key_rows = duplicate_key_count = identical_key_groups = 0
    else:
        duplicate_key_rows = len(evidence)
        duplicate_key_count = int(evidence["AUDIT_ARREST_KEY"].nunique())
        identical_key_groups = int(
            evidence.loc[
                evidence["AUDIT_ALL_ROWS_IDENTICAL_FOR_KEY"].fillna(False),
                "AUDIT_ARREST_KEY",
            ].nunique()
        )
    nonblank_key_count = 0
    missing_key_count = None
    if "ARREST_KEY" in frame.columns:
        keys = frame["ARREST_KEY"].astype("string").str.strip().replace("", pd.NA)
        nonblank_key_count = int(keys.notna().sum())
        missing_key_count = int(keys.isna().sum())

    return {
        "exact_duplicate_rows_beyond_first": exact_beyond_first,
        "exact_duplicate_rows_involved": exact_involved,
        "exact_duplicate_rate_percentage": _percentage(exact_beyond_first, len(frame)),
        "arrest_key_available": "ARREST_KEY" in frame.columns,
        "nonblank_arrest_key_count": nonblank_key_count,
        "missing_or_blank_arrest_key_count": missing_key_count,
        "duplicated_arrest_key_count": duplicate_key_count,
        "rows_with_duplicated_arrest_key": duplicate_key_rows,
        "duplicated_key_groups_with_identical_rows": identical_key_groups,
        "interpretation": (
            "Exact row copies can be removed without losing distinct source "
            "information. Repeated ARREST_KEY values with any field difference "
            "require investigation and are retained."
        ),
    }


def audit_dataset(
    frame: pd.DataFrame,
    snapshot_date: date | datetime | str | pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Run schema, completeness, uniqueness, date, category and coordinate checks."""

    original_columns = [str(column) for column in frame.columns]
    working = standardize_column_names(frame)
    source_path = working.attrs.get("source_path")
    normalised_columns = list(working.columns)
    expected = set(EXPECTED_COLUMNS)
    actual = set(normalised_columns)
    missingness = build_missingness_summary(working)
    category_validation = _build_categorical_validation(working)
    coordinate_validation = _build_coordinate_validation(working)
    date_validation = _build_date_validation(working, snapshot_date)

    top_missing = missingness.loc[
        missingness["effective_missing_count"].gt(0)
    ].head(10)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_file": source_path,
        "snapshot_date": _coerce_snapshot_date(snapshot_date).isoformat(),
        "dataset": {
            "row_count": int(len(working)),
            "column_count": int(len(working.columns)),
            "original_column_names": original_columns,
            "standardized_column_names": normalised_columns,
        },
        "schema_validation": {
            "expected_19_column_schema": list(EXPECTED_COLUMNS),
            "matches_expected_19_column_schema": (
                len(normalised_columns) == len(EXPECTED_COLUMNS) and actual == expected
            ),
            "columns_in_expected_order": normalised_columns == list(EXPECTED_COLUMNS),
            "missing_expected_columns": sorted(expected - actual),
            "unexpected_columns": sorted(actual - expected),
            "column_name_mapping": working.attrs.get("column_name_mapping", {}),
        },
        "missingness": {
            "columns_with_parser_nulls": int(
                missingness["missing_count"].gt(0).sum()
            ),
            "columns_with_effective_missingness": int(
                missingness["effective_missing_count"].gt(0).sum()
            ),
            "total_parser_null_cells": int(missingness["missing_count"].sum()),
            "total_blank_string_cells": int(
                missingness["blank_string_count"].sum()
            ),
            "total_sentinel_missing_cells": int(
                missingness["sentinel_missing_count"].sum()
            ),
            "total_effective_missing_cells": int(
                missingness["effective_missing_count"].sum()
            ),
            "highest_effective_missingness": top_missing.to_dict(orient="records"),
            "treatment": "Missing values are reported and retained; no imputation is performed.",
        },
        "duplicates": _build_duplicate_validation(working),
        "date_validation": date_validation,
        "categorical_validation": category_validation,
        "coordinate_validation": coordinate_validation,
        "variable_roles": {
            "record_identifier": [
                column for column in ("ARREST_KEY",) if column in working.columns
            ],
            "temporal": [
                column for column in ("ARREST_DATE",) if column in working.columns
            ],
            "continuous_numeric": [
                column
                for column in CONTINUOUS_NUMERIC_COLUMNS
                if column in working.columns
            ],
            "numeric_coded_categorical": [
                column
                for column in NUMERIC_CODED_CATEGORICAL_COLUMNS
                if column in working.columns
            ],
            "text_preserved_in_processed": [
                column
                for column in TEXTUAL_IDENTIFIER_AND_CODE_COLUMNS
                if column in working.columns
            ],
            "numeric_code_caveat": (
                "PD_CD, KY_CD, ARREST_PRECINCT and JURISDICTION_CODE are "
                "identifier/category codes even when pandas infers a numeric "
                "dtype. Means, medians and variances are not substantively "
                "interpretable for these fields."
            ),
        },
    }
    passed_checks: list[str] = []
    if report["schema_validation"]["matches_expected_19_column_schema"]:
        passed_checks.append("Source matches the expected 19-column schema.")
    if not report["duplicates"]["exact_duplicate_rows_beyond_first"]:
        passed_checks.append("No exact duplicate rows were detected.")
    if not report["duplicates"]["duplicated_arrest_key_count"]:
        passed_checks.append("All nonblank ARREST_KEY values are unique.")
    if report["date_validation"].get("available") and not any(
        report["date_validation"][name]
        for name in ("parse_failure_count", "future_date_count", "abnormal_year_count")
    ):
        passed_checks.append(
            "All supplied ARREST_DATE values parsed and passed YTD year/future checks."
        )
    coordinates = report["coordinate_validation"]
    if coordinates.get("available") and not any(
        (
            coordinates["rows_with_any_source_coordinate_missing_or_blank"],
            coordinates["latitude"]["non_numeric_count"],
            coordinates["longitude"]["non_numeric_count"],
            coordinates["latitude"]["zero_count"],
            coordinates["longitude"]["zero_count"],
            coordinates["latitude"]["impossible_global_range_count"],
            coordinates["longitude"]["impossible_global_range_count"],
            coordinates["outside_approximate_nyc_bounds_count"],
        )
    ):
        passed_checks.append(
            "All latitude/longitude pairs passed numeric, nonzero, global-range, and conservative NYC-bound screening."
        )
    report["passed_checks"] = passed_checks
    report["findings"] = _build_quality_findings(report)
    report["analysis_readiness"] = {
        "temporal_descriptive_analysis": "ready_with_observed-period_caveat",
        "monthly_severity_composition": "ready_with_explicit_F_M_V_filter_and_coverage",
        "shared_baseline": "ready_no_unjustified_imputation_or_row_deletion",
    }
    return report


def _count_key_duplicates_after_exact_deduplication(
    frame: pd.DataFrame,
) -> tuple[int, int]:
    if "ARREST_KEY" not in frame.columns:
        return 0, 0
    keys = frame["ARREST_KEY"].astype("string").str.strip().replace("", pd.NA)
    duplicated = keys.notna() & keys.duplicated(keep=False)
    return int(keys.loc[duplicated].nunique()), int(duplicated.sum())


def _normalise_textual_code_value(value: object) -> object:
    """Render numeric identifiers as text without inventing a decimal suffix."""

    if value is pd.NA or value is pd.NaT or pd.isna(value):
        return pd.NA
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Integral) and not isinstance(value, bool):
        return str(int(value))
    if isinstance(value, Real) and not isinstance(value, bool):
        numeric = float(value)
        if math.isfinite(numeric) and numeric.is_integer():
            return str(int(numeric))
        return str(value)
    return str(value).strip()


def clean_dataset(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply the minimal information-preserving Part 1 cleaning policy.

    Only later occurrences of fully identical source rows are deleted.  Invalid
    dates, missing values, coordinate anomalies and non-identical repeated keys
    remain in the shared baseline dataset.
    """

    working = standardize_column_names(frame)
    original_rows = len(working)
    exact_involved = int(working.duplicated(keep=False).sum())
    duplicate_mask = working.duplicated(keep="first")
    exact_removed = int(duplicate_mask.sum())
    cleaned = working.loc[~duplicate_mask].copy()

    duplicate_keys_retained, duplicate_key_rows_retained = (
        _count_key_duplicates_after_exact_deduplication(cleaned)
    )

    trimmed_cells = 0
    trimmed_columns: list[str] = []
    for column in cleaned.columns:
        series = cleaned[column]
        if not (
            pd.api.types.is_object_dtype(series.dtype)
            or pd.api.types.is_string_dtype(series.dtype)
        ):
            continue
        changed = series.map(
            lambda value: isinstance(value, str) and value != value.strip()
        )
        changed_count = int(changed.sum())
        if changed_count:
            trimmed_cells += changed_count
            trimmed_columns.append(column)
            cleaned[column] = series.map(
                lambda value: value.strip() if isinstance(value, str) else value
            )

    numeric_code_values_converted_to_text: dict[str, int] = {}
    textual_identifier_columns: list[str] = []
    for column in TEXTUAL_IDENTIFIER_AND_CODE_COLUMNS:
        if column not in cleaned.columns:
            continue
        source = cleaned[column]
        numeric_count = int(
            source.map(
                lambda value: (
                    value is not pd.NA
                    and value is not pd.NaT
                    and not pd.isna(value)
                    and isinstance(value, Real)
                    and not isinstance(value, bool)
                )
            ).sum()
        )
        cleaned[column] = source.map(_normalise_textual_code_value).astype("string")
        numeric_code_values_converted_to_text[column] = numeric_count
        textual_identifier_columns.append(column)

    invalid_dates = 0
    missing_dates = 0
    if "ARREST_DATE" in cleaned.columns:
        raw_dates = cleaned["ARREST_DATE"]
        provided_dates = _nonblank_mask(raw_dates)
        parsed_dates = _parse_arrest_dates(raw_dates)
        invalid_dates = int((provided_dates & parsed_dates.isna()).sum())
        missing_dates = int((~provided_dates).sum())
        cleaned["ARREST_DATE"] = parsed_dates
        cleaned["YEAR"] = parsed_dates.dt.year.astype("Int64")
        cleaned["MONTH"] = parsed_dates.dt.month.astype("Int64")
        cleaned["MONTH_NAME"] = parsed_dates.dt.month_name().astype("string")
        cleaned["DAY_OF_WEEK"] = parsed_dates.dt.day_name().astype("string")
        # ISO-like convention: Monday=1, ..., Sunday=7.
        cleaned["DAY_OF_WEEK_NUM"] = (parsed_dates.dt.dayofweek + 1).astype("Int64")

    coordinate_coercion_failures: dict[str, int] = {}
    for column in ("LATITUDE", "LONGITUDE", "X_COORD_CD", "Y_COORD_CD"):
        if column not in cleaned.columns:
            continue
        source = cleaned[column]
        numeric = pd.to_numeric(source, errors="coerce")
        failures = int((_nonblank_mask(source) & numeric.isna()).sum())
        coordinate_coercion_failures[column] = failures
        cleaned[column] = numeric

    cleaned.attrs.update(working.attrs)
    stats = {
        "original_rows": int(original_rows),
        "exact_duplicate_rows_involved": exact_involved,
        "exact_duplicate_rows_removed": exact_removed,
        "rows_removed_for_invalid_dates": 0,
        "invalid_date_values_retained_as_nat": invalid_dates,
        "missing_date_values_retained_as_nat": missing_dates,
        "string_cells_trimmed": trimmed_cells,
        "string_columns_with_trimmed_values": trimmed_columns,
        "textual_identifier_and_code_columns": textual_identifier_columns,
        "numeric_identifier_and_code_values_converted_to_text": (
            numeric_code_values_converted_to_text
        ),
        "coordinate_coercion_failures_retained_as_missing": coordinate_coercion_failures,
        "non_identical_duplicated_arrest_keys_retained": duplicate_keys_retained,
        "rows_with_non_identical_duplicated_arrest_keys_retained": (
            duplicate_key_rows_retained
        ),
        "imputed_values": 0,
        "final_processed_rows": int(len(cleaned)),
        "derived_temporal_columns": [
            column
            for column in (
                "YEAR",
                "MONTH",
                "MONTH_NAME",
                "DAY_OF_WEEK",
                "DAY_OF_WEEK_NUM",
            )
            if column in cleaned.columns
        ],
        "day_of_week_numbering": "Monday=1 through Sunday=7",
    }
    return cleaned, stats


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (date, datetime, pd.Timestamp)):
        return value.isoformat()
    if value is pd.NA or value is pd.NaT:
        return None
    if hasattr(value, "item"):
        scalar = value.item()
        if isinstance(scalar, float) and math.isnan(scalar):
            return None
        return scalar
    if isinstance(value, float) and math.isnan(value):
        return None
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serialisable")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
            default=_json_default,
        )
        handle.write("\n")


def _coordinate_log_lines(coordinate_validation: dict[str, Any]) -> list[str]:
    if not coordinate_validation.get("available"):
        return [
            "Missing coordinates: Not assessed (LATITUDE/LONGITUDE pair unavailable)",
            "Rows removed for coordinate issues: 0",
        ]
    latitude = coordinate_validation["latitude"]
    longitude = coordinate_validation["longitude"]
    return [
        f"Latitude missing/blank: {latitude['source_missing_or_blank_count']}",
        f"Longitude missing/blank: {longitude['source_missing_or_blank_count']}",
        "Rows with at least one source coordinate missing/blank: "
        f"{coordinate_validation['rows_with_any_source_coordinate_missing_or_blank']}",
        "Coordinate pairs outside approximate NYC bounds: "
        f"{coordinate_validation['outside_approximate_nyc_bounds_count']}",
        "Rows removed for coordinate issues: 0",
    ]


def _write_cleaning_log(
    path: Path,
    stats: dict[str, Any],
    report: dict[str, Any],
) -> None:
    coordinate_lines = _coordinate_log_lines(report["coordinate_validation"])
    law_validation = report["categorical_validation"]["columns"].get(
        "LAW_CAT_CD", {}
    )
    law_special_lines: list[str] = []
    if law_validation:
        law_special_lines = [
            "- LAW_CAT_CD true nulls were retained: "
            f"{law_validation['missing_count']}.",
            "- LAW_CAT_CD source missing sentinels were retained without "
            "recoding: "
            f"{law_validation.get('source_missing_sentinel_counts', {})}.",
            "- LAW_CAT_CD known non-core values were retained: "
            f"{law_validation.get('known_non_core_value_counts', {})}.",
            "- LAW_CAT_CD unrecognised non-core values were retained for "
            "investigation: "
            f"{law_validation.get('unrecognised_non_core_value_counts', {})}.",
        ]
    failures = stats["coordinate_coercion_failures_retained_as_missing"]
    failure_text = ", ".join(
        f"{column}={count}" for column, count in failures.items()
    ) or "Not applicable"
    lines = [
        "# NYPD Arrests YTD Cleaning Log",
        "",
        f"Source snapshot: `{report.get('source_file') or 'not recorded'}`",
        f"Snapshot date: {report['snapshot_date']}",
        "",
        "## Row accounting",
        "",
        f"Original rows: {stats['original_rows']}",
        "Exact duplicate rows involved: "
        f"{stats['exact_duplicate_rows_involved']}",
        f"Exact duplicate rows removed: {stats['exact_duplicate_rows_removed']}",
        "Non-identical duplicated ARREST_KEY values retained: "
        f"{stats['non_identical_duplicated_arrest_keys_retained']}",
        "Rows with non-identical duplicated ARREST_KEY values retained: "
        f"{stats['rows_with_non_identical_duplicated_arrest_keys_retained']}",
        f"Final processed rows: {stats['final_processed_rows']}",
        "",
        "## Transformations and retention decisions",
        "",
        "- Column names were standardised to uppercase snake case.",
        "- Only later copies of rows that were identical across every raw field "
        "were removed. This avoids double-counting without choosing between "
        "conflicting versions of a record.",
        "- Non-identical records sharing an ARREST_KEY were retained for "
        "downstream investigation.",
        f"- String cells trimmed for surrounding whitespace: {stats['string_cells_trimmed']}.",
        "- ARREST_KEY, PD_CD, KY_CD, ARREST_PRECINCT, and "
        "JURISDICTION_CODE were preserved as textual identifier/category "
        "values; integer codes are written without artificial `.0` suffixes.",
        "- ARREST_DATE was parsed to datetime; invalid supplied values retained "
        f"as NaT: {stats['invalid_date_values_retained_as_nat']}.",
        f"- Rows removed for invalid dates: {stats['rows_removed_for_invalid_dates']}.",
        "- Missing ARREST_DATE values retained as NaT: "
        f"{stats['missing_date_values_retained_as_nat']}.",
        "- Coordinate fields were coerced to numeric. Supplied non-numeric "
        f"values converted to missing: {failure_text}.",
        *[f"- {line}." for line in coordinate_lines],
        "- Missing and anomalous coordinates were retained for the spatial "
        "analysis owner to filter under an explicit use-specific rule.",
        "- No missing values were imputed.",
        *law_special_lines,
        "- Derived temporal columns: "
        + ", ".join(stats["derived_temporal_columns"])
        + ".",
        "- DAY_OF_WEEK_NUM uses Monday=1 through Sunday=7.",
        "",
        "## Numeric-code caveat",
        "",
        "`ARREST_KEY` is an identifier. `PD_CD`, `KY_CD`, `ARREST_PRECINCT`, "
        "and `JURISDICTION_CODE` are numeric-coded categorical/identifier "
        "fields. Their numeric means, "
        "medians, and variances are not interpreted as quantities.",
        "",
        "## Rationale",
        "",
        "The shared baseline remains information-rich: invalid or missing field "
        "values are documented rather than used as a reason to discard entire "
        "arrest records. The frozen raw CSV remains unchanged.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def _resolve_output_path(
    value: str | Path | None,
    default_relative: Path,
    project_root: Path,
) -> Path:
    return _project_path(value if value is not None else default_relative, project_root)


def run_pipeline(
    raw_path: str | Path | None = None,
    processed_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Run the complete audit/clean pipeline and write all required artifacts."""

    root = Path(project_root).expanduser().resolve()
    resolved_raw = resolve_raw_path(raw_path, root)
    resolved_processed = _resolve_output_path(
        processed_path, DEFAULT_PROCESSED_RELATIVE_PATH, root
    )
    resolved_output = _resolve_output_path(
        output_dir, DEFAULT_OUTPUT_RELATIVE_DIR, root
    )
    resolved_processed.parent.mkdir(parents=True, exist_ok=True)
    resolved_output.mkdir(parents=True, exist_ok=True)

    raw = load_raw_data(resolved_raw, root)
    standardised = standardize_column_names(raw)
    snapshot_date = _snapshot_date_from_path(resolved_raw) or date.today()
    report = audit_dataset(standardised, snapshot_date=snapshot_date)
    try:
        report["source_file"] = resolved_raw.relative_to(root).as_posix()
    except ValueError:
        # An explicitly supplied snapshot outside the project remains traceable.
        report["source_file"] = str(resolved_raw)
    cleaning_source = load_raw_data(
        resolved_raw,
        root,
        preserve_code_text=True,
    )
    cleaned, cleaning_stats = clean_dataset(cleaning_source)
    report["cleaning"] = cleaning_stats

    schema_path = resolved_output / "schema_summary.csv"
    missingness_path = resolved_output / "missingness_summary.csv"
    report_path = resolved_output / "data_quality_report.json"
    categorical_path = resolved_output / "categorical_validation.json"
    coordinate_path = resolved_output / "coordinate_validation.json"
    duplicate_path = resolved_output / "duplicate_arrest_key_evidence.csv"
    cleaning_log_path = resolved_output / "cleaning_log.md"

    build_schema_summary(standardised).to_csv(
        schema_path, index=False, encoding="utf-8"
    )
    build_missingness_summary(standardised).to_csv(
        missingness_path, index=False, encoding="utf-8"
    )
    duplicate_evidence = build_duplicate_key_evidence(standardised)
    if duplicate_evidence.empty:
        if duplicate_path.exists():
            duplicate_path.unlink()
    else:
        duplicate_evidence.to_csv(duplicate_path, index=False, encoding="utf-8")

    _write_json(report_path, report)
    _write_json(categorical_path, report["categorical_validation"])
    _write_json(coordinate_path, report["coordinate_validation"])
    _write_cleaning_log(cleaning_log_path, cleaning_stats, report)
    cleaned.to_csv(
        resolved_processed,
        index=False,
        encoding="utf-8",
        date_format="%Y-%m-%d",
    )

    artifacts = {
        "processed_csv": str(resolved_processed),
        "schema_summary_csv": str(schema_path),
        "missingness_summary_csv": str(missingness_path),
        "data_quality_report_json": str(report_path),
        "categorical_validation_json": str(categorical_path),
        "coordinate_validation_json": str(coordinate_path),
        "cleaning_log_md": str(cleaning_log_path),
        "duplicate_arrest_key_evidence_csv": (
            str(duplicate_path) if not duplicate_evidence.empty else None
        ),
    }
    return {
        "raw_path": str(resolved_raw),
        "processed_path": str(resolved_processed),
        "output_dir": str(resolved_output),
        "snapshot_date": snapshot_date.isoformat(),
        "raw_rows": int(len(standardised)),
        "raw_columns": int(len(standardised.columns)),
        "processed_rows": int(len(cleaned)),
        "processed_columns": int(len(cleaned.columns)),
        "min_arrest_date": report["date_validation"].get("min_date"),
        "max_arrest_date": report["date_validation"].get("max_date"),
        "artifacts": artifacts,
        "report": report,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit and conservatively clean a frozen NYPD Arrests YTD CSV."
        )
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=None,
        help=(
            "Raw snapshot path. Relative paths are resolved from --project-root; "
            "if omitted, the latest dated snapshot is discovered."
        ),
    )
    parser.add_argument(
        "--processed",
        type=Path,
        default=None,
        help="Processed CSV path (default: data/processed/nypd_arrests_clean.csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Audit artifact directory (default: outputs/part1).",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Project root used for relative paths.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_pipeline(
        raw_path=args.raw,
        processed_path=args.processed,
        output_dir=args.output_dir,
        project_root=args.project_root,
    )
    summary = {
        "raw_snapshot": result["raw_path"],
        "raw_shape": f"{result['raw_rows']} rows x {result['raw_columns']} columns",
        "processed_shape": (
            f"{result['processed_rows']} rows x "
            f"{result['processed_columns']} columns"
        ),
        "date_range": [result["min_arrest_date"], result["max_arrest_date"]],
        "processed_csv": result["processed_path"],
        "audit_output_dir": result["output_dir"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
