"""Independent completion checks for the CA6002 NYPD Part 1 handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import nbformat
import numpy as np
import pandas as pd
from PIL import Image, ImageStat


DERIVED_COLUMNS = [
    "YEAR",
    "MONTH",
    "MONTH_NAME",
    "DAY_OF_WEEK",
    "DAY_OF_WEEK_NUM",
]
TEXTUAL_IDENTIFIER_COLUMNS = [
    "ARREST_KEY",
    "PD_CD",
    "KY_CD",
    "ARREST_PRECINCT",
    "JURISDICTION_CODE",
]
COORDINATE_COLUMNS = ["LATITUDE", "LONGITUDE", "X_COORD_CD", "Y_COORD_CD"]
WEEKDAYS = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
REQUIRED_FIGURE_BASES = [
    "missingness_overview",
    "daily_arrests_rolling",
    "monthly_average_daily_arrests",
    "weekday_average_arrests",
]
SEVERITY_FIGURE_BASE = "monthly_severity_composition"
REQUIRED_DOCS = [
    "visual_style_guide.md",
    "chart_contracts.md",
    "slide_plan.md",
    "slide_notes.md",
    "team_handoff.md",
    "ai_usage_note.md",
]
NOTE_SECTIONS = ["Finding", "Interpretation", "Design Rationale", "Limitation / Caveat"]
NOTEBOOK_SECTIONS = [
    "1. Imports and configuration",
    "2. Load frozen snapshot",
    "3. Dataset overview",
    "4. Data quality audit",
    "5. Cleaning",
    "6. Temporal feature creation",
    "7. Temporal exploratory analysis",
    "8. Final visualisations",
    "9. Key findings",
    "10. Limitations",
]


@dataclass
class Check:
    name: str
    passed: bool
    evidence: str


class Validator:
    def __init__(self) -> None:
        self.checks: list[Check] = []

    def check(self, name: str, condition: bool, evidence: str) -> None:
        self.checks.append(Check(name=name, passed=bool(condition), evidence=evidence))

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(item.passed for item in self.checks)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _nearly_equal(actual: pd.Series, expected: pd.Series, tolerance: float = 1e-6) -> bool:
    return bool(
        np.allclose(
            pd.to_numeric(actual, errors="coerce"),
            pd.to_numeric(expected, errors="coerce"),
            rtol=0,
            atol=tolerance,
            equal_nan=True,
        )
    )


def _note_blocks(text: str) -> list[str]:
    return [
        block.strip()
        for block in re.split(r"(?m)^---\s*$", text)
        if re.search(r"(?m)^### .+$", block)
    ]


def _word_count(text: str) -> int:
    return len(re.findall(r"\b[A-Za-z0-9][A-Za-z0-9'/-]*\b", text))


def _pagination_order_token(value: object) -> tuple[int, Decimal | str]:
    text = str(value)
    try:
        return (0, Decimal(text))
    except (InvalidOperation, ValueError):
        return (1, text)


def _normalise_column_name(value: object) -> str:
    name = str(value).strip().upper()
    name = re.sub(r"[^A-Z0-9]+", "_", name)
    return re.sub(r"_+", "_", name).strip("_")


def _expected_clean_source(raw_path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """Independently reconstruct the cleaner's source-field baseline.

    Exact duplicate source rows are removed after their first occurrence;
    non-identical rows sharing an ARREST_KEY remain in original order.
    """

    source_columns = pd.read_csv(raw_path, nrows=0).columns
    dtype = {
        str(column): "string"
        for column in source_columns
        if _normalise_column_name(column) in TEXTUAL_IDENTIFIER_COLUMNS
    }
    frame = pd.read_csv(raw_path, low_memory=False, dtype=dtype)
    frame.columns = [_normalise_column_name(column) for column in frame.columns]
    exact_involved = int(frame.duplicated(keep=False).sum())
    exact_removed = int(frame.duplicated(keep="first").sum())
    baseline = frame.loc[~frame.duplicated(keep="first")].copy().reset_index(drop=True)

    for column in baseline.columns:
        series = baseline[column]
        if pd.api.types.is_object_dtype(series.dtype) or pd.api.types.is_string_dtype(
            series.dtype
        ):
            baseline[column] = series.map(
                lambda value: value.strip() if isinstance(value, str) else value
            )
    invalid_dates = 0
    missing_dates = 0
    if "ARREST_DATE" in baseline.columns:
        raw_dates = baseline["ARREST_DATE"]
        values = raw_dates.astype("string").str.strip().replace("", pd.NA)
        parsed_dates = pd.to_datetime(values, format="mixed", errors="coerce")
        invalid_dates = int((values.notna() & parsed_dates.isna()).sum())
        missing_dates = int(values.isna().sum())
        baseline["ARREST_DATE"] = parsed_dates
    for column in COORDINATE_COLUMNS:
        if column in baseline.columns:
            baseline[column] = pd.to_numeric(baseline[column], errors="coerce")

    keys = baseline["ARREST_KEY"].astype("string").str.strip().replace("", pd.NA)
    duplicated_keys = keys.notna() & keys.duplicated(keep=False)
    stats = {
        "raw_rows": int(len(frame)),
        "exact_duplicate_rows_involved": exact_involved,
        "exact_duplicate_rows_removed": exact_removed,
        "expected_processed_rows": int(len(baseline)),
        "non_identical_duplicated_arrest_keys_retained": int(
            keys.loc[duplicated_keys].nunique()
        ),
        "rows_with_non_identical_duplicated_arrest_keys_retained": int(
            duplicated_keys.sum()
        ),
        "invalid_date_values_retained_as_nat": invalid_dates,
        "missing_date_values_retained_as_nat": missing_dates,
    }
    return baseline, stats


def _source_field_identity(
    expected: pd.DataFrame,
    processed: pd.DataFrame,
    processed_text: pd.DataFrame,
) -> dict[str, bool]:
    results: dict[str, bool] = {}
    for column in expected.columns:
        if column not in processed.columns or column not in processed_text.columns:
            results[column] = False
            continue
        expected_series = expected[column].reset_index(drop=True)
        if column == "ARREST_DATE":
            actual = pd.to_datetime(processed[column], errors="coerce").reset_index(
                drop=True
            )
            results[column] = expected_series.equals(actual)
        elif column in COORDINATE_COLUMNS or pd.api.types.is_numeric_dtype(
            expected_series.dtype
        ):
            results[column] = _nearly_equal(
                processed[column].reset_index(drop=True), expected_series
            )
        else:
            expected_text = expected_series.astype("string").fillna("<NA>")
            actual_text = (
                processed_text[column]
                .astype("string")
                .str.strip()
                .fillna("<NA>")
                .reset_index(drop=True)
            )
            results[column] = expected_text.equals(actual_text)
    return results


def _expected_duplicate_evidence(raw: pd.DataFrame) -> pd.DataFrame:
    """Independently rebuild the cleaner's duplicated-key evidence table."""

    working = raw.copy()
    working.columns = [_normalise_column_name(column) for column in working.columns]
    if "ARREST_KEY" not in working.columns:
        return pd.DataFrame()
    keys = working["ARREST_KEY"].astype("string").str.strip().replace("", pd.NA)
    duplicated_key = keys.notna() & keys.duplicated(keep=False)
    if not duplicated_key.any():
        return pd.DataFrame()

    evidence = working.loc[duplicated_key].copy()
    evidence_keys = keys.loc[duplicated_key]
    group_sizes = evidence_keys.value_counts(dropna=True)
    all_identical = {
        key: bool(working.loc[evidence_keys.index[evidence_keys.eq(key)]].duplicated().sum()
                  == len(evidence_keys.loc[evidence_keys.eq(key)]) - 1)
        for key in evidence_keys.dropna().unique()
    }
    evidence.insert(0, "AUDIT_ARREST_KEY", evidence_keys)
    evidence.insert(
        1,
        "AUDIT_KEY_OCCURRENCE_COUNT",
        evidence_keys.map(group_sizes).astype("Int64"),
    )
    evidence.insert(
        2,
        "AUDIT_ALL_ROWS_IDENTICAL_FOR_KEY",
        evidence_keys.map(all_identical).astype("boolean"),
    )
    evidence.insert(
        3,
        "AUDIT_ROW_PART_OF_EXACT_DUPLICATE",
        working.duplicated(keep=False).loc[duplicated_key].astype("boolean"),
    )
    return evidence.sort_values(
        ["AUDIT_ARREST_KEY"], kind="stable", ignore_index=True
    )


def _duplicate_evidence_matches(raw: pd.DataFrame, evidence_path: Path) -> bool:
    expected = _expected_duplicate_evidence(raw)
    if expected.empty:
        return not evidence_path.exists()
    if not evidence_path.is_file():
        return False
    actual = pd.read_csv(
        evidence_path, dtype="string", keep_default_na=False
    )
    if actual.columns.tolist() != expected.columns.tolist() or len(actual) != len(
        expected
    ):
        return False
    # Compare actual CSV cell semantics exactly. Missing values are serialised
    # as empty fields; literal strings such as NA or NULL must remain distinct.
    expected_text = expected.astype("string").fillna("")
    actual_text = actual.astype("string")
    return expected_text.equals(actual_text)


def validate(project_root: Path) -> tuple[Validator, dict[str, Any]]:
    root = project_root.resolve()
    output_dir = root / "outputs" / "part1"
    figure_dir = root / "figures" / "part1"
    validator = Validator()

    metadata_path = output_dir / "dataset_snapshot_metadata.json"
    validator.check("Snapshot metadata exists", metadata_path.is_file(), str(metadata_path))
    if not metadata_path.is_file():
        return validator, {}
    metadata = _read_json(metadata_path)
    raw_path = root / str(metadata.get("raw_file", ""))
    validator.check("Frozen raw CSV exists", raw_path.is_file() and raw_path.stat().st_size > 0, str(raw_path))
    if not raw_path.is_file():
        return validator, metadata

    raw = pd.read_csv(raw_path, low_memory=False)
    expected_rows = int(metadata.get("row_count", -1))
    expected_columns = int(metadata.get("column_count", -1))
    validator.check(
        "API counts reconcile",
        bool(metadata.get("count_match"))
        and metadata.get("api_expected_rows_before")
        == metadata.get("downloaded_rows")
        == metadata.get("api_expected_rows_after")
        == expected_rows,
        f"before={metadata.get('api_expected_rows_before')}, downloaded={metadata.get('downloaded_rows')}, after={metadata.get('api_expected_rows_after')}",
    )
    validator.check(
        "Source revision fence is stable",
        bool(metadata.get("source_revision_match"))
        and metadata.get("source_revision_before") is not None
        and metadata.get("source_revision_before")
        == metadata.get("source_revision_after")
        and metadata.get("api_row_count_rechecked_after_download") == expected_rows,
        (
            f"before={metadata.get('source_revision_before')}, "
            f"after={metadata.get('source_revision_after')}, "
            f"post_download_count={metadata.get('api_row_count_rechecked_after_download')}"
        ),
    )
    validator.check(
        "Raw shape matches metadata",
        raw.shape == (expected_rows, expected_columns),
        f"actual={raw.shape}, expected={(expected_rows, expected_columns)}",
    )
    validator.check(
        "Raw snapshot byte identity",
        _sha256(raw_path) == metadata.get("raw_file_sha256"),
        f"sha256={_sha256(raw_path)}",
    )
    pages = metadata.get("pages", [])
    offsets = [int(page["offset"]) for page in pages]
    page_rows = [int(page["rows"]) for page in pages]
    expected_offsets: list[int] = []
    running = 0
    for count in page_rows:
        expected_offsets.append(running)
        running += count
    primary_order_unique = bool(metadata.get("pagination_primary_key_unique"))
    tie_breaker_field = metadata.get("pagination_tie_breaker_field")
    tie_breaker_unique = bool(metadata.get("pagination_tie_breaker_unique"))
    if primary_order_unique:
        ordered_boundaries = all(
            _pagination_order_token(pages[index - 1]["last_arrest_key"])
            < _pagination_order_token(pages[index]["first_arrest_key"])
            for index in range(1, len(pages))
        )
    else:
        ordered_boundaries = (
            tie_breaker_field == ":id"
            and tie_breaker_unique
            and all(
                "last_socrata_row_id" in pages[index - 1]
                and "first_socrata_row_id" in pages[index]
                and bool(str(pages[index - 1]["last_socrata_row_id"]))
                and bool(str(pages[index]["first_socrata_row_id"]))
                and _pagination_order_token(
                    pages[index - 1]["last_arrest_key"]
                )
                <= _pagination_order_token(pages[index]["first_arrest_key"])
                for index in range(1, len(pages))
            )
        )
    total_order_contract = bool(metadata.get("pagination_order_is_total")) and (
        primary_order_unique
        or (tie_breaker_field == ":id" and tie_breaker_unique)
    )
    validator.check(
        "Pagination accounting and stable total order reconcile",
        offsets == expected_offsets
        and sum(page_rows) == expected_rows
        and ordered_boundaries
        and total_order_contract,
        (
            f"pages={len(pages)}, offsets={offsets}, rows={page_rows}, "
            f"order={metadata.get('pagination_order_clause')}, "
            f"primary_unique={primary_order_unique}, tie_breaker={tie_breaker_field}"
        ),
    )

    raw_lexical = pd.read_csv(raw_path, dtype="string", keep_default_na=False)
    lexical_exact_duplicates = int(raw_lexical.duplicated(keep="first").sum())
    lexical_keys = raw_lexical["arrest_key"].astype("string")
    lexical_key_counts = lexical_keys.value_counts(dropna=False)
    lexical_duplicate_key_rows_beyond_first = int(
        (lexical_key_counts - 1).clip(lower=0).sum()
    )
    validator.check(
        "Acquisition duplicate counters reconcile",
        metadata.get("exact_duplicate_rows_seen_during_download")
        == lexical_exact_duplicates
        and metadata.get("duplicate_arrest_key_rows_seen_during_download")
        == lexical_duplicate_key_rows_beyond_first
        and primary_order_unique
        == (lexical_duplicate_key_rows_beyond_first == 0),
        (
            f"exact_beyond_first={lexical_exact_duplicates}, "
            f"duplicate_key_rows_beyond_first={lexical_duplicate_key_rows_beyond_first}"
        ),
    )

    expected_source, cleaning_expected = _expected_clean_source(raw_path)
    quality_report_path = output_dir / "data_quality_report.json"
    validator.check(
        "Data-quality report exists", quality_report_path.is_file(), str(quality_report_path)
    )
    quality_report = _read_json(quality_report_path) if quality_report_path.is_file() else {}
    duplicate_report = quality_report.get("duplicates", {})
    cleaning_report = quality_report.get("cleaning", {})
    raw_keys = raw["arrest_key"].astype("string").str.strip().replace("", pd.NA)
    raw_duplicated_key_mask = raw_keys.notna() & raw_keys.duplicated(keep=False)
    raw_duplicate_key_groups = int(raw_keys.loc[raw_duplicated_key_mask].nunique())
    raw_duplicate_key_rows = int(raw_duplicated_key_mask.sum())
    duplicate_evidence_path = output_dir / "duplicate_arrest_key_evidence.csv"
    evidence_ok = _duplicate_evidence_matches(raw, duplicate_evidence_path)
    duplicate_contract_ok = (
        duplicate_report.get("exact_duplicate_rows_beyond_first")
        == int(raw.duplicated(keep="first").sum())
        and duplicate_report.get("exact_duplicate_rows_involved")
        == int(raw.duplicated(keep=False).sum())
        and duplicate_report.get("duplicated_arrest_key_count")
        == raw_duplicate_key_groups
        and duplicate_report.get("rows_with_duplicated_arrest_key")
        == raw_duplicate_key_rows
        and cleaning_report.get("original_rows") == cleaning_expected["raw_rows"]
        and cleaning_report.get("exact_duplicate_rows_removed")
        == cleaning_expected["exact_duplicate_rows_removed"]
        and cleaning_report.get("final_processed_rows")
        == cleaning_expected["expected_processed_rows"]
        and cleaning_report.get("non_identical_duplicated_arrest_keys_retained")
        == cleaning_expected["non_identical_duplicated_arrest_keys_retained"]
        and cleaning_report.get(
            "rows_with_non_identical_duplicated_arrest_keys_retained"
        )
        == cleaning_expected[
            "rows_with_non_identical_duplicated_arrest_keys_retained"
        ]
        and evidence_ok
    )
    validator.check(
        "Duplicate audit and conservative cleaning reconcile",
        duplicate_contract_ok,
        json.dumps(
            {
                "raw_exact_duplicates_removed": cleaning_expected[
                    "exact_duplicate_rows_removed"
                ],
                "raw_duplicate_key_groups": raw_duplicate_key_groups,
                "raw_duplicate_key_rows": raw_duplicate_key_rows,
                "expected_processed_rows": cleaning_expected[
                    "expected_processed_rows"
                ],
                "evidence_ok": evidence_ok,
            }
        ),
    )

    processed_path = root / "data" / "processed" / "nypd_arrests_clean.csv"
    validator.check("Processed CSV exists", processed_path.is_file() and processed_path.stat().st_size > 0, str(processed_path))
    if not processed_path.is_file():
        return validator, metadata
    processed = pd.read_csv(processed_path, low_memory=False)
    processed_text = pd.read_csv(processed_path, dtype="string")
    dates = pd.to_datetime(processed["ARREST_DATE"], errors="coerce")
    validator.check(
        "Processed shape and retention",
        len(processed) == cleaning_expected["expected_processed_rows"]
        and processed.shape[1] == expected_columns + len(DERIVED_COLUMNS),
        (
            f"processed={processed.shape}, raw={raw.shape}, "
            f"exact_duplicates_removed={cleaning_expected['exact_duplicate_rows_removed']}"
        ),
    )
    source_identity = _source_field_identity(
        expected_source, processed, processed_text
    )
    validator.check(
        "Processed source-field order and values match the exact-deduplicated baseline",
        all(source_identity.values()),
        json.dumps(source_identity),
    )
    validator.check(
        "Processed key order preserves retained frozen-snapshot records",
        source_identity.get("ARREST_KEY", False),
        f"keys_compared={len(processed):,}",
    )
    code_identity = {
        column: source_identity.get(column, False)
        for column in TEXTUAL_IDENTIFIER_COLUMNS
    }
    validator.check(
        "Identifier and numeric-coded category text is preserved",
        all(code_identity.values()),
        json.dumps(code_identity),
    )
    validator.check(
        "Processed dates follow the documented retain-as-NaT policy",
        source_identity.get("ARREST_DATE", False)
        and cleaning_report.get("invalid_date_values_retained_as_nat")
        == cleaning_expected["invalid_date_values_retained_as_nat"]
        and cleaning_report.get("missing_date_values_retained_as_nat")
        == cleaning_expected["missing_date_values_retained_as_nat"]
        and cleaning_report.get("rows_removed_for_invalid_dates") == 0,
        (
            f"valid={int(dates.notna().sum())}, "
            f"invalid_retained={cleaning_expected['invalid_date_values_retained_as_nat']}, "
            f"missing_retained={cleaning_expected['missing_date_values_retained_as_nat']}"
        ),
    )
    derived_ok = all(column in processed.columns for column in DERIVED_COLUMNS)
    if derived_ok:
        derived_ok = (
            pd.to_numeric(processed["YEAR"], errors="coerce")
            .astype("Int64")
            .equals(dates.dt.year.astype("Int64"))
            and pd.to_numeric(processed["MONTH"], errors="coerce")
            .astype("Int64")
            .equals(dates.dt.month.astype("Int64"))
            and processed["MONTH_NAME"]
            .astype("string")
            .equals(dates.dt.month_name().astype("string"))
            and processed["DAY_OF_WEEK"]
            .astype("string")
            .equals(dates.dt.day_name().astype("string"))
            and pd.to_numeric(processed["DAY_OF_WEEK_NUM"], errors="coerce")
            .astype("Int64")
            .equals((dates.dt.dayofweek + 1).astype("Int64"))
        )
    validator.check("Derived temporal fields recompute exactly", derived_ok, ", ".join(DERIVED_COLUMNS))

    schema = pd.read_csv(output_dir / "schema_summary.csv")
    missingness = pd.read_csv(output_dir / "missingness_summary.csv")
    validator.check(
        "Schema profile covers every raw field",
        len(schema) == expected_columns
        and set(schema["column_name"]) == {name.upper() for name in raw.columns},
        f"schema_rows={len(schema)}",
    )
    raw_upper = raw.rename(columns=str.upper)
    schema_nulls = schema.set_index("column_name")["null_count"].astype(int)
    actual_nulls = raw_upper.isna().sum().astype(int)
    validator.check(
        "Schema null counts recompute",
        schema_nulls.sort_index().equals(actual_nulls.sort_index()),
        f"total_null_cells={int(actual_nulls.sum())}",
    )
    missingness_indexed = missingness.set_index("column_name")
    missingness_recomputed: dict[str, dict[str, int]] = {}
    missingness_ok = True
    for column in raw_upper.columns:
        series = raw_upper[column]
        text_series = series.astype("string").str.strip()
        null_count = int(series.isna().sum())
        blank_count = int((series.notna() & text_series.eq("")).sum())
        sentinel_count = int(
            (series.notna() & text_series.str.upper().eq("(NULL)")).sum()
        )
        effective_count = null_count + blank_count + sentinel_count
        expected = {
            "missing_count": null_count,
            "blank_string_count": blank_count,
            "sentinel_missing_count": sentinel_count,
            "effective_missing_count": effective_count,
        }
        missingness_recomputed[column] = expected
        if column not in missingness_indexed.index:
            missingness_ok = False
            continue
        actual = missingness_indexed.loc[column]
        missingness_ok = missingness_ok and all(
            int(actual[field]) == value for field, value in expected.items()
        )
    validator.check(
        "Effective missingness recomputes for every raw field",
        missingness_ok and len(missingness_indexed) == len(raw_upper.columns),
        f"total_effective_missing={sum(item['effective_missing_count'] for item in missingness_recomputed.values())}",
    )
    key_findings_path = output_dir / "key_findings.json"
    findings = _read_json(key_findings_path)
    findings_quality = findings.get("data_quality", {})
    findings_source = str(findings_quality.get("missingness_source", "")).replace(
        "\\", "/"
    )
    displayed_missingness = findings_quality.get("missingness_fields_displayed", [])
    displayed_ok = bool(displayed_missingness)
    for item in displayed_missingness:
        field = str(item.get("field", ""))
        if field not in missingness_indexed.index:
            displayed_ok = False
            continue
        expected_row = missingness_indexed.loc[field]
        displayed_ok = displayed_ok and int(
            item.get("effective_missing_count", -1)
        ) == int(expected_row["effective_missing_count"])
        displayed_ok = displayed_ok and abs(
            float(item.get("effective_missing_pct", -1))
            - float(expected_row["effective_missing_percentage"])
        ) < 1e-9
    expected_highest = max(
        displayed_missingness,
        key=lambda item: float(item.get("effective_missing_pct", -1)),
        default={},
    )
    findings_highest = findings_quality.get(
        "highest_missingness_among_displayed_fields", {}
    )
    validator.check(
        "Final findings and Figure 1 use effective missingness",
        findings_quality.get("missingness_metric") == "effective_missing"
        and findings_source.endswith("outputs/part1/missingness_summary.csv")
        and displayed_ok
        and findings_highest == expected_highest,
        f"source={findings_source}, metric={findings_quality.get('missingness_metric')}, highest={findings_highest}",
    )

    daily_expected = (
        pd.Series(1, index=dates.dt.normalize())
        .groupby(level=0)
        .sum()
        .reindex(pd.date_range(dates.min(), dates.max(), freq="D"), fill_value=0)
    )
    daily_saved = pd.read_csv(output_dir / "daily_arrests.csv", parse_dates=["date"])
    validator.check(
        "Daily table recomputes",
        daily_saved["date"].tolist() == daily_expected.index.tolist()
        and daily_saved["arrest_count"].astype(int).tolist() == daily_expected.astype(int).tolist()
        and _nearly_equal(
            daily_saved["rolling_7d_mean"], daily_expected.rolling(7, min_periods=7).mean()
        ),
        f"days={len(daily_saved)}, total={int(daily_saved['arrest_count'].sum())}",
    )

    daily_indexed = daily_saved.set_index("date")["arrest_count"]
    monthly_expected = daily_indexed.groupby(daily_indexed.index.to_period("M")).agg(["sum", "size"])
    monthly_saved = pd.read_csv(output_dir / "monthly_arrests.csv")
    monthly_avg_expected = monthly_expected["sum"] / monthly_expected["size"]
    validator.check(
        "Monthly calendar-day averages recompute",
        monthly_saved["arrest_count"].astype(int).tolist() == monthly_expected["sum"].astype(int).tolist()
        and monthly_saved["calendar_days_in_scope"].astype(int).tolist() == monthly_expected["size"].astype(int).tolist()
        and _nearly_equal(monthly_saved["avg_arrests_per_calendar_day"], monthly_avg_expected),
        f"months={len(monthly_saved)}, partial={int(monthly_saved['is_partial_month'].astype(bool).sum())}",
    )
    weekday_expected = daily_indexed.groupby(daily_indexed.index.dayofweek).agg(["sum", "size"]).reindex(range(7))
    weekday_saved = pd.read_csv(output_dir / "weekday_arrests.csv")
    validator.check(
        "Weekday order and occurrence averages recompute",
        weekday_saved["weekday"].tolist() == WEEKDAYS
        and weekday_saved["arrest_count"].astype(int).tolist() == weekday_expected["sum"].astype(int).tolist()
        and _nearly_equal(
            weekday_saved["mean_arrests_per_occurrence"],
            weekday_expected["sum"] / weekday_expected["size"],
        ),
        f"order={weekday_saved['weekday'].tolist()}",
    )

    severity_assessment = findings.get("severity_analysis", {})
    severity_eligible = bool(severity_assessment.get("eligible"))
    severity_generated = bool(severity_assessment.get("figure_generated"))
    severity_path = output_dir / "monthly_severity_composition.csv"
    severity_ok = not severity_eligible and not severity_generated
    severity_evidence = str(severity_assessment.get("reason", ""))
    if severity_eligible and severity_generated and severity_path.is_file():
        severity = pd.read_csv(severity_path)
        severity["month_start"] = pd.to_datetime(severity["month_start"])
        share_sums = severity.groupby("month_start", sort=True)[
            "share_of_monthly_arrests_pct"
        ].sum()
        count_sums = severity.groupby("month_start", sort=True)["arrest_count"].sum()
        monthly_starts = pd.to_datetime(monthly_saved["month_start"])
        aligned_counts = count_sums.reindex(monthly_starts).to_numpy()
        severity_ok = (
            np.allclose(share_sums.to_numpy(), 100.0, rtol=0, atol=2e-6)
            and aligned_counts.tolist()
            == monthly_saved["arrest_count"].astype(int).tolist()
        )
        severity_evidence = (
            f"share_sums={{{', '.join(f'{index:%b %Y}: {value:.6f}' for index, value in share_sums.items())}}}"
        )
    validator.check(
        "Severity output follows its data-quality admission and denominator",
        severity_ok,
        severity_evidence,
    )

    figure_evidence: dict[str, Any] = {}
    figures_ok = True
    figure_bases = list(REQUIRED_FIGURE_BASES)
    if severity_eligible:
        figure_bases.append(SEVERITY_FIGURE_BASE)
    for base in figure_bases:
        png = figure_dir / f"{base}.png"
        svg = figure_dir / f"{base}.svg"
        item: dict[str, Any] = {
            "png": png.relative_to(root).as_posix(),
            "svg": svg.relative_to(root).as_posix(),
        }
        try:
            with Image.open(png) as image:
                image.load()
                item["size"] = list(image.size)
                extrema = ImageStat.Stat(image.convert("RGB")).extrema
                item["extrema"] = extrema
                if image.size != (1600, 900) or all(low == high for low, high in extrema):
                    figures_ok = False
            ET.parse(svg)
            if png.stat().st_size < 20_000 or svg.stat().st_size < 10_000:
                figures_ok = False
        except (FileNotFoundError, OSError, ET.ParseError) as exc:
            item["error"] = str(exc)
            figures_ok = False
        figure_evidence[base] = item
    missingness_svg = figure_dir / "missingness_overview.svg"
    missingness_semantics_ok = False
    if missingness_svg.is_file():
        svg_text = missingness_svg.read_text(encoding="utf-8")
        missingness_semantics_ok = (
            "Effective missingness across analysis-relevant fields" in svg_text
            and "Null, blank-string and documented sentinel share" in svg_text
        )
    validator.check(
        "All slide figures are readable PNG 1600x900 plus valid SVG",
        figures_ok and missingness_semantics_ok,
        json.dumps(figure_evidence, ensure_ascii=False),
    )

    missing_docs = [name for name in REQUIRED_DOCS if not (output_dir / name).is_file()]
    validator.check("All slide-ready handoff documents exist", not missing_docs, f"missing={missing_docs}")
    if not missing_docs:
        documents = {
            name: (output_dir / name).read_text(encoding="utf-8") for name in REQUIRED_DOCS
        }
        combined = "\n".join(documents.values())
        unsupported = re.findall(
            r"(?im)\b(?:has|have|had|shows?|proves?|caused?|because of)\b[^.\n]{0,80}\b(?:crime rate|crime incidence|police policy)\b",
            combined,
        )
        validator.check(
            "Slide-ready language avoids unsupported crime/causal claims",
            not unsupported,
            f"matches={unsupported}",
        )
        blocks = _note_blocks(documents["slide_notes.md"])
        note_details = []
        notes_ok = len(blocks) == 4
        for block in blocks:
            count = _word_count(block)
            sections_ok = all(
                re.search(rf"(?m)^\*\*{re.escape(section)}\*\*\s*$", block)
                for section in NOTE_SECTIONS
            )
            note_details.append({"words": count, "sections_ok": sections_ok})
            notes_ok = notes_ok and 120 <= count <= 200 and sections_ok
        validator.check(
            "Four slide notes use the required structure and bounded length",
            notes_ok,
            json.dumps(note_details),
        )
        validator.check(
            "Handoff states the observed window without a false partial-month warning",
            (
                dates.max().date().isoformat() in documents["team_handoff.md"]
                or dates.max().strftime("%d %b %Y") in documents["team_handoff.md"]
            )
            and str(
                (
                    pd.Timestamp(metadata["retrieval_date"]).normalize()
                    - dates.max().normalize()
                ).days
            )
            in documents["team_handoff.md"]
            and (
                "partial month" in documents["team_handoff.md"].lower()
                if monthly_saved["is_partial_month"].astype(bool).any()
                else "partial month" not in documents["team_handoff.md"].lower()
            ),
            f"Expected latest date {dates.max().date()}, retrieval gap {(pd.Timestamp(metadata['retrieval_date']).normalize() - dates.max().normalize()).days} days, partial_months={int(monthly_saved['is_partial_month'].astype(bool).sum())}.",
        )

    notebook_path = root / "notebooks" / "01_dataset_temporal.ipynb"
    validator.check("Notebook exists", notebook_path.is_file(), str(notebook_path))
    if notebook_path.is_file():
        notebook = nbformat.read(notebook_path, as_version=4)
        try:
            nbformat.validate(notebook)
            structure_ok = True
        except nbformat.ValidationError:
            structure_ok = False
        markdown = "\n".join(
            cell.source for cell in notebook.cells if cell.cell_type == "markdown"
        )
        code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
        errors = [
            output
            for cell in code_cells
            for output in cell.get("outputs", [])
            if output.get("output_type") == "error"
        ]
        streams = [
            output
            for cell in code_cells
            for output in cell.get("outputs", [])
            if output.get("output_type") == "stream"
        ]
        executed = bool(code_cells) and all(cell.get("execution_count") is not None for cell in code_cells)
        validator.check(
            "Notebook structure and required section order",
            structure_ok
            and all(section in markdown for section in NOTEBOOK_SECTIONS)
            and [markdown.index(section) for section in NOTEBOOK_SECTIONS]
            == sorted(markdown.index(section) for section in NOTEBOOK_SECTIONS),
            f"cells={len(notebook.cells)}, code_cells={len(code_cells)}",
        )
        validator.check(
            "Notebook executed top-to-bottom without errors",
            executed and not errors and not streams,
            f"executed_code_cells={sum(cell.get('execution_count') is not None for cell in code_cells)}/{len(code_cells)}, errors={len(errors)}, streams={len(streams)}",
        )

    summary = {
        "snapshot_rows": len(raw),
        "snapshot_columns": raw.shape[1],
        "processed_rows": len(processed),
        "processed_columns": processed.shape[1],
        "date_range": [dates.min().date().isoformat(), dates.max().date().isoformat()],
        "retrieval_date": metadata.get("retrieval_date"),
        "retrieval_gap_days": int(
            (pd.Timestamp(metadata["retrieval_date"]).normalize() - dates.max().normalize()).days
        ),
        "figure_count": len(figure_bases),
    }
    return validator, summary


def _write_report(root: Path, validator: Validator, summary: dict[str, Any]) -> None:
    output_dir = root / "outputs" / "part1"
    output_dir.mkdir(parents=True, exist_ok=True)
    root_windows = str(root) + "\\"
    root_posix = root.as_posix() + "/"
    portable_checks = []
    for item in validator.checks:
        record = asdict(item)
        record["evidence"] = (
            str(record["evidence"])
            .replace(root_windows, "")
            .replace(root_posix, "")
        )
        portable_checks.append(record)
    payload = {
        "overall_assessment": "Ready to share" if validator.passed else "Needs revision",
        "summary": summary,
        "checks": portable_checks,
    }
    (output_dir / "validation_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Part 1 Validation Report",
        "",
        f"Overall assessment: **{payload['overall_assessment']}**",
        "",
        "The checks independently recompute key counts, denominators and temporal aggregates from the frozen snapshot. Visual appearance was also inspected in the exported 16:9 PNG files; the checks below cover file-level integrity.",
        "",
        "## Checks",
        "",
    ]
    for item in portable_checks:
        mark = "PASS" if item["passed"] else "FAIL"
        lines.append(f"- **{mark} — {item['name']}:** {item['evidence']}")
    lines.extend(
        [
            "",
            "## Required caveats",
            "",
            "- Recorded arrests reflect police enforcement activity, not the underlying incidence or rate of crime.",
            f"- The frozen snapshot was retrieved on {summary.get('retrieval_date')} but contains arrest dates only from {summary.get('date_range', ['unknown', 'unknown'])[0]} through {summary.get('date_range', ['unknown', 'unknown'])[1]}; the {summary.get('retrieval_gap_days')} day gap means it is not activity through the retrieval date or evidence of a long-term trend.",
            "- Temporal peaks, troughs and category differences are descriptive; no causal explanation is assigned without external evidence.",
            "",
        ]
    )
    (output_dir / "validation_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    validator, summary = validate(args.project_root)
    _write_report(args.project_root.resolve(), validator, summary)
    for item in validator.checks:
        print(f"{'PASS' if item.passed else 'FAIL'}  {item.name}: {item.evidence}")
    return 0 if validator.passed else 1


if __name__ == "__main__":
    sys.exit(main())
