"""Generate the reproducible CA6002 Part 1 analysis notebook with nbformat."""

from __future__ import annotations

import argparse
from pathlib import Path
from textwrap import dedent
from typing import Sequence

import nbformat
from nbformat import NotebookNode


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SECTION_TITLES = (
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
)


def _markdown(title: str, body: str) -> NotebookNode:
    return nbformat.v4.new_markdown_cell(f"# {title}\n\n{body.strip()}")


def _code(source: str) -> NotebookNode:
    return nbformat.v4.new_code_cell(dedent(source).strip())


def _make_notebook() -> NotebookNode:
    cells: list[NotebookNode] = []

    cells.extend(
        [
            _markdown(
                SECTION_TITLES[0],
                "Configure one deterministic project root, import the reusable pipeline modules, "
                "and keep all tabular output bounded for a reader-facing handoff.",
            ),
            _code(
                """
                from pathlib import Path
                from calendar import monthrange
                import json
                import sys

                import pandas as pd
                from IPython.display import Markdown, display

                working_directory = Path.cwd().resolve()
                if (working_directory / "src").is_dir():
                    PROJECT_ROOT = working_directory
                elif (working_directory.parent / "src").is_dir():
                    PROJECT_ROOT = working_directory.parent
                else:
                    raise RuntimeError("Run this notebook from the project root or notebooks directory.")

                SRC_DIR = PROJECT_ROOT / "src"
                if str(SRC_DIR) not in sys.path:
                    sys.path.insert(0, str(SRC_DIR))

                import clean_nypd_data
                import analyze_temporal
                from build_deliverables import build_deliverables

                OUTPUT_DIR = PROJECT_ROOT / "outputs" / "part1"
                FIGURES_DIR = PROJECT_ROOT / "figures" / "part1"
                PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "nypd_arrests_clean.csv"

                pd.set_option("display.max_rows", 15)
                pd.set_option("display.max_columns", 12)
                pd.set_option("display.width", 120)
                """
            ),
        ]
    )

    cells.extend(
        [
            _markdown(
                SECTION_TITLES[1],
                "Load only the immutable raw file named by snapshot metadata. The checks below "
                "reject path escape, empty files, and metadata shape drift before analysis begins.",
            ),
            _code(
                """
                metadata_path = OUTPUT_DIR / "dataset_snapshot_metadata.json"
                if not metadata_path.is_file():
                    raise FileNotFoundError(f"Snapshot metadata not found: {metadata_path}")

                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                raw_reference = metadata.get("raw_file")
                if not raw_reference:
                    raise KeyError("dataset_snapshot_metadata.json must contain raw_file")

                raw_path = Path(raw_reference)
                if not raw_path.is_absolute():
                    raw_path = PROJECT_ROOT / raw_path
                raw_path = raw_path.resolve(strict=True)
                try:
                    raw_path.relative_to(PROJECT_ROOT)
                except ValueError as exc:
                    raise ValueError("Metadata raw_file must resolve inside the project root") from exc

                raw_df = pd.read_csv(raw_path, low_memory=False)
                if raw_df.empty:
                    raise ValueError(f"Frozen snapshot is empty: {raw_path}")

                expected_rows = int(metadata["row_count"])
                expected_columns = int(metadata["column_count"])
                if raw_df.shape != (expected_rows, expected_columns):
                    raise AssertionError(
                        f"Snapshot shape {raw_df.shape} does not match metadata "
                        f"({expected_rows}, {expected_columns})"
                    )

                server_rows = metadata.get(
                    "api_expected_rows_after",
                    metadata.get("api_expected_rows", metadata.get("expected_rows")),
                )
                if server_rows is not None and int(server_rows) != len(raw_df):
                    raise AssertionError("Downloaded row count does not match the API expected count")

                display(pd.DataFrame([{
                    "raw_file": raw_path.relative_to(PROJECT_ROOT).as_posix(),
                    "rows": len(raw_df),
                    "columns": raw_df.shape[1],
                    "retrieval_date": metadata.get("retrieval_date"),
                    "api_rows_reconciled": server_rows is None or int(server_rows) == len(raw_df),
                }]))
                """
            ),
        ]
    )

    cells.extend(
        [
            _markdown(
                SECTION_TITLES[2],
                "Summarise shape, date coverage, and the audited schema without displaying raw "
                "records or a wide unbounded dataframe.",
            ),
            _code(
                """
                raw_column_lookup = {str(column).strip().upper(): column for column in raw_df.columns}
                if "ARREST_DATE" not in raw_column_lookup:
                    raise KeyError("Frozen snapshot has no ARREST_DATE column")

                raw_arrest_dates = pd.to_datetime(
                    raw_df[raw_column_lookup["ARREST_DATE"]], errors="coerce"
                )
                overview = pd.DataFrame([{
                    "rows": len(raw_df),
                    "columns": raw_df.shape[1],
                    "parsed_dates": int(raw_arrest_dates.notna().sum()),
                    "invalid_dates": int(raw_arrest_dates.isna().sum()),
                    "min_arrest_date": raw_arrest_dates.min().date() if raw_arrest_dates.notna().any() else None,
                    "max_arrest_date": raw_arrest_dates.max().date() if raw_arrest_dates.notna().any() else None,
                }])
                display(overview)
                display(pd.DataFrame({"column_name": list(raw_df.columns)}))
                """
            ),
        ]
    )

    cells.extend(
        [
            _markdown(
                SECTION_TITLES[3],
                "Run the importable audit functions on a standardised copy. The compact preview "
                "shows the fields with the greatest missingness; full evidence is written by the pipeline.",
            ),
            _code(
                """
                standardised_raw_df = clean_nypd_data.standardize_column_names(raw_df.copy())
                audit_report = clean_nypd_data.audit_dataset(
                    standardised_raw_df,
                    snapshot_date=metadata.get("retrieval_date"),
                )
                schema_preview = clean_nypd_data.build_schema_summary(standardised_raw_df)
                missingness_preview = clean_nypd_data.build_missingness_summary(standardised_raw_df)

                display(
                    missingness_preview
                    .sort_values("effective_missing_percentage", ascending=False)
                    .loc[:, ["column_name", "effective_missing_count", "effective_missing_percentage"]]
                    .head(12)
                    .reset_index(drop=True)
                )
                display(pd.DataFrame({
                    "audit_section": list(audit_report.keys()),
                    "status": ["computed"] * len(audit_report),
                }))
                """
            ),
        ]
    )

    cells.extend(
        [
            _markdown(
                SECTION_TITLES[4],
                "Execute the conservative cleaning pipeline against the same frozen raw path. "
                "Raw data remains unchanged; the shared processed CSV and audit files are regenerated.",
            ),
            _code(
                """
                cleaning_result = clean_nypd_data.run_pipeline(
                    raw_path=raw_path,
                    processed_path=PROCESSED_PATH,
                    output_dir=OUTPUT_DIR,
                    project_root=PROJECT_ROOT,
                )
                if not PROCESSED_PATH.is_file():
                    raise FileNotFoundError(f"Cleaning pipeline did not create {PROCESSED_PATH}")

                clean_df = pd.read_csv(PROCESSED_PATH, low_memory=False)
                clean_df["ARREST_DATE"] = pd.to_datetime(clean_df["ARREST_DATE"], errors="coerce")

                display(pd.DataFrame([{
                    "raw_rows": len(raw_df),
                    "processed_rows": len(clean_df),
                    "rows_removed": len(raw_df) - len(clean_df),
                    "invalid_or_missing_dates_retained": int(clean_df["ARREST_DATE"].isna().sum()),
                    "processed_columns": clean_df.shape[1],
                    "cleaning_log_exists": (OUTPUT_DIR / "cleaning_log.md").is_file(),
                }]))
                """
            ),
        ]
    )

    cells.extend(
        [
            _markdown(
                SECTION_TITLES[5],
                "Verify that reusable calendar fields were created by the cleaning pipeline and "
                "that their values reconcile with ARREST_DATE.",
            ),
            _code(
                """
                temporal_columns = [
                    "YEAR", "MONTH", "MONTH_NAME", "DAY_OF_WEEK", "DAY_OF_WEEK_NUM"
                ]
                missing_temporal_columns = sorted(set(temporal_columns).difference(clean_df.columns))
                if missing_temporal_columns:
                    raise AssertionError(
                        f"Processed data is missing temporal fields: {missing_temporal_columns}"
                    )

                valid_temporal_rows = clean_df["ARREST_DATE"].notna()
                temporal_checks = {
                    "YEAR_matches": bool((
                        clean_df.loc[valid_temporal_rows, "YEAR"]
                        == clean_df.loc[valid_temporal_rows, "ARREST_DATE"].dt.year
                    ).all()),
                    "MONTH_matches": bool((
                        clean_df.loc[valid_temporal_rows, "MONTH"]
                        == clean_df.loc[valid_temporal_rows, "ARREST_DATE"].dt.month
                    ).all()),
                    "DAY_OF_WEEK_NUM_range": bool(
                        clean_df.loc[valid_temporal_rows, "DAY_OF_WEEK_NUM"].between(1, 7).all()
                    ),
                    "observed_years": ", ".join(
                        map(str, sorted(clean_df["YEAR"].dropna().astype(int).unique()))
                    ),
                    "observed_months": int(clean_df["MONTH"].nunique()),
                }
                if not all(temporal_checks[key] for key in (
                    "YEAR_matches", "MONTH_matches", "DAY_OF_WEEK_NUM_range"
                )):
                    raise AssertionError("Derived temporal fields do not reconcile with ARREST_DATE")
                display(pd.DataFrame([temporal_checks]))
                """
            ),
        ]
    )

    cells.extend(
        [
            _markdown(
                SECTION_TITLES[6],
                "Regenerate the daily, monthly, weekday, and admissible severity summaries, then "
                "reconcile their counts to the processed baseline before using any finding.",
            ),
            _code(
                """
                analysis_result = analyze_temporal.analyze_temporal(
                    processed_csv=PROCESSED_PATH,
                    audit_csv=OUTPUT_DIR / "missingness_summary.csv",
                    metadata_json=metadata_path,
                    outputs_dir=OUTPUT_DIR,
                    figures_dir=FIGURES_DIR,
                )

                daily_arrests = pd.read_csv(OUTPUT_DIR / "daily_arrests.csv", parse_dates=["date"])
                monthly_arrests = pd.read_csv(
                    OUTPUT_DIR / "monthly_arrests.csv", parse_dates=["month_start"]
                )
                weekday_arrests = pd.read_csv(OUTPUT_DIR / "weekday_arrests.csv")

                reconciliations = {
                    "valid_date_rows": int(clean_df["ARREST_DATE"].notna().sum()),
                    "daily_total": int(daily_arrests["arrest_count"].sum()),
                    "monthly_total": int(monthly_arrests["arrest_count"].sum()),
                    "weekday_total": int(weekday_arrests["arrest_count"].sum()),
                }
                if len(set(reconciliations.values())) != 1:
                    raise AssertionError(f"Temporal totals do not reconcile: {reconciliations}")

                observed_max = clean_df["ARREST_DATE"].max()
                expected_partial = observed_max.day != monthrange(observed_max.year, observed_max.month)[1]
                reported_partial = bool(monthly_arrests["is_partial_month"].astype(str).str.lower().isin(
                    ["true", "1", "yes"]
                ).any())
                if expected_partial != reported_partial:
                    raise AssertionError("Reported final-month completeness does not match max ARREST_DATE")

                display(pd.DataFrame([{
                    "processed_rows": len(clean_df),
                    **reconciliations,
                    "months": len(monthly_arrests),
                    "weekdays": len(weekday_arrests),
                    "final_month_is_incomplete": reported_partial,
                }]))
                """
            ),
        ]
    )

    cells.extend(
        [
            _markdown(
                SECTION_TITLES[7],
                "Embed compact links to the validated PNG assets. The notebook remains small; "
                "full-resolution PNG and SVG files stay in figures/part1 for PowerPoint use.",
            ),
            _code(
                """
                figure_specs = [
                    ("Dataset completeness", "missingness_overview.png"),
                    ("Daily arrests and 7-day mean", "daily_arrests_rolling.png"),
                    ("Average daily arrests by month", "monthly_average_daily_arrests.png"),
                    ("Average arrests by weekday", "weekday_average_arrests.png"),
                ]
                optional_severity = FIGURES_DIR / "monthly_severity_composition.png"
                if optional_severity.is_file():
                    figure_specs.append(("Monthly severity composition", optional_severity.name))

                missing_figures = [
                    name for _, name in figure_specs if not (FIGURES_DIR / name).is_file()
                ]
                if missing_figures:
                    raise FileNotFoundError(f"Required figures are missing: {missing_figures}")

                figure_markdown = []
                for label, filename in figure_specs:
                    relative_path = (FIGURES_DIR / filename).relative_to(PROJECT_ROOT).as_posix()
                    figure_markdown.append(f"**{label}**\\n\\n![{label}](../{relative_path})")
                display(Markdown("\\n\\n".join(figure_markdown)))
                """
            ),
        ]
    )

    cells.extend(
        [
            _markdown(
                SECTION_TITLES[8],
                "Read the machine-readable findings, generate the reader-facing handoff, and "
                "show only a bounded inventory rather than dumping nested JSON.",
            ),
            _code(
                """
                key_findings_path = OUTPUT_DIR / "key_findings.json"
                key_findings = json.loads(key_findings_path.read_text(encoding="utf-8"))
                expected_finding_sections = [
                    "source", "analysis_window", "data_quality", "temporal_findings",
                    "severity_analysis", "generated_files"
                ]
                missing_finding_sections = [
                    key for key in expected_finding_sections if key not in key_findings
                ]
                if missing_finding_sections:
                    raise AssertionError(
                        f"key_findings.json is missing sections: {missing_finding_sections}"
                    )

                deliverable_paths = build_deliverables(project_root=PROJECT_ROOT)
                display(pd.DataFrame({
                    "finding_section": expected_finding_sections,
                    "status": ["verified artifact present"] * len(expected_finding_sections),
                }))
                display(pd.DataFrame({
                    "deliverable": list(deliverable_paths),
                    "path": [path.relative_to(PROJECT_ROOT).as_posix() for path in deliverable_paths.values()],
                }))
                """
            ),
        ]
    )

    cells.extend(
        [
            _markdown(
                SECTION_TITLES[9],
                "State the interpretation boundary from executed evidence. Calendar completeness "
                "language is derived from the actual monthly table.",
            ),
            _code(
                """
                limitations = [
                    "NYPD arrest records measure recorded arrest and enforcement activity, not underlying crime incidence or a crime rate.",
                    "The frozen Year-to-Date window cannot establish a long-term trend or recurring seasonality.",
                    "Temporal associations are descriptive; this dataset alone does not identify a cause.",
                    "Counts are not adjusted by population, exposure, mobility, or enforcement deployment.",
                    "Missing demographic, offence, or coordinate values were retained rather than unjustifiably imputed.",
                ]
                retrieval_timestamp = pd.to_datetime(metadata["retrieval_date"])
                latest_record_timestamp = clean_df["ARREST_DATE"].max()
                freshness_gap_days = int(
                    (retrieval_timestamp.normalize() - latest_record_timestamp.normalize()).days
                )
                if freshness_gap_days > 0:
                    limitations.append(
                        f"The latest observed arrest date is {freshness_gap_days} days before retrieval; "
                        "the snapshot is not activity through the retrieval date."
                    )
                incomplete_rows = monthly_arrests.loc[
                    monthly_arrests["is_partial_month"].astype(str).str.lower().isin(["true", "1", "yes"])
                ]
                if not incomplete_rows.empty:
                    incomplete_label = str(incomplete_rows.iloc[0]["month_label"])
                    limitations.append(
                        f"{incomplete_label} is incomplete and must remain explicitly labelled in monthly comparisons."
                    )
                display(Markdown("\\n".join(f"- {item}" for item in limitations)))
                """
            ),
        ]
    )

    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        "ca6002": {
            "part": "Part 1 Data Foundation and Temporal Exploration",
            "generated_by": "src/build_notebook.py",
            "raw_source": "outputs/part1/dataset_snapshot_metadata.json:raw_file",
        },
    }
    validate_notebook(notebook)
    return notebook


def validate_notebook(
    notebook_or_path: NotebookNode | str | Path,
    *,
    require_executed: bool = False,
) -> None:
    """Validate the ten-section contract and optionally require a clean execution."""

    if isinstance(notebook_or_path, (str, Path)):
        with Path(notebook_or_path).open("r", encoding="utf-8") as handle:
            notebook = nbformat.read(handle, as_version=4)
    else:
        notebook = notebook_or_path

    nbformat.validate(notebook)
    headings = [
        cell.source.splitlines()[0].removeprefix("# ")
        for cell in notebook.cells
        if cell.cell_type == "markdown" and cell.source.startswith("# ")
    ]
    if headings != list(SECTION_TITLES):
        raise AssertionError(f"Notebook section contract mismatch: {headings}")

    code_source = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    )
    required_calls = (
        "clean_nypd_data.run_pipeline(",
        "analyze_temporal.analyze_temporal(",
        "build_deliverables(",
    )
    missing_calls = [call for call in required_calls if call not in code_source]
    if missing_calls:
        raise AssertionError(f"Notebook is missing reusable pipeline calls: {missing_calls}")
    if "download_snapshot(" in code_source:
        raise AssertionError("Notebook must not download or replace the frozen snapshot")

    error_outputs = []
    unexecuted = []
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        if require_executed and cell.execution_count is None:
            unexecuted.append(index)
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                error_outputs.append((index, output.get("ename"), output.get("evalue")))
    if error_outputs:
        raise AssertionError(f"Notebook contains error outputs: {error_outputs}")
    if require_executed and unexecuted:
        raise AssertionError(f"Notebook has unexecuted code cells: {unexecuted}")


def build_notebook(
    project_root: str | Path = PROJECT_ROOT,
    output_path: str | Path | None = None,
) -> Path:
    """Write a fresh, output-free notebook and return its absolute path."""

    root = Path(project_root).expanduser().resolve()
    destination = (
        Path(output_path).expanduser().resolve()
        if output_path
        else root / "notebooks" / "01_dataset_temporal.ipynb"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    notebook = _make_notebook()
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        nbformat.write(notebook, handle)
    temporary.replace(destination)
    validate_notebook(destination)
    return destination.resolve()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate notebooks/01_dataset_temporal.ipynb with nbformat."
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    print(build_notebook(project_root=args.project_root, output_path=args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
