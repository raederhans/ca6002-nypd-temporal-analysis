# CA6002 Part 1 — NYPD Arrest Data Foundation and Temporal Exploration

This workspace contains the reproducible Part 1 handoff for the CA6002 group assignment. It acquires and freezes the official [NYPD Arrest Data (Year to Date)](https://data.cityofnewyork.us/Public-Safety/NYPD-Arrest-Data-Year-to-Date-/uip8-fykc/about_data) dataset (`uip8-fykc`), audits and conservatively cleans it, then produces temporal summaries, slide-ready figures and four-slide source material.

The unit of analysis is an arrest record. These data describe recorded police enforcement activity, not the underlying incidence or rate of crime.

## Frozen snapshot

| Item | Value |
|---|---:|
| Retrieval date | 2026-08-22 |
| Official API rows | 141,870 |
| Raw columns | 19 |
| Observed arrest dates | 2026-01-01 to 2026-06-30 |
| Processed shape | 141,870 × 24 |
| Latest-record gap at retrieval | 53 days |

The six observed months are complete calendar months. However, the dataset did not contain activity through the 2026-08-22 retrieval date, so this snapshot must not be described as January-to-August coverage.

## Project layout

```text
data/raw/                     immutable dated source snapshot
data/processed/               shared information-rich baseline CSV
notebooks/                    executed reader-facing analysis notebook
src/                          download, audit, analysis and delivery code
figures/part1/                1600×900 PNG plus SVG figures
outputs/part1/                metadata, audit tables, findings and slide handoff
tests/                        focused cleaning and audit edge-case tests
```

The shared downstream dataset is:

```text
data/processed/nypd_arrests_clean.csv
```

It retains all source fields and demographic/offence/location information. It adds `YEAR`, `MONTH`, `MONTH_NAME`, `DAY_OF_WEEK`, and `DAY_OF_WEEK_NUM` (Monday=1 through Sunday=7). It is not one-hot encoded, scaled or split for modelling.

## Reproduce from the frozen raw snapshot

From the project root on Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src\run_part1_pipeline.py
```

The final command reruns cleaning, audit, temporal tables, all figures, Markdown handoff files, the notebook top-to-bottom, and independent validation. It does not contact the API or replace the frozen raw snapshot.

To obtain a new dated official snapshot in the future, explicitly run:

```powershell
.\.venv\Scripts\python.exe src\run_part1_pipeline.py --download
```

If a snapshot for the same retrieval date already exists, the downloader always refuses to overwrite it. A later retrieval uses a new date-stamped raw filename; existing raw snapshots remain untouched. `SOCRATA_APP_TOKEN` is used when present; anonymous official API access is otherwise supported.

## Pipeline stages

Individual stages can also be run separately:

```powershell
.\.venv\Scripts\python.exe src\download_nypd_data.py
.\.venv\Scripts\python.exe src\clean_nypd_data.py
.\.venv\Scripts\python.exe src\analyze_temporal.py
.\.venv\Scripts\python.exe src\build_deliverables.py
.\.venv\Scripts\python.exe src\build_notebook.py
.\.venv\Scripts\python.exe src\validate_part1.py
```

Snapshot metadata records the official API counts before and after pagination, page boundaries, retrieval timestamp, observed date range and raw-file SHA-256 identity. The cleaning log records every retention, transformation and deletion decision; no rows were deleted in this snapshot.

## Primary handoff files

- `data/processed/nypd_arrests_clean.csv` — shared baseline for later project parts
- `notebooks/01_dataset_temporal.ipynb` — executed, reproducible analysis
- `outputs/part1/slide_plan.md` and `slide_notes.md` — four-slide content
- `outputs/part1/team_handoff.md` — downstream data guidance and warnings
- `outputs/part1/validation_report.md` — independent completion evidence
- `figures/part1/` — five final chart families in PNG and SVG

Spatial analysis, demographic interpretation, feature engineering and predictive modelling are intentionally out of scope for this part.
