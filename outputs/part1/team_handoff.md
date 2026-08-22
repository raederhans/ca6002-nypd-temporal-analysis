# Part 1 Team Handoff

## Dataset snapshot

| Item | Verified value |
|---|---|
| Dataset | NYPD Arrest Data (Year to Date) (`uip8-fykc`) |
| Source | NYC Open Data / NYPD |
| Retrieval date | 22 Aug 2026 |
| Raw file | `data/raw/nypd_arrests_ytd_2026-08-22.csv` |
| Raw size | 141,870 rows × 19 columns |
| Processed size | 141,870 rows × 24 columns |
| Arrest-date range | 01 Jan 2026 to 30 Jun 2026 |
| Observed months | 6 |
| Latest-record to retrieval gap | 53 calendar days |
| API expected rows | 141,870; this matches the downloaded count of 141,870. |

## Important quality issues

- Exact duplicate rows detected in raw data: 0.
- Rows associated with duplicated ARREST_KEY values: 0; see the audit evidence before any downstream removal.
- Unparseable or invalid ARREST_DATE values: 0.
- Highest reported field-level missingness: LAW_CAT_CD at 0.61%.
- F/M/V severity codes cover 99.02% (140,476 rows). The remaining 1,394 rows were retained, including 864 true nulls, 99 rows coded `I`, 426 rows coded `9`, 5 source `(null)` sentinels.
- The latest observed arrest date precedes retrieval by 53 days; the snapshot is not activity through 22 Aug 2026.

The complete machine-readable evidence is in `outputs/part1/data_quality_report.json`; missing values were not automatically imputed or discarded.

## Cleaning performed

- Standardised column names, parsed ARREST_DATE, and created the shared temporal fields.
- Trimmed accidental whitespace in string categories without imputing absent values.
- Coerced coordinates to numeric while retaining rows with missing coordinates.
- Removed 0 confirmed exact duplicate rows.
- Date parsing found 0 invalid values; the policy retains them as NaT and removed 0 rows.

The complete action counts and rationales are in `outputs/part1/cleaning_log.md`.

## Derived columns

- `YEAR`
- `MONTH`
- `MONTH_NAME`
- `DAY_OF_WEEK`
- `DAY_OF_WEEK_NUM`

## Files other members should use

- **Shared baseline:** `data/processed/nypd_arrests_clean.csv`
- **Snapshot provenance:** `outputs/part1/dataset_snapshot_metadata.json`
- **Quality evidence:** `outputs/part1/data_quality_report.json`, `schema_summary.csv`, and `missingness_summary.csv`
- **Temporal evidence:** `daily_arrests.csv`, `monthly_arrests.csv`, `weekday_arrests.csv`, and `key_findings.json` under `outputs/part1/`
- **Important source and derived fields:** `ARREST_KEY`, `ARREST_DATE`, `ARREST_BORO`, `LAW_CAT_CD`, `AGE_GROUP`, `PERP_SEX`, `PERP_RACE`, `LATITUDE`, `LONGITUDE`, `YEAR`, `MONTH`, `MONTH_NAME`, `DAY_OF_WEEK`, and `DAY_OF_WEEK_NUM`

## Important warnings

- **Do not interpret arrests as crime incidence or a crime rate.** These records reflect recorded arrest and enforcement activity.
- Do not infer a cause from the descriptive temporal patterns without external evidence.
- Do not impute absent demographic, offence, or coordinate values without a separately documented analytical reason.
- Despite the source dataset's Year-to-Date name, this snapshot ends on 30 Jun 2026 and is not activity through 22 Aug 2026.
- This snapshot is not evidence of a long-term trend.
- All 6 observed months are complete calendar months; this does not make the snapshot complete through the retrieval date.
