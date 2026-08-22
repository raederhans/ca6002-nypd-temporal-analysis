# Part 1 Validation Report

Overall assessment: **Ready to share**

The checks independently recompute key counts, denominators and temporal aggregates from the frozen snapshot. Visual appearance was also inspected in the exported 16:9 PNG files; the checks below cover file-level integrity.

## Checks

- **PASS — Snapshot metadata exists:** outputs\part1\dataset_snapshot_metadata.json
- **PASS — Frozen raw CSV exists:** data\raw\nypd_arrests_ytd_2026-08-22.csv
- **PASS — API counts reconcile:** before=141870, downloaded=141870, after=141870
- **PASS — Source revision fence is stable:** before=1785165936, after=1785165936, post_download_count=141870
- **PASS — Raw shape matches metadata:** actual=(141870, 19), expected=(141870, 19)
- **PASS — Raw snapshot byte identity:** sha256=5dd32e102e2665a0c0e6eed87d1bb3b918758722494e4c8d42acaf612c528c15
- **PASS — Pagination accounting and stable total order reconcile:** pages=6, offsets=[0, 25000, 50000, 75000, 100000, 125000], rows=[25000, 25000, 25000, 25000, 25000, 16870], order=arrest_key, primary_unique=True, tie_breaker=None
- **PASS — Acquisition duplicate counters reconcile:** exact_beyond_first=0, duplicate_key_rows_beyond_first=0
- **PASS — Data-quality report exists:** outputs\part1\data_quality_report.json
- **PASS — Duplicate audit and conservative cleaning reconcile:** {"raw_exact_duplicates_removed": 0, "raw_duplicate_key_groups": 0, "raw_duplicate_key_rows": 0, "expected_processed_rows": 141870, "evidence_ok": true}
- **PASS — Processed CSV exists:** data\processed\nypd_arrests_clean.csv
- **PASS — Processed shape and retention:** processed=(141870, 24), raw=(141870, 19), exact_duplicates_removed=0
- **PASS — Processed source-field order and values match the exact-deduplicated baseline:** {"ARREST_KEY": true, "ARREST_DATE": true, "PD_CD": true, "PD_DESC": true, "KY_CD": true, "OFNS_DESC": true, "LAW_CODE": true, "LAW_CAT_CD": true, "ARREST_BORO": true, "ARREST_PRECINCT": true, "JURISDICTION_CODE": true, "AGE_GROUP": true, "PERP_SEX": true, "PERP_RACE": true, "X_COORD_CD": true, "Y_COORD_CD": true, "LATITUDE": true, "LONGITUDE": true, "GEOCODED_COLUMN": true}
- **PASS — Processed key order preserves retained frozen-snapshot records:** keys_compared=141,870
- **PASS — Identifier and numeric-coded category text is preserved:** {"ARREST_KEY": true, "PD_CD": true, "KY_CD": true, "ARREST_PRECINCT": true, "JURISDICTION_CODE": true}
- **PASS — Processed dates follow the documented retain-as-NaT policy:** valid=141870, invalid_retained=0, missing_retained=0
- **PASS — Derived temporal fields recompute exactly:** YEAR, MONTH, MONTH_NAME, DAY_OF_WEEK, DAY_OF_WEEK_NUM
- **PASS — Schema profile covers every raw field:** schema_rows=19
- **PASS — Schema null counts recompute:** total_null_cells=893
- **PASS — Effective missingness recomputes for every raw field:** total_effective_missing=951
- **PASS — Final findings and Figure 1 use effective missingness:** source=outputs/part1/missingness_summary.csv, metric=effective_missing, highest={'field': 'LAW_CAT_CD', 'reported_missing_count': 869, 'reported_missing_pct': 0.6125, 'effective_missing_count': 869, 'effective_missing_pct': 0.6125}
- **PASS — Daily table recomputes:** days=181, total=141870
- **PASS — Monthly calendar-day averages recompute:** months=6, partial=0
- **PASS — Weekday order and occurrence averages recompute:** order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
- **PASS — Severity output follows its data-quality admission and denominator:** share_sums={Jan 2026: 99.999999, Feb 2026: 100.000000, Mar 2026: 100.000000, Apr 2026: 99.999999, May 2026: 99.999999, Jun 2026: 100.000000}
- **PASS — All slide figures are readable PNG 1600x900 plus valid SVG:** {"missingness_overview": {"png": "figures/part1/missingness_overview.png", "svg": "figures/part1/missingness_overview.svg", "size": [1600, 900], "extrema": [[23, 255], [49, 255], [58, 255]]}, "daily_arrests_rolling": {"png": "figures/part1/daily_arrests_rolling.png", "svg": "figures/part1/daily_arrests_rolling.svg", "size": [1600, 900], "extrema": [[0, 255], [0, 255], [0, 255]]}, "monthly_average_daily_arrests": {"png": "figures/part1/monthly_average_daily_arrests.png", "svg": "figures/part1/monthly_average_daily_arrests.svg", "size": [1600, 900], "extrema": [[0, 255], [49, 255], [58, 255]]}, "weekday_average_arrests": {"png": "figures/part1/weekday_average_arrests.png", "svg": "figures/part1/weekday_average_arrests.svg", "size": [1600, 900], "extrema": [[0, 255], [49, 255], [58, 255]]}, "monthly_severity_composition": {"png": "figures/part1/monthly_severity_composition.png", "svg": "figures/part1/monthly_severity_composition.svg", "size": [1600, 900], "extrema": [[0, 255], [0, 255], [0, 255]]}}
- **PASS — All slide-ready handoff documents exist:** missing=[]
- **PASS — Slide-ready language avoids unsupported crime/causal claims:** matches=[]
- **PASS — Four slide notes use the required structure and bounded length:** [{"words": 199, "sections_ok": true}, {"words": 178, "sections_ok": true}, {"words": 162, "sections_ok": true}, {"words": 165, "sections_ok": true}]
- **PASS — Handoff states the observed window without a false partial-month warning:** Expected latest date 2026-06-30, retrieval gap 53 days, partial_months=0.
- **PASS — Notebook exists:** notebooks\01_dataset_temporal.ipynb
- **PASS — Notebook structure and required section order:** cells=20, code_cells=10
- **PASS — Notebook executed top-to-bottom without errors:** executed_code_cells=10/10, errors=0, streams=0

## Required caveats

- Recorded arrests reflect police enforcement activity, not the underlying incidence or rate of crime.
- The frozen snapshot was retrieved on 2026-08-22 but contains arrest dates only from 2026-01-01 through 2026-06-30; the 53 day gap means it is not activity through the retrieval date or evidence of a long-term trend.
- Temporal peaks, troughs and category differences are descriptive; no causal explanation is assigned without external evidence.
