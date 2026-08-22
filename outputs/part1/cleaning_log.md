# NYPD Arrests YTD Cleaning Log

Source snapshot: `data/raw/nypd_arrests_ytd_2026-08-22.csv`
Snapshot date: 2026-08-22

## Row accounting

Original rows: 141870
Exact duplicate rows involved: 0
Exact duplicate rows removed: 0
Non-identical duplicated ARREST_KEY values retained: 0
Rows with non-identical duplicated ARREST_KEY values retained: 0
Final processed rows: 141870

## Transformations and retention decisions

- Column names were standardised to uppercase snake case.
- Only later copies of rows that were identical across every raw field were removed. This avoids double-counting without choosing between conflicting versions of a record.
- Non-identical records sharing an ARREST_KEY were retained for downstream investigation.
- String cells trimmed for surrounding whitespace: 0.
- ARREST_KEY, PD_CD, KY_CD, ARREST_PRECINCT, and JURISDICTION_CODE were preserved as textual identifier/category values; integer codes are written without artificial `.0` suffixes.
- ARREST_DATE was parsed to datetime; invalid supplied values retained as NaT: 0.
- Rows removed for invalid dates: 0.
- Missing ARREST_DATE values retained as NaT: 0.
- Coordinate fields were coerced to numeric. Supplied non-numeric values converted to missing: LATITUDE=0, LONGITUDE=0, X_COORD_CD=0, Y_COORD_CD=0.
- Latitude missing/blank: 0.
- Longitude missing/blank: 0.
- Rows with at least one source coordinate missing/blank: 0.
- Coordinate pairs outside approximate NYC bounds: 0.
- Rows removed for coordinate issues: 0.
- Missing and anomalous coordinates were retained for the spatial analysis owner to filter under an explicit use-specific rule.
- No missing values were imputed.
- LAW_CAT_CD true nulls were retained: 864.
- LAW_CAT_CD source missing sentinels were retained without recoding: {'(null)': 5}.
- LAW_CAT_CD known non-core values were retained: {'I': 99}.
- LAW_CAT_CD unrecognised non-core values were retained for investigation: {'9': 426}.
- Derived temporal columns: YEAR, MONTH, MONTH_NAME, DAY_OF_WEEK, DAY_OF_WEEK_NUM.
- DAY_OF_WEEK_NUM uses Monday=1 through Sunday=7.

## Numeric-code caveat

`ARREST_KEY` is an identifier. `PD_CD`, `KY_CD`, `ARREST_PRECINCT`, and `JURISDICTION_CODE` are numeric-coded categorical/identifier fields. Their numeric means, medians, and variances are not interpreted as quantities.

## Rationale

The shared baseline remains information-rich: invalid or missing field values are documented rather than used as a reason to discard entire arrest records. The frozen raw CSV remains unchanged.
