# Part 1 Chart Contracts

Every chart is descriptive of the frozen snapshot only. Titles and annotations must say **arrest activity**, never imply underlying crime incidence, and never assign a cause.

## Figure 1 — Dataset completeness overview

- Asset: `figures/part1/missingness_overview.png` and matching SVG.
- Source: `outputs/part1/missingness_summary.csv` / `data_quality_report.json`.
- Mark and encoding: horizontal bars; x = missing percentage, y = selected key field, sorted by missing percentage.
- Hierarchy: one accent colour for analytically important missingness; neutral bars elsewhere. Label percentages directly.
- Constraint: show a readable key-field subset rather than a dense screenshot of every column.

## Figure 2 — Daily activity and 7-day mean

- Asset: `figures/part1/daily_arrests_rolling.png` and matching SVG.
- Source: `outputs/part1/daily_arrests.csv` (`date`, `arrest_count`, `rolling_7d_mean`).
- Mark and encoding: thin low-opacity daily line behind a heavier blue 7-day rolling line; x = date, y = arrests.
- Annotation: observed 7-day high of 894.6 on 09 May 2026.
- Constraint: preserve the calendar sequence, include zero-count dates, and use a non-truncated count axis.

## Figure 3 — Average daily arrests by month

- Asset: `figures/part1/monthly_average_daily_arrests.png` and matching SVG.
- Source: `outputs/part1/monthly_arrests.csv` (`month_label`, `avg_arrests_per_calendar_day`, `is_partial_month`).
- Mark and encoding: ordered vertical bars on a common baseline; y = arrests per calendar day in scope.
- Annotation: highest comparable month is Apr 2026 at 837.6 per day.

## Figure 4 — Average arrests by weekday occurrence

- Asset: `figures/part1/weekday_average_arrests.png` and matching SVG.
- Source: `outputs/part1/weekday_arrests.csv` (`weekday_num`, `weekday`, `mean_arrests_per_occurrence`).
- Mark and encoding: ordered bars, Monday through Sunday; y = mean arrests per occurrence of that weekday.
- Annotation: Wednesday is highest at 973.8; Sunday is lowest at 599.1.
- Constraint: never alphabetise weekdays and do not infer a behavioural cause.

## Optional Figure 5 — Monthly severity composition

- Asset when admitted: `figures/part1/monthly_severity_composition.png` and matching SVG.
- Source: `outputs/part1/monthly_severity_composition.csv`.
- Mark and encoding: 100% stacked monthly bars using `share_of_monthly_arrests_pct` and the stable severity mapping in the style guide; y = share of all valid-date arrests that month.
- Admission: Generated because the severity admission gate passed and the validated figure is present.
- Constraint: show classified coverage and retain the explicit Other or missing category. Do not describe severity composition as crime severity.
