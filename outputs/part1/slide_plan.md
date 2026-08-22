# Part 1 Four-Slide Plan

## P1-1 — The Frozen NYPD Snapshot Contains 141,870 Recorded Arrest Events

- **Purpose:** establish source, scope, and analytical reliability.
- **On-slide facts:** NYC Open Data / NYPD; retrieved 22 Aug 2026; 141,870 rows × 19 columns; 01 Jan 2026 to 30 Jun 2026; latest observation is 53 days before retrieval.
- **Key variable groups:** temporal; offence and severity; borough and coordinates; age, sex, and race.
- **Visual:** `figures/part1/missingness_overview.png` with a compact snapshot fact strip.
- **Takeaway:** this is an official, frozen event-level dataset, but it measures recorded arrest activity rather than underlying crime incidence and is not activity through the retrieval date.

## P1-2 — The 7-Day Arrest Average Reached Its Observed High on 09 May 2026

- **Purpose:** show the observed daily path without letting day-to-day noise dominate.
- **On-slide facts:** 7-day mean ranged from 599.7 to 894.6; observed high ended 09 May 2026.
- **Visual:** `figures/part1/daily_arrests_rolling.png`.
- **Takeaway:** describe the timing and size of observed movement only; do not supply an external cause.

## P1-3 — Apr 2026 Had the Highest Average Daily Arrest Activity Among Comparable Months

- **Purpose:** compare months fairly after accounting for calendar days observed.
- **On-slide facts:** Apr 2026 averaged 837.6 arrests per day versus 743.5 in Jun 2026 among comparable months.
- **Visual:** `figures/part1/monthly_average_daily_arrests.png`.
- **Takeaway:** average daily activity is the comparison measure; raw totals are supporting context only.

## P1-4 — Average Arrest Activity Was Highest on Wednesday

- **Purpose:** compare like-for-like weekday occurrences.
- **On-slide facts:** Wednesday averaged 973.8; Sunday averaged 599.1 arrests per occurrence.
- **Visual:** `figures/part1/weekday_average_arrests.png`.
- **Takeaway:** weekday differences are descriptive and do not establish behaviour, exposure, or enforcement causes.
