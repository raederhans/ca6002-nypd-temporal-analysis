# Part 1 Speaker Notes

### The Frozen NYPD Snapshot Contains 141,870 Recorded Arrest Events

**Finding**

The NYPD Arrest Data (Year to Date) snapshot was retrieved on 22 Aug 2026. It contains 141,870 raw rows and 19 columns covering 01 Jan 2026 through 30 Jun 2026; the pipeline retains 141,870 rows. All 6 observed months are calendar-complete, but the latest observation is 53 days before retrieval.

**Interpretation**

The snapshot provides event dates and classification fields for the team's later temporal, spatial, demographic, and modelling work. The audit identifies where missingness or validation conditions require explicit handling rather than silent imputation.

**Design Rationale**

A horizontal missingness chart uses position and length on a common scale, which supports more accurate comparison than pie slices. A compact fact strip creates visual hierarchy, while direct percentage labels reduce lookup effort. Neutral colours keep attention on material quality issues, and aligned groups separate snapshot scope from field completeness.

**Limitation / Caveat**

These records measure recorded arrest activity, not underlying crime incidence or a population-adjusted crime rate. Despite the source dataset's Year-to-Date name, this frozen snapshot is not activity through the retrieval date. It cannot establish a long-term trend or explain why a pattern occurred.

---

### The 7-Day Arrest Average Reached Its Observed High on 09 May 2026

**Finding**

Across the observed window, the 7-day average ranged from 599.7 arrests on 30 Jan 2026 to 894.6 on 09 May 2026. The largest single-day count was 1,104 on 01 Apr 2026.

**Interpretation**

The series shows when recorded arrest activity was relatively higher or lower inside this snapshot. The smoothed path helps distinguish sustained movement from isolated daily variation, but the chart remains descriptive and does not identify an external driver.

**Design Rationale**

Daily counts appear as a thin, low-opacity grey context line, while the blue 7-day mean receives greater weight and contrast. This hierarchy directs the eye to the stable signal without hiding the underlying observations. Dates remain chronological, gridlines are restrained, and the observed high is annotated directly to avoid a separate legend search.

**Limitation / Caveat**

A trailing rolling mean smooths short-lived variation and is less informative at the boundary of the series. The observed high is not evidence of seasonality, policy effects, individual risk, or a change in crime incidence.

---

### Apr 2026 Had the Highest Average Daily Arrest Activity Among Comparable Months

**Finding**

Among comparable months, Apr 2026 recorded the highest average daily arrest activity at 837.6, while Jun 2026 recorded the lowest at 743.5 arrests per calendar day in scope.

**Interpretation**

Dividing by calendar days in scope makes month lengths comparable and avoids rewarding a 31-day month simply for containing more days. The differences show variation within the observed Year-to-Date period; they do not by themselves demonstrate a recurring seasonal pattern.

**Design Rationale**

Ordered bars use length from a shared zero baseline, a perceptually accurate encoding for comparison. A single blue hue keeps category identity consistent, and the highest comparable value is labelled directly. Month labels follow calendar order rather than sorting by magnitude, preserving the temporal story.

**Limitation / Caveat**

The metric averages recorded arrests over observed calendar days and has no population or exposure denominator. It cannot support a crime-rate statement or a causal explanation for month-to-month differences.

---

### Average Arrest Activity Was Highest on Wednesday

**Finding**

Wednesday had the highest mean at 973.8 arrests per occurrence, compared with 599.1 on Sunday. The Wednesday average was 62.5% higher than the Sunday average within this descriptive metric.

**Interpretation**

Averaging by the number of each weekday observed avoids bias when the snapshot contains unequal counts of Mondays, Tuesdays, or other weekdays. The resulting profile describes timing in recorded arrest activity, not an explanation for people's behaviour or police operations.

**Design Rationale**

Bars use position and length on one baseline, and weekdays remain in the familiar Monday-to-Sunday sequence rather than an alphabetical or rank order. Consistent blue marks support similarity; direct labels reduce legend dependence; light gridlines aid value lookup without competing with the data.

**Limitation / Caveat**

The comparison is not adjusted for population, mobility, events, enforcement deployment, or exposure. It reflects only this Year-to-Date snapshot and cannot establish that weekday alone produced the observed difference or that underlying crime followed the same profile.
