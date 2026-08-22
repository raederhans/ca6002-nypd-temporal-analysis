# Part 1 Visual Style Guide

## Language and evidence boundary

- Describe the dataset as **recorded arrest activity**, **arrest events**, or **arrest counts**.
- Do not substitute arrest activity for underlying crime incidence or a crime rate.
- Treat every pattern as descriptive within 01 Jan 2026 to 30 Jun 2026; do not attach an unsupported cause.

## Visual system

- Canvas: white (`#FFFFFF`); primary text: charcoal (`#1F2937`); supporting text: slate (`#4B5563`).
- Primary arrest series and complete-month bars: blue (`#0072B2`).
- Daily background marks: cool grey (`#9CA3AF`) at low opacity; the 7-day mean receives the strongest line weight.
- Reference lines and restrained gridlines: light grey (`#D9DEE3`). Keep only gridlines that support value lookup.
- Use direct labels for highlighted values. Legends are reserved for charts with more than one data series.

## Stable severity mapping

| LAW_CAT_CD display label | Hex | Use |
|---|---:|---|
| Felony | `#0072B2` | Consistent categorical identity |
| Misdemeanor | `#E69F00` | Consistent categorical identity |
| Violation | `#CC79A7` | Consistent categorical identity |
| Other or missing | `#9AA3AA` | Retained, de-emphasised category |

The palette is colourblind-aware and avoids a red-versus-green-only comparison. In monochrome, direct labels, ordering, line weight, and hatch carry meaning in addition to hue.

## Perception and layout rules

- Prefer position and length on a common baseline over angle or area; use bars rather than pie charts.
- Keep chronological axes in chronological order, including Monday through Sunday for weekday categories.
- Use alignment, proximity, and consistent spacing to group titles, annotations, plots, and source notes.
- Use sentence-case, finding-driven titles. Keep slide text short and place detailed caveats in speaker notes.
- Export on a 16:9 canvas with readable presentation typography. Do not use 3D marks, gradients, decorative icons, or rainbow scales.
