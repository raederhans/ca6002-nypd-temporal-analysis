"""Build Part 2 Markdown handoff files from the verified analysis outputs."""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "part2"


def build_deliverables(output_dir: Path = OUTPUT_DIR) -> None:
    meta = json.loads((output_dir / "analysis_metadata.json").read_text())
    borough = pd.read_csv(output_dir / "borough_counts.csv")
    precinct = pd.read_csv(output_dir / "precinct_counts.csv")
    age = pd.read_csv(output_dir / "age_severity_pct.csv", index_col=0)
    delta = pd.read_csv(output_dir / "borough_severity_delta_pp.csv", index_col=0)
    top_borough = borough.iloc[0]; top_precinct = precinct.iloc[0]
    biggest = delta.abs().stack().idxmax(); biggest_value = delta.loc[biggest]
    findings = f"""# Part 2 Key Findings

- The analysis contains **{meta['records']:,} recorded arrests** from **1 Jan to 30 Jun 2026**.
- **{top_borough['borough']}** has the largest raw borough count ({int(top_borough['arrest_records']):,}), but this is not a population-adjusted rate.
- **Precinct {int(top_precinct['precinct'])}** has the highest recorded volume ({int(top_precinct['arrest_records']):,}); the top 10 precincts account for **{meta['top_10_precinct_share_pct']:.1f}%** of records.
- The under-18 group has a **{age.loc['<18','Felony']:.1f}% felony share**, the clearest age-severity contrast in the descriptive profiles.
- The largest borough departure is **{biggest[0]} — {biggest[1]} ({biggest_value:+.2f} percentage points)** relative to the citywide share.
- Association strength is small: Cramér's V is **{meta['borough_cramers_v']:.3f}** for borough × severity and **{meta['age_cramers_v']:.3f}** for age × severity.

These findings describe recorded arrest activity and should not be interpreted as crime incidence or causal effects.
"""
    (output_dir / "key_findings.md").write_text(findings, encoding="utf-8")
    (output_dir / "methodology.md").write_text("""# Part 2 Methodology

## Data interface

The analysis reads `data/processed/nypd_arrests_clean.csv`, the unchanged Part 1 baseline. One row is one recorded arrest. Core fields are `ARREST_BORO`, `ARREST_PRECINCT`, `AGE_GROUP`, `LAW_CAT_CD`, `LATITUDE`, and `LONGITUDE`.

## Quality admission

All borough, age, precinct, and coordinate values pass the documented category/range checks. Core F/M/V analysis uses 140,476 records. The remaining 1,394 missing or non-core law-category values remain in the quality audit but are excluded from severity-composition denominators.

## Analytical choices

- Raw borough and precinct counts answer where records are concentrated; they are not rates.
- Age and borough severity profiles are row-normalised so differently sized groups can be compared.
- Borough heatmap cells report percentage-point difference from the citywide severity mix.
- Cramér's V summarises association strength without treating statistical association as causation.
- The map uses simplified official NYC Department of City Planning police-precinct boundaries, joined using precinct number.
""", encoding="utf-8")
    (output_dir / "limitations.md").write_text("""# Part 2 Limitations

- Arrest records reflect policing, reporting, enforcement, and administrative recording processes; arrest ≠ crime incidence.
- Raw geographic counts are not adjusted for resident population, daytime population, footfall, exposure, or precinct area.
- The snapshot covers six months only (1 Jan–30 Jun 2026) and is not evidence of a long-term trend.
- Descriptive associations cannot establish that age or location causes arrest severity.
- F/M/V composition excludes 1,394 missing or non-core category values; this denominator must be stated.
- Small categories, especially violations and some age groups, may produce unstable percentages.
- Precinct polygons are simplified for display and should not be used for high-precision spatial measurement.
""", encoding="utf-8")
    (output_dir / "slide_plan.md").write_text("""# Part 2 Slide Plan

## Slide B1 — Recorded arrests are spatially concentrated

- Visual: `01_precinct_arrest_map`
- Takeaway: A subset of precincts contains visibly higher recorded arrest volumes.
- Note: Raw counts are not population-adjusted rates.

## Slide B2 — A small group of precincts accounts for substantial volume

- Visual: `02_top_precincts`
- Takeaway: Precinct 75 ranks first; the top 10 precincts account for 27.1% of all records.
- Note: Ranking does not identify underlying crime risk.

## Slide B3 — Age profiles show the clearest severity contrast

- Visual: `03_age_severity_profile`
- Takeaway: Under-18 records have a substantially higher felony share than other age groups.
- Note: Descriptive association does not imply causation.

## Slide B4 — Borough severity differences are modest

- Visual: `04_borough_severity_deviation`
- Takeaway: No borough differs from the citywide mix by more than approximately 2.25 percentage points in any core severity class.
- Note: Cramér's V = 0.064, indicating a small association.
""", encoding="utf-8")
    (output_dir / "team_handoff.md").write_text("""# Part 2 Team Handoff

## Reproduce

From the repository root:

```bash
python src/analyze_spatial_demographic.py
python src/build_part2_deliverables.py
python src/build_part2_notebook.py
python src/validate_part2.py
```

## Files for the presentation team

- `figures/part2/`: four final chart families, each in PNG and SVG.
- `outputs/part2/slide_plan.md`: slide titles, takeaways, and notes.
- `outputs/part2/key_findings.md`: verified numbers and bounded interpretations.

## Files for modelling/evaluation teammates

- Continue using `data/processed/nypd_arrests_clean.csv`; do not use the aggregated Part 2 tables as model input.
- Candidate contextual features supported by EDA include borough, precinct, and age group.
- LAW_CAT_CD should be handled consistently: core F/M/V coverage is 140,476 of 141,870 records.

## Interpretation boundary

Use “recorded arrests” or “arrest activity,” not “crime rate.” Do not assign causal explanations to descriptive patterns.
""", encoding="utf-8")


if __name__ == "__main__":
    build_deliverables(); print(f"Part 2 documents written to {OUTPUT_DIR}")
