# Part 2 Methodology

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
