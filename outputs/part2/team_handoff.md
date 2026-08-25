# Part 2 Team Handoff

## Reproduce

From the repository root:

```bash
python src/run_part2_pipeline.py
```

The one-command pipeline regenerates tables and figures, builds the Markdown handoff files, executes the notebook top-to-bottom, and runs independent validation. Individual stages remain available through the four Part 2 scripts when needed.

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
