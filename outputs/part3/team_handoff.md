# Part 3 Team Handoff

## Reproduce

From the repository root:

```bash
python src/train_predictive_model.py
python src/build_part3_deliverables.py
python src/build_part3_notebook.py
python src/validate_part3.py
```

or the one-command pipeline `python src/run_part3_pipeline.py`. Training takes roughly 10 minutes on a 12-core machine; `random_state=42` fixes the split and every randomised step.

## Files for the presentation team

- `figures/part3/`: five final chart families, each in PNG and SVG.
- `outputs/part3/slide_plan.md`: slide titles, takeaways, and notes.
- `outputs/part3/key_findings.md`: verified numbers and bounded interpretations.

## Key numbers

- Holdout: accuracy **0.592**, macro-F1 **0.497**, majority baseline **0.590**.
- Tuned parameters: `n_estimators=200`, `max_depth=no limit`, `min_samples_leaf=5`, CV macro-F1 **0.4787**.
- Leakage comparison: +**0.405** accuracy when charge-encoding fields are added — cited only as a warning, never as performance.

## Interpretation boundary

- Use "recorded arrests" or "arrest activity"; never "crime rate".
- The model describes the recording process in Jan–Jun 2026; it does not predict future events.
- Do not attach causal meaning to feature importance or partial dependence.
- Do not cite the leakage-augmented model's score as model performance.
