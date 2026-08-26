"""Build Part 3 Markdown handoff files from the verified modelling outputs."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "part3"


def build_deliverables(output_dir: Path = OUTPUT_DIR) -> None:
    meta = json.loads((output_dir / "analysis_metadata.json").read_text())
    metrics = json.loads((output_dir / "model_metrics.json").read_text())
    importance = pd.read_csv(output_dir / "feature_importance.csv")
    pdp = pd.read_csv(output_dir / "pdp_values.csv")
    leakage = pd.read_csv(output_dir / "leakage_comparison.csv")
    best = json.loads((output_dir / "best_parameters.json").read_text())

    top_feature = importance.iloc[0]
    age_pdp = pdp[pdp["feature"] == "Age group"].set_index("category")["p_felony"]
    age_spread = age_pdp.loc["<18"] - age_pdp.loc["65+"]
    baseline = meta["majority_baseline_accuracy"]
    contextual = leakage.set_index("model")
    inflation = contextual.loc["Leakage-augmented model", "accuracy"] - contextual.loc["Contextual model", "accuracy"]
    max_depth = best["max_depth"] if best["max_depth"] is not None else "no limit"

    (output_dir / "key_findings.md").write_text(f"""# Part 3 Key Findings

- The model is trained on **{meta['core_fmv_records']:,} recorded arrests** with a core Felony/Misdemeanor/Violation label ({meta['train_size']:,} training, {meta['test_size']:,} holdout).
- Severity classes are imbalanced: **Misdemeanor {meta['class_distribution']['Misdemeanor']['share_pct']:.1f}%**, **Felony {meta['class_distribution']['Felony']['share_pct']:.1f}%**, **Violation {meta['class_distribution']['Violation']['share_pct']:.1f}%**; a majority-class guess scores **{baseline:.3f}** accuracy.
- The tuned Random Forest scores **{metrics['accuracy']:.3f} accuracy** on the untouched holdout set and **{metrics['macro_f1']:.3f} macro-F1**, roughly double the majority baseline's **{contextual.loc['Majority baseline', 'macro_f1']:.3f}**.
- **{top_feature['feature']}** is the strongest model input (**{top_feature['share_pct']:.1f}%** of total impurity decrease).
- The predicted Felony probability declines from the under-18 group to the 65+ group by **{age_spread:.3f}** in the partial-dependence analysis.
- Felony recall is **{metrics['per_class']['Felony']['recall']:.3f}**. Violation recall is **{metrics['per_class']['Violation']['recall']:.3f}** but precision is only **{metrics['per_class']['Violation']['precision']:.3f}**: the model flags too many records as Violation.
- Adding charge-encoding fields (LAW_CODE, PD_CD, OFNS_DESC) to the identical pipeline inflates holdout accuracy by **{inflation:.3f}**; those fields are excluded from the reported model.

These findings describe a model of recorded arrest activity and should not be interpreted as crime incidence or future predictions.
""", encoding="utf-8")

    (output_dir / "methodology.md").write_text(f"""# Part 3 Methodology

## Task and data interface

The model predicts the recorded arrest severity `LAW_CAT_CD` (Felony / Misdemeanor / Violation) from contextual characteristics of the record: borough, precinct, age group, month, day of week, jurisdiction, latitude and longitude. It reads `data/processed/nypd_arrests_clean.csv`, the unchanged Part 1 baseline. The 1,394 records with missing or non-core severity codes are excluded, leaving 140,476 labelled records.

## Leakage-aware feature selection

Fields that directly encode the charge itself are excluded so that accuracy cannot be inflated by the answer leaking into the inputs:

| Candidate field | Included | Reason |
|---|---|---|
| LAW_CODE | No | Directly encodes the charge |
| PD_DESC | No | Directly encodes the charge |
| OFNS_DESC | No | Directly encodes the charge |
| Borough, precinct, age group, month, day of week, jurisdiction, coordinates | Yes | Contextual characteristics of the record |

## Algorithm selection

A Random Forest is used rather than a neural network: most inputs are categorical, trees handle one-hot categories and mixed scales without architecture work, and the fitted forest directly supports the required explainability evidence (impurity-based feature importance, partial dependence, confusion matrix). The default `max_features="sqrt"` is retained.

## Parameter selection and fine tuning

- Parameters: `n_estimators` (200, 400), `max_depth` (no limit, 15, 30), `min_samples_leaf` (1, 5); 12 combinations.
- Tuning: grid search with stratified 5-fold cross-validation on a 30,000-row stratified subsample of the training split; the holdout set is never touched.
- Scoring: **macro-F1**, not accuracy: with a 59.0% majority class, accuracy rewards always predicting Misdemeanor, while macro-F1 requires the rare Violation class to count equally. `class_weight="balanced"` counters the imbalance at fit time.
- Result: `n_estimators={best['n_estimators']}`, `max_depth={max_depth}`, `min_samples_leaf={best['min_samples_leaf']}`, cross-validated macro-F1 **{best['best_cv_f1_macro']:.4f}**.
- The chosen model is then refitted on all {meta['train_size']:,} training records and scored once on the holdout set.

## Evaluation and explainability evidence

- Stratified 80/20 split with `random_state=42`; classification report and confusion matrix on the {meta['test_size']:,}-record holdout set.
- Learning curve: five training sizes (10% to 100%) × 3-fold cross-validation, macro-F1.
- Feature importance: mean decrease in impurity, aggregated from one-hot categories back to the eight model inputs.
- Partial dependence: Felony probability over age group, borough and month (the Violation class, at 1.3% of records, produces noisy curves and is omitted).
- Leakage comparison: an identical model additionally given LAW_CODE, PD_CD and OFNS_DESC demonstrates the artificial accuracy gain that motivated their exclusion.
""", encoding="utf-8")

    (output_dir / "limitations.md").write_text(f"""# Part 3 Limitations

- Arrest records reflect policing, reporting and enforcement processes; the model describes recorded arrest activity, not crime incidence, and does not predict future events.
- The Violation class is 1.3% of labelled records; recall is {metrics['per_class']['Violation']['recall']:.3f} but precision is {metrics['per_class']['Violation']['precision']:.3f} — balanced weighting makes the model flag Violation liberally, so most Violation predictions are wrong.
- `class_weight="balanced"` trades calibration for minority-class attention; predicted probabilities should not be read as true likelihoods.
- Hyperparameters were tuned on a single split with one 5-fold grid search; no nested cross-validation was run, so the reported holdout score is a single point estimate.
- The snapshot covers six months (Jan–Jun 2026) and is not evidence of longer-term stability; there is no temporal holdout ordering.
- Impurity-based importance favours features with more categories (jurisdiction has 23); importance ranks describe the fitted model, not societal mechanisms.
- Partial dependence summarises the fitted model's association and does not establish a causal relationship.
- The leakage-augmented model exists only to demonstrate inflated accuracy; its score must not be reported as model performance.
""", encoding="utf-8")

    (output_dir / "slide_plan.md").write_text(f"""# Part 3 Slide Plan

## Slide C1 — Context alone can predict recorded severity better than guessing

- Visual: `01_confusion_matrix` and `04_learning_curve`
- Takeaway: The Random Forest scores **{metrics['macro_f1']:.3f} macro-F1** on the untouched holdout set — double the majority baseline's **{contextual.loc['Majority baseline', 'macro_f1']:.3f}** — using only contextual characteristics.
- Note: The holdout set was untouched during tuning; validation macro-F1 still improves with more training records.

## Slide C2 — Location and jurisdiction context drive the model's decisions

- Visual: `02_feature_importance`
- Takeaway: **{top_feature['feature']}** contributes **{top_feature['share_pct']:.1f}%** of total impurity decrease; latitude, longitude and precinct follow.
- Note: Importance ranks describe the fitted model, not societal mechanisms.

## Slide C3 — The under-18 group has the highest predicted Felony probability

- Visual: `03_partial_dependence`
- Takeaway: Predicted Felony probability falls by **{age_spread:.3f}** from the under-18 group to the 65+ group; borough and month matter less.
- Note: Partial dependence is a descriptive association of the fitted model.

## Slide C4 — Fields that encode the charge itself inflate accuracy artificially

- Visual: `05_leakage_comparison`
- Takeaway: Adding LAW_CODE, PD_CD and OFNS_DESC to the identical pipeline lifts accuracy by **{inflation:.3f}** — an artificially inflated score, not a better model.
- Note: The reported model excludes those fields; excluding them is a deliberate design choice.

All scores describe recorded arrests in Jan–Jun 2026; none of them are causal claims.
""", encoding="utf-8")

    (output_dir / "team_handoff.md").write_text(f"""# Part 3 Team Handoff

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

- Holdout: accuracy **{metrics['accuracy']:.3f}**, macro-F1 **{metrics['macro_f1']:.3f}**, majority baseline **{baseline:.3f}**.
- Tuned parameters: `n_estimators={best['n_estimators']}`, `max_depth={max_depth}`, `min_samples_leaf={best['min_samples_leaf']}`, CV macro-F1 **{best['best_cv_f1_macro']:.4f}**.
- Leakage comparison: +**{inflation:.3f}** accuracy when charge-encoding fields are added — cited only as a warning, never as performance.

## Interpretation boundary

- Use "recorded arrests" or "arrest activity"; never "crime rate".
- The model describes the recording process in Jan–Jun 2026; it does not predict future events.
- Do not attach causal meaning to feature importance or partial dependence.
- Do not cite the leakage-augmented model's score as model performance.
""", encoding="utf-8")


if __name__ == "__main__":
    build_deliverables()
    print(f"Part 3 documents written to {OUTPUT_DIR}")
