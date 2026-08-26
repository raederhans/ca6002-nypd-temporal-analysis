# Part 3 Methodology

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
- Result: `n_estimators=200`, `max_depth=no limit`, `min_samples_leaf=5`, cross-validated macro-F1 **0.4787**.
- The chosen model is then refitted on all 112,380 training records and scored once on the holdout set.

## Evaluation and explainability evidence

- Stratified 80/20 split with `random_state=42`; classification report and confusion matrix on the 28,096-record holdout set.
- Learning curve: five training sizes (10% to 100%) × 3-fold cross-validation, macro-F1.
- Feature importance: mean decrease in impurity, aggregated from one-hot categories back to the eight model inputs.
- Partial dependence: Felony probability over age group, borough and month (the Violation class, at 1.3% of records, produces noisy curves and is omitted).
- Leakage comparison: an identical model additionally given LAW_CODE, PD_CD and OFNS_DESC demonstrates the artificial accuracy gain that motivated their exclusion.
