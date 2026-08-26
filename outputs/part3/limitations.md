# Part 3 Limitations

- Arrest records reflect policing, reporting and enforcement processes; the model describes recorded arrest activity, not crime incidence, and does not predict future events.
- The Violation class is 1.3% of labelled records; recall is 0.765 but precision is 0.181 — balanced weighting makes the model flag Violation liberally, so most Violation predictions are wrong.
- `class_weight="balanced"` trades calibration for minority-class attention; predicted probabilities should not be read as true likelihoods.
- Hyperparameters were tuned on a single split with one 5-fold grid search; no nested cross-validation was run, so the reported holdout score is a single point estimate.
- The snapshot covers six months (Jan–Jun 2026) and is not evidence of longer-term stability; there is no temporal holdout ordering.
- Impurity-based importance favours features with more categories (jurisdiction has 23); importance ranks describe the fitted model, not societal mechanisms.
- Partial dependence summarises the fitted model's association and does not establish a causal relationship.
- The leakage-augmented model exists only to demonstrate inflated accuracy; its score must not be reported as model performance.
