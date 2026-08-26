# Part 3 Slide Plan

## Slide C1 — Context alone can predict recorded severity better than guessing

- Visual: `01_confusion_matrix` and `04_learning_curve`
- Takeaway: The Random Forest scores **0.497 macro-F1** on the untouched holdout set — double the majority baseline's **0.247** — using only contextual characteristics.
- Note: The holdout set was untouched during tuning; validation macro-F1 still improves with more training records.

## Slide C2 — Location and jurisdiction context drive the model's decisions

- Visual: `02_feature_importance`
- Takeaway: **Jurisdiction** contributes **46.1%** of total impurity decrease; latitude, longitude and precinct follow.
- Note: Importance ranks describe the fitted model, not societal mechanisms.

## Slide C3 — The under-18 group has the highest predicted Felony probability

- Visual: `03_partial_dependence`
- Takeaway: Predicted Felony probability falls by **0.181** from the under-18 group to the 65+ group; borough and month matter less.
- Note: Partial dependence is a descriptive association of the fitted model.

## Slide C4 — Fields that encode the charge itself inflate accuracy artificially

- Visual: `05_leakage_comparison`
- Takeaway: Adding LAW_CODE, PD_CD and OFNS_DESC to the identical pipeline lifts accuracy by **0.405** — an artificially inflated score, not a better model.
- Note: The reported model excludes those fields; excluding them is a deliberate design choice.

All scores describe recorded arrests in Jan–Jun 2026; none of them are causal claims.
