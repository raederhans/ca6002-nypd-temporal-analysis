# Part 4 — Model Evaluation & Final Story

## 1. Model vs Baseline

The final Random Forest achieved a test accuracy of **59.15%**, compared with a majority-class baseline accuracy of **59.03%**.

This is an improvement of only **0.117 percentage points**, so raw accuracy alone does not show a meaningful advantage over simply predicting the majority class.

However, the Random Forest achieved a **Macro F1 score of 0.497**, compared with approximately **0.247** for the majority baseline. This is an absolute improvement of **0.250**.

This indicates that the contextual model provides substantially better balanced performance across the three severity classes, even though overall accuracy remains almost unchanged.

## 2. Class-Level Performance

The model behaves differently across the three arrest severity classes:

- **Felony** — Precision: **0.534**, Recall: **0.622**, F1: **0.575**
- **Misdemeanor** — Precision: **0.696**, Recall: **0.567**, F1: **0.625**
- **Violation** — Precision: **0.181**, Recall: **0.765**, F1: **0.292**

Misdemeanor has the strongest F1 score. Felony performance is moderate. Violation is the most unusual class: it has relatively high recall but very low precision.

## 3. Confusion Matrix and Error Analysis

The normalised confusion matrix shows that **Felony and Misdemeanor are frequently confused with each other**.

For Violation, the model correctly identifies many true cases, but this comes at the cost of many false positives.

The test set contains only **379 actual Violation records**, while the model predicts **1,605 Violation records**. This means the model predicts approximately **4.2×** as many Violations as actually occur.

This over-prediction explains why Violation recall is high while precision remains very low.

## 4. Leakage Sanity Check

The contextual model intentionally excludes offence- and charge-related variables that could directly reveal the target class.

When leakage-prone features are added, accuracy rises to approximately **99.63%**.

This near-perfect result should not be interpreted as a better model. Instead, it demonstrates that some offence-related variables contain information that directly or indirectly reveals the legal severity classification.

The contextual model is therefore more meaningful for the stated analytical question.

## 5. Final Interpretation

The evaluation suggests that contextual characteristics such as location, age group, time and jurisdiction contain **measurable predictive information** about recorded arrest severity.

However, these variables are **not sufficient to reliably determine** whether an arrest is recorded as Felony, Misdemeanor or Violation.

The main evaluation lesson is that **accuracy alone is misleading in this imbalanced multi-class problem**. Macro F1, class-level metrics and the confusion matrix provide a more complete picture of performance.

These findings describe model behaviour on recorded NYPD arrest data and should not be interpreted as causal relationships or as estimates of the underlying incidence of crime in New York City.
