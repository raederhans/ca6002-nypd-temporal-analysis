# Part 3 Key Findings

- The model is trained on **140,476 recorded arrests** with a core Felony/Misdemeanor/Violation label (112,380 training, 28,096 holdout).
- Severity classes are imbalanced: **Misdemeanor 59.0%**, **Felony 39.6%**, **Violation 1.3%**; a majority-class guess scores **0.590** accuracy.
- The tuned Random Forest scores **0.592 accuracy** on the untouched holdout set and **0.497 macro-F1**, roughly double the majority baseline's **0.247**.
- **Jurisdiction** is the strongest model input (**46.1%** of total impurity decrease).
- The predicted Felony probability declines from the under-18 group to the 65+ group by **0.181** in the partial-dependence analysis.
- Felony recall is **0.622**. Violation recall is **0.765** but precision is only **0.181**: the model flags too many records as Violation.
- Adding charge-encoding fields (LAW_CODE, PD_CD, OFNS_DESC) to the identical pipeline inflates holdout accuracy by **0.405**; those fields are excluded from the reported model.

These findings describe a model of recorded arrest activity and should not be interpreted as crime incidence or future predictions.
