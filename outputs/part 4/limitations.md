# Part 4 — Evaluation Limitations

## 1. Severe Class Imbalance

The three target classes are highly imbalanced. Violation represents only a very small proportion of the test set. As a result, overall accuracy can be dominated by the majority Misdemeanor class.

## 2. Accuracy Alone Is Insufficient

The Random Forest accuracy is almost identical to the majority-class baseline. However, Macro F1 is substantially higher. This demonstrates why multiple evaluation metrics are required for this task.

## 3. Violation False Positives

Violation achieves high recall but very low precision. The model identifies many true Violation cases, but it also produces a large number of false-positive Violation predictions.

## 4. Contextual Features Are Incomplete

The model intentionally relies on contextual variables rather than direct offence or charge information. These contextual variables contain predictive signal, but they cannot fully determine legal severity.

## 5. Target Leakage

Adding offence- and charge-related variables produces near-perfect performance. This is treated as a leakage warning rather than a preferred model result because these fields can directly or indirectly reveal the target classification.

## 6. Predictive Association Is Not Causality

Model performance and feature contributions show statistical and predictive associations only. They do not establish causal relationships between location, demographic characteristics, time, jurisdiction and arrest severity.

## 7. Arrest Records Are Not Crime Incidence

The dataset contains recorded NYPD arrests. It reflects observed enforcement activity and should not be interpreted as a direct measure of the true incidence or rate of crime in New York City.
