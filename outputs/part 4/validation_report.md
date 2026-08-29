# Part 4 — Validation Report

The purpose of this validation step is to confirm that the Part 4 evaluation figures and conclusions are mathematically consistent with the frozen Part 3 model outputs.

## Validation Summary

- Test-set size: **28,096**
- Validation checks passed: **6/6**

## Checks

- PASS — Confusion matrix total equals test set size
- PASS — Row totals match class supports
- PASS — Accuracy recomputes correctly
- PASS — Macro F1 recomputes correctly
- PASS — Balanced accuracy recomputes correctly
- PASS — Majority baseline recomputes correctly

## Recomputed Metrics

- Accuracy from confusion matrix: **0.591508**
- Reported model accuracy: **0.591508**
- Macro F1 from class-level values: **0.497248**
- Reported Macro F1: **0.497248**
- Balanced accuracy from class recalls: **0.651460**
- Reported balanced accuracy: **0.651460**
- Majority-class baseline from test distribution: **0.590333**
- Reported majority baseline: **0.590333**

All evaluation figures in Part 4 are based on these verified values rather than manually entered presentation numbers.
