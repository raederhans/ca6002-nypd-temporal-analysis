# Part 4 Validation Report

The purpose of this validation step is to confirm that the Part 4 evaluation figures and conclusions are mathematically consistent with the frozen Part 3 model outputs.

## Validation Summary

- Test-set size: **28,096**
- Validation checks passed: **10/10**

## Checks

- PASS — Confusion matrix has 3 classes
- PASS — Test-set total matches report support
- PASS — Row totals match class support
- PASS — Accuracy matches `model_metrics.json`
- PASS — Balanced accuracy matches `model_metrics.json`
- PASS — Macro F1 matches `model_metrics.json`
- PASS — Per-class precision matches report
- PASS — Per-class recall matches report
- PASS — Per-class F1 matches report
- PASS — Majority baseline matches reported baseline

## Recomputed Metrics

- Accuracy from confusion matrix: **0.591508**
- Reported model accuracy: **0.591508**
- Macro F1 from class-level values: **0.497248**
- Reported Macro F1: **0.497248**
- Balanced accuracy from class recalls: **0.651460**
- Reported balanced accuracy: **0.651460**
- Majority-class baseline from test distribution: **0.590333**
- Reported majority baseline: **0.590333**

All Part 4 evaluation figures are based on these verified values rather than manually entered presentation numbers.

## Consistency Note

This report is aligned with the validation logic in `notebooks/04_model_evaluation_story.ipynb`, which performs the same 10 validation checks and reports:

`All 10 validation checks passed.`
